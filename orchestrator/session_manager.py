"""Stateless session manager helpers for Kaggle cloud operations and tunnel health checks.

Per project rules and TICKET-005:
- _kaggle_push() is the single wrapped point of invocation for `kaggle kernels push`.
- kernels status is informational only; readiness is verified strictly by tunnel health.
- No global mutable state lives in this module.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Final, List, Optional, Union
from urllib.parse import urlparse, urlunparse
import httpx

logger = logging.getLogger("audiogen.orchestrator.session_manager")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Final[Path] = REPO_ROOT / "config" / "orchestrator_config.yaml"
DEFAULT_KERNEL_SLUG: Final[str] = "avidok/audiogen"
DEFAULT_STAGE_DIR: Final[Path] = REPO_ROOT / "server"

ENV_REGISTRY_WEBHOOK_URL: Final[str] = "TUNNEL_REGISTRY_WEBHOOK_URL"
ENV_REGISTRY_AUTH_TOKEN: Final[str] = "TUNNEL_REGISTRY_AUTH_TOKEN"
ENV_KAGGLE_KERNEL_SLUG: Final[str] = "KAGGLE_KERNEL_SLUG"

# Verified against Kaggle API kernel enums (QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_*)
TERMINAL_FAILURE_STATUSES: Final[frozenset[str]] = frozenset({
    "ERROR",
    "FAILED",
    "CANCELLED",
    "CANCEL",
    "CANCEL_REQUESTED",
    "CANCEL_ACKNOWLEDGED",
    "COMPLETE",
    "KERNELWORKERSTATUS.ERROR",
    "KERNELWORKERSTATUS.FAILED",
    "KERNELWORKERSTATUS.CANCELLED",
})


class KagglePushError(RuntimeError):
    """Raised when `kaggle kernels push` fails or CLI is missing."""

    pass


class KaggleStatusError(RuntimeError):
    """Raised when `kaggle kernels status` fails or CLI is missing."""

    pass


def _resolve_kernel_slug(kernel_slug: Optional[str] = None) -> str:
    """Resolve kernel slug from argument, environment variable, or default."""
    if kernel_slug and str(kernel_slug).strip():
        return str(kernel_slug).strip()
    env_slug = os.environ.get(ENV_KAGGLE_KERNEL_SLUG)
    if env_slug and env_slug.strip():
        return env_slug.strip()
    return DEFAULT_KERNEL_SLUG


def _kaggle_push(
    kernel_slug: Optional[str] = None,
    stage_dir: Optional[Union[str, Path]] = None,
    kaggle_cmd: Optional[List[str]] = None,
) -> None:
    """Invoke `kaggle kernels push`.

    This function is the single wrapped location in the codebase where
    `kaggle kernels push` is called.

    Args:
        kernel_slug: Optional Kaggle kernel slug (e.g. 'avidok/audiogen').
        stage_dir: Directory containing kernel metadata and notebook to push.
                   Defaults to DEFAULT_STAGE_DIR (server directory).
        kaggle_cmd: Optional binary / command override for Kaggle CLI (e.g. ['kaggle']).

    Raises:
        KagglePushError: If the Kaggle CLI is missing, times out, or fails.
    """
    cmd_base = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]
    executable = cmd_base[0]

    # Verify executable availability
    if not shutil.which(executable) and not Path(executable).exists():
        raise KagglePushError(f"Kaggle CLI executable '{executable}' was not found in PATH.")

    target_dir = Path(stage_dir).resolve() if stage_dir is not None else DEFAULT_STAGE_DIR

    # Ensure kernel-metadata.json is present in the target stage directory
    meta_path = target_dir / "kernel-metadata.json"
    created_temp_meta = False
    if not meta_path.exists():
        config_meta = REPO_ROOT / "config" / "kernel-metadata.json"
        if config_meta.exists():
            try:
                meta_path.symlink_to(config_meta.resolve())
                created_temp_meta = True
            except OSError:
                shutil.copy2(config_meta, meta_path)
                created_temp_meta = True

    push_cmd = [*cmd_base, "kernels", "push", "-p", str(target_dir)]
    logger.info("Running Kaggle push command: %s", " ".join(push_cmd))

    # Stage ephemeral runtime secrets for headless worker if credentials are present
    secrets_file = target_dir / "runtime_secrets.json"
    runtime_secrets: dict[str, str] = {}
    webhook_url = os.environ.get("TUNNEL_REGISTRY_WEBHOOK_URL")
    if webhook_url:
        runtime_secrets["TUNNEL_REGISTRY_WEBHOOK_URL"] = webhook_url
    bearer_token = os.environ.get("SERVER_BEARER_TOKEN") or os.environ.get("SHARED_SECRET")
    if bearer_token:
        runtime_secrets["SERVER_BEARER_TOKEN"] = bearer_token
    auth_token = os.environ.get("TUNNEL_REGISTRY_AUTH_TOKEN")
    if auth_token:
        runtime_secrets["TUNNEL_REGISTRY_AUTH_TOKEN"] = auth_token

    if runtime_secrets:
        try:
            payload = json.dumps(runtime_secrets, indent=2).encode("utf-8")
            fd = os.open(
                str(secrets_file),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            try:
                with open(fd, "wb") as f:
                    f.write(payload)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            logger.info("Staged ephemeral runtime_secrets.json for Kaggle worker push.")
        except OSError as exc:
            logger.warning("Could not write ephemeral runtime_secrets.json: %s", exc)

    try:
        proc = subprocess.run(
            push_cmd,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KagglePushError(f"Kaggle CLI not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise KagglePushError(f"Kaggle push timed out after {exc.timeout}s") from exc
    except Exception as exc:
        raise KagglePushError(f"Failed to execute Kaggle push: {exc}") from exc
    finally:
        if secrets_file.exists():
            try:
                secrets_file.unlink()
            except OSError as exc:
                logger.warning(
                    "Failed to unlink ephemeral runtime_secrets.json at %s: %s",
                    secrets_file,
                    exc,
                )
        if created_temp_meta:
            try:
                if meta_path.is_symlink() or meta_path.exists():
                    meta_path.unlink()
            except OSError as exc:
                logger.warning(
                    "Failed to unlink temporary kernel-metadata.json at %s: %s",
                    meta_path,
                    exc,
                )

    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "").strip()
        raise KagglePushError(
            f"Kaggle push failed with exit code {proc.returncode}: {err_msg}"
        )


def get_kaggle_status(
    kernel_slug: Optional[str] = None,
    kaggle_cmd: Optional[List[str]] = None,
) -> str:
    """Wrap `kaggle kernels status` and return a normalized status string.

    Status is normalized to uppercase (e.g., 'RUNNING', 'COMPLETE', 'ERROR', 'QUEUED').

    Args:
        kernel_slug: Optional kernel slug. Defaults to configured or default slug.
        kaggle_cmd: Optional binary / command override.

    Returns:
        Normalized status string.

    Raises:
        KaggleStatusError: If command fails or CLI is missing.
    """
    cmd_base = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]
    executable = cmd_base[0]

    if not shutil.which(executable) and not Path(executable).exists():
        raise KaggleStatusError(f"Kaggle CLI executable '{executable}' was not found in PATH.")

    slug = _resolve_kernel_slug(kernel_slug)
    status_cmd = [*cmd_base, "kernels", "status", slug]

    try:
        proc = subprocess.run(
            status_cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KaggleStatusError(f"Kaggle CLI not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise KaggleStatusError(f"Kaggle status check timed out after {exc.timeout}s") from exc
    except Exception as exc:
        raise KaggleStatusError(f"Failed to execute Kaggle status command: {exc}") from exc

    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "").strip()
        raise KaggleStatusError(
            f"Kaggle status command failed with exit code {proc.returncode}: {err_msg}"
        )

    output = (proc.stdout or "").strip()
    return parse_kaggle_status_output(output)


def parse_kaggle_status_output(output: str) -> str:
    """Extract and normalize status from Kaggle CLI stdout."""
    # Pattern: '<kernel> has status "running"', "<kernel> has status 'error'",
    # or '<kernel> has status "KernelWorkerStatus.ERROR"'
    match = re.search(r'has\s+status\s+["\']?([a-zA-Z0-9_\.\-]+)["\']?', output, re.IGNORECASE)
    if match:
        raw_status = match.group(1).strip().strip('"').strip("'").upper().strip(".")
        normalized = raw_status.rsplit(".", 1)[-1].strip()
        return normalized or "UNKNOWN"

    # Fallback to direct status keywords if stdout does not follow the standard phrasing
    output_upper = output.upper()
    known_statuses = [
        "RUNNING",
        "QUEUED",
        "COMPLETE",
        "ERROR",
        "FAILED",
        "CANCEL_REQUESTED",
        "CANCEL_ACKNOWLEDGED",
        "CANCELLED",
        "CANCEL",
    ]
    for status in known_statuses:
        if re.search(rf"(?:\b|\.){re.escape(status)}\b", output_upper):
            return status

    # Fallback to cleaned output
    first_line = output.splitlines()[0] if output else ""
    cleaned = first_line.strip().strip('"').strip("'").upper().strip(".")
    if cleaned:
        normalized = cleaned.rsplit(".", 1)[-1].strip()
        return normalized or "UNKNOWN"
    return "UNKNOWN"


def resolve_read_endpoint(endpoint_url: str) -> str:
    """Normalize registry read endpoint to explicit /get/tunnel_url path for Upstash Redis REST.

    Supports:
    - Base URL: https://<id>.upstash.io -> https://<id>.upstash.io/get/tunnel_url
    - Write key path: https://<id>.upstash.io/set/tunnel_url -> https://<id>.upstash.io/get/tunnel_url
    - Read key path: https://<id>.upstash.io/get/tunnel_url -> https://<id>.upstash.io/get/tunnel_url
    - Generic URL with no path -> appends /get/tunnel_url
    """
    clean = endpoint_url.strip().rstrip("/")
    parsed = urlparse(clean)
    path = parsed.path.rstrip("/")
    if path.endswith("/set/tunnel_url"):
        new_path = path[:-len("/set/tunnel_url")] + "/get/tunnel_url"
        return urlunparse(parsed._replace(path=new_path))
    if path.endswith("/get/tunnel_url"):
        return clean
    if "upstash.io" in parsed.netloc.lower() or not path:
        return f"{clean}/get/tunnel_url"
    return clean


def extract_tunnel_url_from_response(resp: httpx.Response) -> Optional[str]:
    """Extract tunnel URL from Upstash {"result": ...}, raw JSON, or plain text.

    Upstash Redis REST returns:
    - {"result": "{\"tunnel_url\": \"https://...\", \"secret\": \"...\"}"} (JSON string in result)
    - {"result": "https://..."} (URL string in result)
    - {"result": {"tunnel_url": "https://..."}} (nested dict in result)
    - {"result": null} (key not found)
    Or direct webhooks returning:
    - {"tunnel_url": "https://...", ...} or {"url": "https://..."}
    - "https://..."
    """
    try:
        data = resp.json()
    except Exception:
        text = resp.text.strip().strip('"').strip("'")
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return None

    if isinstance(data, dict):
        if "result" in data:
            res = data["result"]
            if res is None:
                return None
            if isinstance(res, dict):
                raw_url = res.get("tunnel_url") or res.get("url")
                if isinstance(raw_url, str) and (raw_url.startswith("http://") or raw_url.startswith("https://")):
                    return raw_url.strip()
            elif isinstance(res, str):
                res_str = res.strip()
                if res_str.startswith("{") and res_str.endswith("}"):
                    try:
                        parsed_res = json.loads(res_str)
                        if isinstance(parsed_res, dict):
                            raw_url = parsed_res.get("tunnel_url") or parsed_res.get("url")
                            if isinstance(raw_url, str) and (raw_url.startswith("http://") or raw_url.startswith("https://")):
                                return raw_url.strip()
                    except Exception:
                        pass
                if res_str.startswith("http://") or res_str.startswith("https://"):
                    return res_str
            return None

        raw_url = data.get("tunnel_url") or data.get("url")
        if raw_url and isinstance(raw_url, str):
            val = raw_url.strip()
            if val.startswith("http://") or val.startswith("https://"):
                return val
        return None

    elif isinstance(data, str):
        val = data.strip()
        if val.startswith("http://") or val.startswith("https://"):
            return val
        return None

    raw_text = getattr(resp, "text", None)
    if isinstance(raw_text, str):
        text = raw_text.strip().strip('"').strip("'")
        if text.startswith("http://") or text.startswith("https://"):
            return text
    return None


def is_tunnel_healthy(
    registry_url: Optional[str] = None,
    health_timeout: float = 3.0,
    client: Optional[httpx.Client] = None,
) -> Optional[str]:
    """Check registry/KV for a published tunnel URL and verify its /health endpoint.

    Returns the valid tunnel URL if healthy, None otherwise.
    Never treats registry presence alone as readiness.

    Args:
        registry_url: Webhook or KV registry endpoint URL.
                      Defaults to TUNNEL_REGISTRY_WEBHOOK_URL env var.
        health_timeout: Timeout in seconds for HTTP requests.
        client: Optional injected httpx.Client for testing.

    Returns:
        Healthy tunnel URL string (e.g. 'https://xyz.trycloudflare.com') or None.
    """
    target = registry_url or os.environ.get(ENV_REGISTRY_WEBHOOK_URL)
    if not target or not str(target).strip():
        logger.debug("No registry URL configured for tunnel health check.")
        return None

    target = resolve_read_endpoint(str(target).strip())

    auth_token = os.environ.get(ENV_REGISTRY_AUTH_TOKEN)
    headers = {"Authorization": f"Bearer {auth_token.strip()}"} if auth_token and auth_token.strip() else None

    # Step 1: Query registry for tunnel URL
    tunnel_url: Optional[str] = None
    try:
        if client is not None:
            if headers:
                resp = client.get(target, timeout=health_timeout, headers=headers)
            else:
                resp = client.get(target, timeout=health_timeout)
        else:
            with httpx.Client(timeout=health_timeout) as default_client:
                if headers:
                    resp = default_client.get(target, headers=headers)
                else:
                    resp = default_client.get(target)

        if resp.status_code != 200:
            logger.debug("Registry returned HTTP %d for %s", resp.status_code, target)
            return None

        tunnel_url = extract_tunnel_url_from_response(resp)
    except Exception as exc:
        logger.debug("Failed to query registry at %s: %s", target, exc)
        return None

    if not tunnel_url or not (tunnel_url.startswith("http://") or tunnel_url.startswith("https://")):
        logger.debug("No valid tunnel URL found in registry response.")
        return None

    # Step 2: Validate with real GET /health against the tunnel URL
    health_endpoint = f"{tunnel_url.rstrip('/')}/health"
    try:
        if client is not None:
            health_resp = client.get(health_endpoint, timeout=health_timeout)
        else:
            with httpx.Client(timeout=health_timeout) as default_client:
                health_resp = default_client.get(health_endpoint)

        if health_resp.status_code == 200:
            logger.info("Tunnel at %s is healthy and ready.", tunnel_url)
            return tunnel_url.rstrip('/')

        logger.debug(
            "Tunnel health check at %s returned status %d",
            health_endpoint,
            health_resp.status_code,
        )
        return None
    except Exception as exc:
        logger.debug("Tunnel health check failed for %s: %s", health_endpoint, exc)
        return None


# Backward-compatible alias
kaggle_push = _kaggle_push

__all__ = [
    "ENV_REGISTRY_AUTH_TOKEN",
    "TERMINAL_FAILURE_STATUSES",
    "KagglePushError",
    "KaggleStatusError",
    "_kaggle_push",
    "kaggle_push",
    "get_kaggle_status",
    "parse_kaggle_status_output",
    "is_tunnel_healthy",
    "resolve_read_endpoint",
    "extract_tunnel_url_from_response",
]
