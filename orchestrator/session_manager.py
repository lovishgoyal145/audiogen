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
import random
import re
import shutil
import socket
import struct
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Final, List, Optional, Union
from urllib.parse import urlparse, urlunparse
import httpx

# Install DNS fallback for *.trycloudflare.com if local resolver fails (e.g. ISP DNS blocking)
_orig_getaddrinfo = socket.getaddrinfo


def _query_dns_fallback(
    hostname: str, dns_servers=("1.1.1.1", "8.8.8.8"), timeout: float = 2.0
) -> List[str]:
    query_id = random.randint(0, 65535)
    header = struct.pack("!HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
    qname = (
        b"".join(bytes([len(p)]) + p.encode("ascii") for p in hostname.split("."))
        + b"\x00"
    )
    question = qname + struct.pack("!HH", 1, 1)
    packet = header + question

    for server in dns_servers:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(packet, (server, 53))
            data, _ = sock.recvfrom(1024)
        except Exception:
            continue
        finally:
            sock.close()

        offset = 12 + len(question)
        answers: List[str] = []
        while offset < len(data):
            if data[offset] >= 192:
                offset += 2
            else:
                while data[offset] != 0:
                    offset += 1 + data[offset]
                offset += 1
            if offset + 10 > len(data):
                break
            atype, aclass, ttl, rdlength = struct.unpack("!HHIH", data[offset:offset+10])
            offset += 10
            if atype == 1 and rdlength == 4:
                answers.append(socket.inet_ntoa(data[offset:offset+4]))
            offset += rdlength
        if answers:
            return answers
    return []


def _custom_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror as err:
        h_str = (
            host.decode("ascii", errors="ignore")
            if isinstance(host, bytes)
            else str(host or "")
        )
        if h_str.endswith(".trycloudflare.com"):
            ips = _query_dns_fallback(h_str)
            if ips:
                target_ip = (
                    ips[0].encode("ascii") if isinstance(host, bytes) else ips[0]
                )
                return _orig_getaddrinfo(target_ip, port, *args, **kwargs)
        raise err


socket.getaddrinfo = _custom_getaddrinfo

try:
    from scripts.sync_secrets_dataset import sync_secrets_dataset
except ImportError:
    sync_secrets_dataset = None

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
    secrets_cache_dir: Optional[Path] = None,
    dataset_slug_override: Optional[str] = None,
) -> None:
    """Invoke `kaggle kernels push`.

    This function is the single wrapped location in the codebase where
    `kaggle kernels push` is called.

    Args:
        kernel_slug: Optional Kaggle kernel slug (e.g. 'avidok/audiogen').
        stage_dir: Directory containing kernel metadata and notebook to push.
                   Defaults to DEFAULT_STAGE_DIR (server directory).
        kaggle_cmd: Optional binary / command override for Kaggle CLI (e.g. ['kaggle']).
        secrets_cache_dir: Optional directory for .secrets_dataset_hash.
        dataset_slug_override: Optional pre-synced dataset slug override.

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

    try:
        if dataset_slug_override:
            dataset_slug = dataset_slug_override
        else:
            # Check sync_secrets_dataset availability
            if sync_secrets_dataset is None:
                raise KagglePushError(
                    "Secret dataset synchronization utility (scripts.sync_secrets_dataset) is unavailable."
                )

            # Ensure private Kaggle secret dataset is synced before pushing kernel
            try:
                if secrets_cache_dir is not None:
                    dataset_slug = sync_secrets_dataset(kaggle_cmd=kaggle_cmd, cache_dir=secrets_cache_dir)
                else:
                    dataset_slug = sync_secrets_dataset(kaggle_cmd=kaggle_cmd)
            except Exception as exc:
                logger.error("Failed to synchronize Kaggle secrets dataset: %s", exc)
                raise KagglePushError(f"Secret dataset sync failed prior to push: {exc}") from exc

        resolved_slug = _resolve_kernel_slug(kernel_slug)
        parts = resolved_slug.split("/")
        slug_name = parts[-1] if parts else "audiogen"
        clean_slug_name = re.sub(r"[^a-zA-Z0-9]+", "-", slug_name).strip("-").lower()
        if len(parts) == 2:
            resolved_slug = f"{parts[0]}/{clean_slug_name}"

        if clean_slug_name.startswith("audiogen-"):
            kernel_title = clean_slug_name[:50]
        else:
            kernel_title = f"audiogen-{clean_slug_name}"[:50]

        meta_data = {
            "id": resolved_slug,
            "title": kernel_title,
            "code_file": "interactive_notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_slug] if dataset_slug else [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        config_meta = REPO_ROOT / "config" / "kernel-metadata.json"
        if config_meta.exists():
            try:
                with open(config_meta, "r", encoding="utf-8") as f:
                    tpl = json.load(f)
                if isinstance(tpl, dict):
                    # Only inherit execution hardware / environment flags from template.
                    # Do NOT let template overwrite profile-isolated id, title, or dataset_sources.
                    for flag in ("language", "kernel_type", "is_private", "enable_gpu", "enable_internet"):
                        if flag in tpl:
                            meta_data[flag] = tpl[flag]
            except Exception as exc:
                logger.warning("Could not read template kernel-metadata.json: %s", exc)

        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        created_temp_meta = True

        push_cmd = [*cmd_base, "kernels", "push", "-p", str(target_dir)]
        logger.info("Running Kaggle push command: %s", " ".join(push_cmd))

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

        push_out = (proc.stdout or "").strip()
        push_err = (proc.stderr or "").strip()
        logger.info("Kaggle push result: stdout=%r stderr=%r", push_out, push_err)
        if proc.returncode != 0 or "kernel push error" in push_out.lower():
            err_msg = push_err or push_out
            raise KagglePushError(
                f"Kaggle push failed with exit code {proc.returncode}: {err_msg}"
            )
    finally:
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
    match = re.search(r'has\s+status\s+["\']?([a-zA-Z0-9_\.\-]+)["\']?', output, re.IGNORECASE)
    if match:
        raw_status = match.group(1).strip().strip('"').strip("'").upper().strip(".")
        normalized = raw_status.rsplit(".", 1)[-1].strip()
        return normalized or "UNKNOWN"

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

    first_line = output.splitlines()[0] if output else ""
    cleaned = first_line.strip().strip('"').strip("'").upper().strip(".")
    if cleaned:
        normalized = cleaned.rsplit(".", 1)[-1].strip()
        return normalized or "UNKNOWN"
    return "UNKNOWN"


def get_kaggle_kernel_log(
    kernel_slug: Optional[str] = None,
    kaggle_cmd: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Download and return latest execution log for a kernel."""
    cmd_base = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]
    slug = _resolve_kernel_slug(kernel_slug)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(output_dir) if output_dir else Path(tmp)
        target.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [*cmd_base, "kernels", "output", slug, "-p", str(target)],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            for log_file in target.glob("*.log"):
                try:
                    return log_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        except Exception:
            pass
    return None


def resolve_read_endpoint(endpoint_url: str, profile_id: Optional[str] = None) -> str:
    """Normalize registry read endpoint to explicit /get/tunnel_url path for Upstash Redis REST."""
    clean = endpoint_url.strip().rstrip("/")
    parsed = urlparse(clean)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if profile_id:
        clean_pid = profile_id.strip()
        clean_pid = clean_pid if clean_pid.startswith("profile_") else f"profile_{clean_pid}"
        return f"{base}/get/audiogen:{clean_pid}:tunnel_url"

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
    """Extract tunnel URL from Upstash {"result": ...}, raw JSON, or plain text."""
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


class DiscoveryStatus(str, Enum):
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    REGISTRY_UNREACHABLE = "REGISTRY_UNREACHABLE"
    REGISTRY_AUTH_ERROR = "REGISTRY_AUTH_ERROR"
    NO_TUNNEL_REGISTERED = "NO_TUNNEL_REGISTERED"
    TUNNEL_UNHEALTHY = "TUNNEL_UNHEALTHY"
    READY = "READY"


@dataclass
class TunnelDiscoveryResult:
    status: DiscoveryStatus
    tunnel_url: Optional[str] = None
    message: str = ""
    http_status: Optional[int] = None
    error: Optional[str] = None


def check_tunnel_discovery(
    registry_url: Optional[str] = None,
    health_timeout: float = 3.0,
    client: Optional[httpx.Client] = None,
    profile_id: Optional[str] = None,
) -> TunnelDiscoveryResult:
    """Perform structured tunnel discovery against registry/KV and verify tunnel /health.

    Distinguishes:
    - CONFIGURATION_ERROR: Missing/invalid registry URL.
    - REGISTRY_UNREACHABLE: Network or connection error contacting registry.
    - REGISTRY_AUTH_ERROR: 401 or 403 HTTP response from registry.
    - NO_TUNNEL_REGISTERED: Registry reachable, but tunnel URL key is null/empty.
    - TUNNEL_UNHEALTHY: Tunnel URL discovered, but GET /health returned non-200 or timed out.
    - READY: Tunnel URL discovered and GET /health returned HTTP 200.
    """
    target = registry_url or os.environ.get(ENV_REGISTRY_WEBHOOK_URL)
    if not target or not str(target).strip():
        logger.debug("No registry URL configured for tunnel discovery.")
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.CONFIGURATION_ERROR,
            message="No registry URL configured.",
            error="Missing registry URL",
        )

    target_clean = str(target).strip().rstrip("/")
    parsed = urlparse(target_clean)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        logger.warning("Invalid registry URL structure: %s", target_clean)
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.CONFIGURATION_ERROR,
            message=f"Invalid registry URL structure: '{target_clean}'",
            error=f"Invalid registry URL structure: '{target_clean}'",
        )

    read_target = resolve_read_endpoint(target_clean, profile_id=profile_id)
    auth_token = os.environ.get(ENV_REGISTRY_AUTH_TOKEN)
    headers = (
        {"Authorization": f"Bearer {auth_token.strip()}"}
        if auth_token and auth_token.strip()
        else None
    )

    # Step 1: Query registry for tunnel URL
    try:
        if client is not None:
            if headers:
                resp = client.get(read_target, timeout=health_timeout, headers=headers)
            else:
                resp = client.get(read_target, timeout=health_timeout)
        else:
            with httpx.Client(timeout=health_timeout) as default_client:
                if headers:
                    resp = default_client.get(read_target, headers=headers)
                else:
                    resp = default_client.get(read_target)

        if resp.status_code in (401, 403):
            logger.error(
                "Registry authentication failed with HTTP %d for %s",
                resp.status_code,
                read_target,
            )
            return TunnelDiscoveryResult(
                status=DiscoveryStatus.REGISTRY_AUTH_ERROR,
                message=f"Registry authentication failed with HTTP {resp.status_code}",
                http_status=resp.status_code,
                error=resp.text.strip(),
            )

        if resp.status_code != 200:
            logger.warning(
                "Registry returned HTTP %d for %s", resp.status_code, read_target
            )
            return TunnelDiscoveryResult(
                status=DiscoveryStatus.REGISTRY_UNREACHABLE,
                message=f"Registry returned HTTP {resp.status_code}",
                http_status=resp.status_code,
                error=resp.text.strip(),
            )

        tunnel_url = extract_tunnel_url_from_response(resp)
    except Exception as exc:
        logger.warning("Failed to connect to registry at %s: %s", read_target, exc)
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.REGISTRY_UNREACHABLE,
            message=f"Failed to connect to registry at {read_target}: {exc}",
            error=str(exc),
        )

    if not tunnel_url or not (
        tunnel_url.startswith("http://") or tunnel_url.startswith("https://")
    ):
        logger.info(
            "Registry reachable; worker still initializing (no tunnel registered yet)."
        )
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.NO_TUNNEL_REGISTERED,
            message="No tunnel registered in registry yet.",
            http_status=200,
        )

    # Step 2: Validate with real GET /health against the tunnel URL
    health_endpoint = f"{tunnel_url.rstrip('/')}/health"
    try:
        if client is not None:
            health_resp = client.get(health_endpoint, timeout=health_timeout)
        else:
            with httpx.Client(timeout=health_timeout) as default_client:
                health_resp = default_client.get(health_endpoint)

        if health_resp.status_code == 200:
            logger.info("Tunnel at %s is healthy and ready (HTTP 200).", tunnel_url)
            return TunnelDiscoveryResult(
                status=DiscoveryStatus.READY,
                tunnel_url=tunnel_url.rstrip("/"),
                message=f"Tunnel at {tunnel_url.rstrip('/')} is healthy and ready.",
                http_status=200,
            )

        logger.info(
            "Tunnel URL discovered (%s); awaiting healthy /health response (current status: %d)...",
            tunnel_url,
            health_resp.status_code,
        )
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.TUNNEL_UNHEALTHY,
            tunnel_url=tunnel_url.rstrip("/"),
            message=f"Tunnel /health returned status {health_resp.status_code}",
            http_status=health_resp.status_code,
            error=health_resp.text.strip(),
        )
    except Exception as exc:
        logger.info(
            "Tunnel URL discovered (%s); awaiting healthy /health response (probe failed: %s)...",
            tunnel_url,
            exc,
        )
        return TunnelDiscoveryResult(
            status=DiscoveryStatus.TUNNEL_UNHEALTHY,
            tunnel_url=tunnel_url.rstrip("/"),
            message=f"Tunnel probe failed: {exc}",
            error=str(exc),
        )


def is_tunnel_healthy(
    registry_url: Optional[str] = None,
    health_timeout: float = 3.0,
    client: Optional[httpx.Client] = None,
    profile_id: Optional[str] = None,
) -> Optional[str]:
    """Check registry/KV for a published tunnel URL and verify its /health endpoint.

    Backward-compatible wrapper delegating to check_tunnel_discovery.
    Returns the valid tunnel URL if healthy, None otherwise.
    Never treats registry presence alone as readiness.
    """
    result = check_tunnel_discovery(
        registry_url=registry_url,
        health_timeout=health_timeout,
        client=client,
        profile_id=profile_id,
    )
    if result.status == DiscoveryStatus.READY:
        return result.tunnel_url
    return None


def cancel_kaggle_kernel(
    kernel_slug: Optional[str] = None,
    kaggle_cmd: Optional[List[str]] = None,
    kernel_session_id: Optional[Union[str, int]] = None,
    timeout_seconds: float = 5.0,
) -> bool:
    slug = _resolve_kernel_slug(kernel_slug)
    logger.info("Attempting outside-in cancellation of Kaggle kernel '%s'...", slug)

    if kernel_session_id is None:
        logger.info(
            "No active kernel_session_id provided for kernel '%s'. "
            "get_kernel().metadata.id does not provide an active execution session ID. "
            "Outside-in cancellation skipped; remote IdleWatchdog will reclaim resources.",
            slug,
        )
        return False

    try:
        import kagglesdk
        from kagglesdk.kernels.services.kernels_api_service import ApiCancelKernelSessionRequest

        client = kagglesdk.KaggleClient()
        req_cancel = ApiCancelKernelSessionRequest()
        req_cancel.kernel_session_id = int(kernel_session_id)
        client.kernels.kernels_api_client.cancel_kernel_session(req_cancel)
        logger.info("Kaggle kernel session %s cancelled via kagglesdk.", kernel_session_id)
        return True
    except Exception as exc:
        logger.warning("kagglesdk cancel_kernel_session failed: %s", exc)

    return False


def resolve_delete_endpoint(endpoint_url: str, profile_id: Optional[str] = None) -> str:
    clean = endpoint_url.strip().rstrip("/")
    parsed = urlparse(clean)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if profile_id:
        clean_pid = profile_id.strip()
        clean_pid = clean_pid if clean_pid.startswith("profile_") else f"profile_{clean_pid}"
        return f"{base}/del/audiogen:{clean_pid}:tunnel_url"

    path = parsed.path.rstrip("/")
    if path.endswith("/get/tunnel_url"):
        new_path = path[:-len("/get/tunnel_url")] + "/del/tunnel_url"
        return urlunparse(parsed._replace(path=new_path))
    if path.endswith("/set/tunnel_url"):
        new_path = path[:-len("/set/tunnel_url")] + "/del/tunnel_url"
        return urlunparse(parsed._replace(path=new_path))
    if path.endswith("/del/tunnel_url"):
        return clean
    if "upstash.io" in parsed.netloc.lower() or not path:
        return f"{clean}/del/tunnel_url"
    return clean


def delete_tunnel_url(
    endpoint_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    timeout_seconds: float = 5.0,
    client: Optional[httpx.Client] = None,
    profile_id: Optional[str] = None,
) -> bool:
    target = (
        endpoint_url
        or os.environ.get(ENV_REGISTRY_WEBHOOK_URL)
        or DEFAULT_REGISTRY_URL
    )
    if not target or not str(target).strip():
        logger.warning("No registry URL provided for delete_tunnel_url.")
        return False
    token = auth_token or os.environ.get(ENV_REGISTRY_AUTH_TOKEN)
    headers = (
        {"Authorization": f"Bearer {token.strip()}"}
        if token and token.strip()
        else {}
    )
    del_url = resolve_delete_endpoint(str(target).strip(), profile_id=profile_id)
    try:
        if client is not None:
            resp = client.post(del_url, headers=headers, timeout=timeout_seconds)
            if resp.status_code not in (200, 204):
                resp = client.get(del_url, headers=headers, timeout=timeout_seconds)
        else:
            with httpx.Client(timeout=timeout_seconds) as default_client:
                resp = default_client.post(del_url, headers=headers)
                if resp.status_code not in (200, 204):
                    resp = default_client.get(del_url, headers=headers)
        if resp.status_code in (200, 204):
            logger.info("Successfully deleted tunnel URL from registry at %s", del_url)
            return True
        logger.warning(
            "Registry returned HTTP %d on tunnel deletion: %s",
            resp.status_code,
            resp.text,
        )
        return False
    except Exception as exc:
        logger.warning("Failed to delete tunnel URL from registry (%s): %s", del_url, exc)
        return False


DEFAULT_REGISTRY_URL: Final[str] = "https://large-pup-282364.upstash.io"
kaggle_push = _kaggle_push

__all__ = [
    "ENV_REGISTRY_AUTH_TOKEN",
    "ENV_REGISTRY_WEBHOOK_URL",
    "DEFAULT_REGISTRY_URL",
    "TERMINAL_FAILURE_STATUSES",
    "DiscoveryStatus",
    "TunnelDiscoveryResult",
    "KagglePushError",
    "KaggleStatusError",
    "_kaggle_push",
    "kaggle_push",
    "get_kaggle_status",
    "parse_kaggle_status_output",
    "check_tunnel_discovery",
    "is_tunnel_healthy",
    "resolve_read_endpoint",
    "resolve_delete_endpoint",
    "delete_tunnel_url",
    "cancel_kaggle_kernel",
    "extract_tunnel_url_from_response",
]
