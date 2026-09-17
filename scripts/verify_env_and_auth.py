#!/usr/bin/env python3
"""Automated pre-flight validation and startup check for AudioGen.

Verifies:
1. All required environment variables are present, non-empty, and valid:
   (TUNNEL_REGISTRY_WEBHOOK_URL, TUNNEL_REGISTRY_AUTH_TOKEN, SERVER_BEARER_TOKEN,
    KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_KERNEL_SLUG).
2. Authenticated live health probe against TUNNEL_REGISTRY_WEBHOOK_URL returns HTTP 200 (not 401/403).
3. Local port 17000 is open and not blocked (bindable).

Fails fast with descriptive error messages if any check fails.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    import dotenv
except ImportError:
    dotenv = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_env_and_auth")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

REQUIRED_ENV_VARS: List[str] = [
    "TUNNEL_REGISTRY_WEBHOOK_URL",
    "TUNNEL_REGISTRY_AUTH_TOKEN",
    "SERVER_BEARER_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "KAGGLE_KERNEL_SLUG",
]

DEFAULT_CHECK_PORT: int = 17000
DEFAULT_CHECK_HOST: str = "127.0.0.1"
DEFAULT_PROBE_TIMEOUT: float = 10.0


class PreflightValidationError(RuntimeError):
    """Raised when pre-flight validation or startup check fails."""

    pass


def load_environment(env_file_path: Optional[Path] = None) -> None:
    """Load environment variables from .env file if available."""
    target = env_file_path or (REPO_ROOT / ".env")
    if target.is_file() and dotenv is not None:
        dotenv.load_dotenv(target, override=False)


def validate_env_vars(env_dict: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Verify all required environment variables are present, non-empty, and valid.

    Args:
        env_dict: Optional dictionary of environment variables (defaults to os.environ).

    Returns:
        Dictionary of validated environment variables.

    Raises:
        PreflightValidationError: If any variable is missing, empty, or structurally invalid.
    """
    env = os.environ if env_dict is None else env_dict
    resolved: Dict[str, str] = {}

    for var in REQUIRED_ENV_VARS:
        val = env.get(var)
        if val is None:
            raise PreflightValidationError(
                f"Missing required environment variable: '{var}'. "
                f"Please define it in your .env file or system environment."
            )
        val_str = str(val).strip()
        if not val_str:
            raise PreflightValidationError(
                f"Environment variable '{var}' is empty or whitespace-only. "
                f"A valid non-empty value is required."
            )
        resolved[var] = val_str

    # Specific structural checks
    url = resolved["TUNNEL_REGISTRY_WEBHOOK_URL"]
    parsed_url = urlparse(url)
    if not (parsed_url.scheme in ("http", "https") and parsed_url.netloc):
        raise PreflightValidationError(
            f"Invalid URL structure for TUNNEL_REGISTRY_WEBHOOK_URL: '{url}'. "
            f"Must be a valid http:// or https:// URL with a valid host."
        )

    slug = resolved["KAGGLE_KERNEL_SLUG"]
    if not re.match(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+$", slug):
        raise PreflightValidationError(
            f"Invalid format for KAGGLE_KERNEL_SLUG: '{slug}'. "
            f"Expected format is '<username>/<kernel-name>' (e.g. 'avidok/audiogen')."
        )

    return resolved


def probe_tunnel_registry(
    url: str,
    auth_token: Optional[str] = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    client: Optional[httpx.Client] = None,
) -> bool:
    """Perform live health probe against TUNNEL_REGISTRY_WEBHOOK_URL.

    Sends an authenticated test query with Authorization header if auth_token exists.
    Verifies that HTTP 200 is returned. Fails fast on HTTP 401/403 or network failure.

    Args:
        url: Registry webhook/KV store URL.
        auth_token: Optional bearer token for authentication.
        timeout: HTTP request timeout in seconds.
        client: Optional injected httpx.Client for testing.

    Returns:
        True if probe succeeded with HTTP 200.

    Raises:
        PreflightValidationError: If authentication fails, server returns non-200, or endpoint unreachable.
    """
    headers: Optional[Dict[str, str]] = None
    if auth_token and auth_token.strip():
        headers = {"Authorization": f"Bearer {auth_token.strip()}"}

    def _exec_request(
        c: httpx.Client,
        method: str,
        target_url: str,
    ) -> httpx.Response:
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if headers is not None:
            kwargs["headers"] = headers
        if method.upper() == "GET":
            return c.get(target_url, **kwargs)
        elif method.upper() == "HEAD":
            return c.head(target_url, **kwargs)
        elif method.upper() == "POST":
            return c.post(target_url, **kwargs)
        raise ValueError(f"Unsupported method {method}")

    try:
        def _probe_with_client(c: httpx.Client) -> Tuple[int, str]:
            # Step 1: Probe the configured URL directly
            resp = _exec_request(c, "GET", url)

            # If direct GET returned 200, success!
            if resp.status_code == 200:
                return 200, resp.text

            # If 401 or 403, fail fast immediately (permission / auth error)
            if resp.status_code in (401, 403):
                return resp.status_code, resp.text

            # If endpoint is Upstash Redis REST or similar endpoint returning 400 (EOF / command needed)
            # or 404/405, try /ping endpoint or HEAD request
            if resp.status_code in (400, 404, 405):
                clean_url = url.rstrip("/")
                if not clean_url.endswith("/ping"):
                    ping_url = f"{clean_url}/ping"
                    try:
                        ping_resp = _exec_request(c, "GET", ping_url)
                        if ping_resp.status_code == 200:
                            return 200, ping_resp.text
                        if ping_resp.status_code in (401, 403):
                            return ping_resp.status_code, ping_resp.text
                    except Exception:
                        pass

                try:
                    head_resp = _exec_request(c, "HEAD", url)
                    if head_resp.status_code == 200:
                        return 200, ""
                    if head_resp.status_code in (401, 403):
                        return head_resp.status_code, ""
                except Exception:
                    pass

            return resp.status_code, resp.text

        if client is not None:
            status_code, body = _probe_with_client(client)
        else:
            with httpx.Client(timeout=timeout) as live_client:
                status_code, body = _probe_with_client(live_client)

        if status_code == 200:
            logger.info("Tunnel registry health probe succeeded (HTTP 200).")
            return True
        elif status_code == 401:
            raise PreflightValidationError(
                f"Authentication failed for TUNNEL_REGISTRY_WEBHOOK_URL ({url}): HTTP 401 Unauthorized. "
                f"TUNNEL_REGISTRY_AUTH_TOKEN was rejected by the remote registry coordinator. "
                f"Details: {body.strip()}"
            )
        elif status_code == 403:
            raise PreflightValidationError(
                f"Access forbidden for TUNNEL_REGISTRY_WEBHOOK_URL ({url}): HTTP 403 Forbidden. "
                f"TUNNEL_REGISTRY_AUTH_TOKEN lacks required permissions on the remote coordinator. "
                f"Details: {body.strip()}"
            )
        else:
            raise PreflightValidationError(
                f"Health probe for TUNNEL_REGISTRY_WEBHOOK_URL ({url}) failed with HTTP {status_code}. "
                f"Expected HTTP 200 OK. Details: {body.strip()}"
            )

    except httpx.RequestError as exc:
        raise PreflightValidationError(
            f"Failed to connect to TUNNEL_REGISTRY_WEBHOOK_URL ({url}): {exc}. "
            f"Check your network connection and verify the webhook host is resolvable and reachable."
        ) from exc


def check_port_open(
    port: int = DEFAULT_CHECK_PORT,
    host: str = DEFAULT_CHECK_HOST,
) -> bool:
    """Verify that the local port is open and not blocked (can be bound).

    Args:
        port: Local TCP port number to check (default 17000).
        host: Local IP host to bind (default '127.0.0.1').

    Returns:
        True if port can be bound.

    Raises:
        PreflightValidationError: If port is already occupied or blocked by another process.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            logger.info("Local port %d on %s is open and ready to bind.", port, host)
            return True
        except OSError as exc:
            if exc.errno == 98 or "address already in use" in str(exc).lower():
                raise PreflightValidationError(
                    f"Port conflict detected: Local port {port} on {host} is BLOCKED / already in use "
                    f"by another running process (OSError: {exc}). "
                    f"Please terminate the existing process listening on port {port} before starting the server."
                ) from exc
            elif exc.errno == 13 or "permission denied" in str(exc).lower():
                raise PreflightValidationError(
                    f"Permission denied attempting to bind port {port} on {host}: {exc}. "
                    f"Ensure you have sufficient privileges to bind this port."
                ) from exc
            else:
                raise PreflightValidationError(
                    f"Local port {port} on {host} is not open/bindable: {exc}."
                ) from exc


def run_preflight_checks(
    env_file_path: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
    port: int = DEFAULT_CHECK_PORT,
    host: str = DEFAULT_CHECK_HOST,
) -> bool:
    """Run full pre-flight verification sequence.

    Returns:
        True if all checks passed.

    Raises:
        PreflightValidationError: If any pre-flight check fails.
    """
    logger.info("Starting AudioGen pre-flight verification...")
    load_environment(env_file_path)

    # 1. Environment variables check
    logger.info("Validating required environment variables...")
    validated_vars = validate_env_vars()
    logger.info("All required environment variables present and structurally valid.")

    # 2. Live health probe to tunnel registry
    logger.info("Executing live health probe to TUNNEL_REGISTRY_WEBHOOK_URL...")
    probe_tunnel_registry(
        url=validated_vars["TUNNEL_REGISTRY_WEBHOOK_URL"],
        auth_token=validated_vars.get("TUNNEL_REGISTRY_AUTH_TOKEN"),
        client=client,
    )

    # 3. Port availability check
    logger.info("Checking local port %d availability...", port)
    check_port_open(port=port, host=host)

    logger.info("ALL PRE-FLIGHT CHECKS PASSED. Ready to launch server.")
    return True


def main() -> int:
    """CLI entry point for scripts/verify_env_and_auth.py."""
    try:
        run_preflight_checks()
        return 0
    except PreflightValidationError as exc:
        logger.error("PRE-FLIGHT CHECK FAILED: %s", exc)
        print(f"\n[PRE-FLIGHT FAILURE] {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during pre-flight verification: %s", exc)
        print(f"\n[UNEXPECTED ERROR] {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
