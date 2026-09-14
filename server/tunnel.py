"""Non-blocking Cloudflare quick tunnel launcher and URL extractor."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Final, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PORT: Final[str] = "17000"


def resolve_target_url(target_url: Optional[str] = None) -> str:
    """Resolve the local server target URL, defaulting to port 17000 or env vars PORT/APP_PORT."""
    if target_url is not None and target_url.strip():
        return target_url.strip()
    port = os.environ.get("PORT") or os.environ.get("APP_PORT") or DEFAULT_PORT
    return f"http://localhost:{port.strip()}"


class CloudflaredNotFoundError(FileNotFoundError):
    """Raised when the cloudflared executable is not found on PATH."""

    pass


class TunnelStartupError(RuntimeError):
    """Raised when cloudflared process fails to start or exits prematurely."""

    pass


class TunnelTimeoutError(TimeoutError):
    """Raised when cloudflared does not output a trycloudflare.com URL within the timeout window."""

    pass


class CloudflareTunnel:
    """Manages spawning and lifecycle of a non-blocking cloudflared quick tunnel."""

    TUNNEL_URL_REGEX: Final[Pattern[str]] = re.compile(
        r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
    )

    def __init__(
        self,
        target_url: Optional[str] = None,
        binary_path: Optional[str] = None,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the tunnel configuration.

        Args:
            target_url: Local server address to proxy (default 'http://localhost:17000' or PORT env).
            binary_path: Optional explicit path to cloudflared executable. If None, checks PATH.
            startup_timeout_seconds: Maximum seconds to wait for trycloudflare URL.
        """
        self._target_url = resolve_target_url(target_url)
        self._binary_path = binary_path
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._proc: Optional[subprocess.Popen] = None
        self._tunnel_url: Optional[str] = None

    def start(self) -> str:
        """Spawn cloudflared tunnel non-blocking and extract the public trycloudflare.com URL.

        Returns:
            Extracted public HTTPS URL (e.g. 'https://foo-bar.trycloudflare.com').

        Raises:
            CloudflaredNotFoundError: If cloudflared binary is not found on the system.
            TunnelStartupError: If process exits prematurely with non-zero exit code.
            TunnelTimeoutError: If tunnel URL is not emitted before startup_timeout_seconds.
        """
        if self._binary_path:
            bin_path = shutil.which(self._binary_path) or (
                self._binary_path if Path(self._binary_path).exists() else None
            )
        else:
            bin_path = shutil.which("cloudflared")

        if not bin_path:
            raise CloudflaredNotFoundError(
                "cloudflared executable not found on PATH. Please install cloudflared."
            )

        cmd = [str(bin_path), "tunnel", "--url", self._target_url]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise CloudflaredNotFoundError(
                "cloudflared executable not found on PATH. Please install cloudflared."
            ) from exc
        except Exception as exc:
            raise TunnelStartupError(f"Failed to spawn cloudflared: {exc}") from exc

        output_lines: List[str] = []
        url_found = threading.Event()
        found_urls: List[str] = []

        def _reader(pipe) -> None:
            if pipe is None:
                return
            try:
                for line in iter(pipe.readline, ""):
                    output_lines.append(line)
                    match = self.TUNNEL_URL_REGEX.search(line)
                    if match:
                        found_urls.append(match.group(0))
                        url_found.set()
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_err = threading.Thread(target=_reader, args=(self._proc.stderr,), daemon=True)
        t_out = threading.Thread(target=_reader, args=(self._proc.stdout,), daemon=True)
        t_err.start()
        t_out.start()

        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if url_found.wait(timeout=0.05):
                self._tunnel_url = found_urls[0]
                return self._tunnel_url

            ret = self._proc.poll()
            if ret is not None:
                t_err.join(timeout=0.2)
                t_out.join(timeout=0.2)
                stderr_msg = "".join(output_lines).strip()
                raise TunnelStartupError(
                    f"cloudflared exited unexpectedly with code {ret}: {stderr_msg}"
                )

        self.stop()
        raise TunnelTimeoutError(
            f"Timed out after {self._startup_timeout_seconds}s waiting for Cloudflare tunnel URL."
        )

    def stop(self) -> None:
        """Terminate the running cloudflared child process cleanly."""
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, Exception):
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
                    except Exception:
                        pass
            self._proc = None

    @property
    def tunnel_url(self) -> Optional[str]:
        """Return the extracted tunnel URL, or None if not started."""
        return self._tunnel_url

    @property
    def is_running(self) -> bool:
        """Check if child process is currently alive."""
        return self._proc is not None and self._proc.poll() is None


def start_tunnel(
    target_url: Optional[str] = None,
    binary_path: Optional[str] = None,
    startup_timeout_seconds: float = 30.0,
) -> Tuple[CloudflareTunnel, str]:
    """Convenience helper to initialize, start tunnel, and return (tunnel_instance, tunnel_url)."""
    tunnel = CloudflareTunnel(
        target_url=target_url,
        binary_path=binary_path,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    url = tunnel.start()
    return tunnel, url
