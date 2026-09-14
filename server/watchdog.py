"""Idle watchdog daemon for automatic Kaggle kernel shutdown to prevent GPU quota burn."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Final, Optional

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT: Final[float] = 600.0
DEFAULT_CHECK_INTERVAL: Final[float] = 1.0


class IdleWatchdog:
    """Thread-safe idle watchdog timer running as a daemon thread.

    Tracks server activity via `touch()`. If elapsed time since last touch exceeds
    `idle_timeout_seconds`, flushes logs and triggers kernel termination via `shutdown_action`.
    """

    def __init__(
        self,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT,
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL,
        shutdown_action: Optional[Callable[[], None]] = None,
        pkill_target: str = "jupyter",
        enable_pkill: bool = False,
    ) -> None:
        """Initialize the idle watchdog.

        Args:
            idle_timeout_seconds: Seconds of inactivity before shutdown triggers (default 600).
            check_interval_seconds: Polling frequency in seconds (default 1.0).
            shutdown_action: Optional custom callback to invoke on shutdown. If None,
                             defaults to logging, flushing streams, and os._exit(0).
            pkill_target: Process pattern to target with pkill (default 'jupyter').
            enable_pkill: Whether to allow executing pkill on shutdown (default False).

        Raises:
            ValueError: If idle_timeout_seconds <= 0 or check_interval_seconds <= 0.
        """
        if idle_timeout_seconds <= 0 or check_interval_seconds <= 0:
            raise ValueError("Timeout and interval must be positive numbers.")

        self._idle_timeout_seconds: float = float(idle_timeout_seconds)
        self._check_interval_seconds: float = float(check_interval_seconds)
        self._shutdown_action: Callable[[], None] = shutdown_action or self._default_shutdown_action
        self._pkill_target: str = pkill_target
        self._enable_pkill: bool = bool(
            enable_pkill
            or (os.environ.get("ALLOW_WATCHDOG_PKILL", "").strip().lower() in ("true", "1", "yes"))
        )

        self._lock = threading.Lock()
        self._last_touch: float = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enable_pkill(self) -> bool:
        """Whether pkill process termination is explicitly enabled."""
        return self._enable_pkill

    @property
    def idle_timeout_seconds(self) -> float:
        """Configured idle timeout in seconds."""
        return self._idle_timeout_seconds

    @property
    def check_interval_seconds(self) -> float:
        """Polling check interval in seconds."""
        return self._check_interval_seconds

    @property
    def pkill_target(self) -> str:
        """Target process name for pkill."""
        return self._pkill_target

    @property
    def daemon(self) -> bool:
        """Always runs as a daemon thread."""
        return True

    def touch(self) -> None:
        """Record activity, resetting the idle countdown. Thread-safe."""
        with self._lock:
            self._last_touch = time.monotonic()

    def get_idle_seconds_remaining(self) -> int:
        """Return the number of integer seconds remaining before shutdown triggers. Thread-safe."""
        with self._lock:
            elapsed = time.monotonic() - self._last_touch
            remaining = self._idle_timeout_seconds - elapsed
            return max(0, int(round(remaining)))

    def start(self) -> None:
        """Start the watchdog background daemon thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="IdleWatchdog",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog daemon thread cleanly (used primarily for test cleanup)."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def is_alive(self) -> bool:
        """Return whether the watchdog daemon thread is actively running."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Internal daemon loop checking activity at check_interval_seconds intervals."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._check_interval_seconds):
                break
            with self._lock:
                elapsed = time.monotonic() - self._last_touch
                timed_out = elapsed >= self._idle_timeout_seconds

            if timed_out:
                logger.warning(
                    "Idle watchdog timeout reached (%.2fs >= %.2fs). Triggering shutdown action.",
                    elapsed,
                    self._idle_timeout_seconds,
                )
                try:
                    self._shutdown_action()
                except Exception as exc:
                    logger.error("Error executing shutdown action: %s", exc, exc_info=True)
                break

    def _default_shutdown_action(self) -> None:
        """Execute default shutdown routine: log reason, flush logs, optional pkill, os._exit(0)."""
        logger.warning(
            "Executing default shutdown action: pkill_target=%s (pkill_enabled=%s), terminating with os._exit(0)",
            self._pkill_target,
            self._enable_pkill,
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            for handler in logging.getLogger().handlers:
                try:
                    handler.flush()
                except Exception:
                    pass
        except Exception:
            pass

        if self._enable_pkill:
            logger.warning("Executing authorized pkill -9 -f %s", self._pkill_target)
            try:
                subprocess.run(["pkill", "-9", "-f", self._pkill_target], check=False)
            except Exception as exc:
                logger.error("Failed to execute pkill: %s", exc)
        else:
            logger.info("Pkill execution is disabled by default to prevent unauthorized process termination.")

        os._exit(0)
