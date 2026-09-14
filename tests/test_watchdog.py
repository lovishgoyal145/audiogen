"""Unit tests for server/watchdog.py (Ticket 002)."""

from __future__ import annotations

import concurrent.futures
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from server.watchdog import (
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_IDLE_TIMEOUT,
    IdleWatchdog,
)


def test_watchdog_initialization() -> None:
    """Verify default timeout 600s, check interval 1s, and daemon properties."""
    wd = IdleWatchdog()
    assert wd.idle_timeout_seconds == DEFAULT_IDLE_TIMEOUT
    assert wd.check_interval_seconds == DEFAULT_CHECK_INTERVAL
    assert wd.pkill_target == "jupyter"
    assert wd.daemon is True
    assert wd.is_alive is False


def test_watchdog_invalid_parameters_raise_value_error() -> None:
    """Verify negative or zero timeout/interval raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        IdleWatchdog(idle_timeout_seconds=0)

    with pytest.raises(ValueError, match="must be positive"):
        IdleWatchdog(idle_timeout_seconds=-10)

    with pytest.raises(ValueError, match="must be positive"):
        IdleWatchdog(check_interval_seconds=0)

    with pytest.raises(ValueError, match="must be positive"):
        IdleWatchdog(check_interval_seconds=-0.5)


def test_watchdog_get_idle_seconds_remaining() -> None:
    """Initialize watchdog with 100s timeout and assert initial remaining seconds is 100."""
    wd = IdleWatchdog(idle_timeout_seconds=100.0)
    assert wd.get_idle_seconds_remaining() == 100


def test_watchdog_touch_resets_countdown() -> None:
    """Verify remaining seconds decreases over time and touch() resets it."""
    wd = IdleWatchdog(idle_timeout_seconds=50.0)
    initial_remaining = wd.get_idle_seconds_remaining()
    assert initial_remaining == 50

    base_time = time.monotonic()
    # Mock time.monotonic advancing by 10 seconds
    with patch("time.monotonic", side_effect=lambda: base_time + 10.0):
        remaining_after_10s = wd.get_idle_seconds_remaining()
        assert remaining_after_10s == 40

    # Mock time.monotonic at 12 seconds, then call touch()
    with patch("time.monotonic", side_effect=lambda: base_time + 12.0):
        wd.touch()
        # Immediately after touch, remaining should reset back to 50
        assert wd.get_idle_seconds_remaining() == 50


def test_watchdog_touch_thread_safety_under_load() -> None:
    """Verify concurrent thread calls to touch() are safe with zero deadlocks."""
    wd = IdleWatchdog(idle_timeout_seconds=100.0)
    num_threads = 20
    calls_per_thread = 50

    def worker() -> None:
        for _ in range(calls_per_thread):
            wd.touch()
            remaining = wd.get_idle_seconds_remaining()
            assert 0 <= remaining <= 100

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert wd.get_idle_seconds_remaining() == 100


def test_watchdog_triggers_shutdown_action_on_timeout() -> None:
    """Verify background loop triggers shutdown action when timeout expires."""
    mock_shutdown = MagicMock()
    wd = IdleWatchdog(
        idle_timeout_seconds=0.05,
        check_interval_seconds=0.01,
        shutdown_action=mock_shutdown,
    )
    wd.start()
    assert wd.is_alive is True

    try:
        # Wait for timeout to trigger
        time.sleep(0.12)
        mock_shutdown.assert_called_once()
    finally:
        wd.stop()
        assert wd.is_alive is False


def test_watchdog_touch_prevents_shutdown() -> None:
    """Verify periodic touch() keeps countdown refreshed and avoids shutdown."""
    mock_shutdown = MagicMock()
    wd = IdleWatchdog(
        idle_timeout_seconds=0.08,
        check_interval_seconds=0.01,
        shutdown_action=mock_shutdown,
    )
    wd.start()

    try:
        # Touch every 0.02s for 0.12s total (exceeding 0.08s timeout)
        start = time.monotonic()
        while time.monotonic() - start < 0.12:
            wd.touch()
            time.sleep(0.02)

        mock_shutdown.assert_not_called()
    finally:
        wd.stop()


def test_default_shutdown_action_prevents_unauthorized_pkill() -> None:
    """Verify default _default_shutdown_action flushes streams and exits without executing pkill."""
    wd = IdleWatchdog(pkill_target="jupyter", enable_pkill=False)
    with patch("subprocess.run") as mock_run, patch("os._exit") as mock_exit, patch(
        "sys.stdout.flush"
    ) as mock_stdout_flush, patch("sys.stderr.flush") as mock_stderr_flush:
        wd._default_shutdown_action()

        mock_stdout_flush.assert_called_once()
        mock_stderr_flush.assert_called_once()
        mock_run.assert_not_called()
        mock_exit.assert_called_once_with(0)


def test_default_shutdown_action_executes_pkill_when_authorized() -> None:
    """Verify _default_shutdown_action executes pkill only when explicitly authorized."""
    wd = IdleWatchdog(pkill_target="jupyter", enable_pkill=True)
    with patch("subprocess.run") as mock_run, patch("os._exit") as mock_exit, patch(
        "sys.stdout.flush"
    ) as mock_stdout_flush, patch("sys.stderr.flush") as mock_stderr_flush:
        wd._default_shutdown_action()

        mock_stdout_flush.assert_called_once()
        mock_stderr_flush.assert_called_once()
        mock_run.assert_called_once_with(["pkill", "-9", "-f", "jupyter"], check=False)
        mock_exit.assert_called_once_with(0)
