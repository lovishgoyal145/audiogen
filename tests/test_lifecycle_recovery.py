"""Comprehensive acceptance tests for AudioGen session lifecycle, persistence, and recovery.

Covers all 13 required scenarios:
1. Start from completely stopped state.
2. Start when Kaggle GPU is already running (reused, no push).
3. Start when the correct notebook already exists.
4. Start when the notebook is already running.
5. Start twice simultaneously (concurrency lock).
6. Close/reopen browser during startup (no termination, state preserved).
7. Restart local backend while Kaggle remains running (reconciles to READY).
8. Recover after partial initialization (reconnecting to active kernel).
9. Retry after Kaggle timeout (resets error and retries cleanly).
10. Existing SaveKernel 409 reconciles to RUNNING -> READY.
11. Stale persisted state vs actual Kaggle state (reconciles to IDLE).
12. Stop followed immediately by Start (serialized cleanly).
13. Browser reconnect while operation in progress (returns accurate progress).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from orchestrator.gateway import (
    SESSION_RUNTIME_FILE,
    SessionGateway,
    create_app,
    remove_session_runtime_file,
    write_session_runtime_file,
)
import orchestrator.session_manager as session_manager


@pytest.fixture
def recovery_gateway() -> Generator[SessionGateway, None, None]:
    """Isolated SessionGateway with ultra-fast polling for fast, deterministic testing."""
    gw = SessionGateway(
        poll_interval_seconds=0.01,
        startup_timeout_seconds=0.5,
        registry_url="https://mock-registry.example.com",
    )
    gw.reset()
    yield gw
    gw.reset()


@pytest.fixture
def client(recovery_gateway: SessionGateway) -> Generator[TestClient, None, None]:
    """TestClient configured with isolated recovery_gateway."""
    test_app = create_app(gateway=recovery_gateway)
    with TestClient(test_app) as tc:
        yield tc


# ==============================================================================
# 1. Start from completely stopped state
# ==============================================================================
def test_scenario_01_start_from_completely_stopped_state(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 1: Starting from completely stopped state pushes notebook and converges on READY."""
    assert recovery_gateway.state == "IDLE"
    tunnel_url = "https://fresh-tunnel.trycloudflare.com"

    with patch.object(recovery_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, tunnel_url]):
        resp = client.post("/session/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "STARTING"
        assert resp.json()["operation_id"] is not None

        # Wait for background poller to transition to READY
        for _ in range(30):
            time.sleep(0.02)
            st = client.get("/session/status").json()
            if st["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.tunnel_url == tunnel_url
        mock_push.assert_called_once()
        assert recovery_gateway.observed_state.resource_ownership == "local_session"


# ==============================================================================
# 2. Start when Kaggle GPU is already running (reused, no push)
# ==============================================================================
def test_scenario_02_start_when_kaggle_gpu_already_running_reuses_without_push(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 2: Starting when tunnel is already active reuses GPU worker without pushing."""
    active_tunnel = "https://already-active-tunnel.trycloudflare.com"

    disc_result = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.READY,
        tunnel_url=active_tunnel,
        message="Active tunnel discovered in registry",
    )

    with patch.object(recovery_gateway, "_call_check_tunnel_discovery", return_value=disc_result), \
         patch.object(recovery_gateway, "_call_kaggle_push") as mock_push:
        resp = client.post("/session/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "READY"
        assert data["tunnel_url"] == active_tunnel
        assert "reused" in data["message"].lower()

        # No push was made
        mock_push.assert_not_called()
        assert recovery_gateway.observed_state.resource_ownership == "reused_session"


# ==============================================================================
# 3. Start when the correct notebook already exists
# ==============================================================================
def test_scenario_03_start_when_correct_notebook_already_exists(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 3: If kernel status is already RUNNING, startup reconnects rather than failing."""
    target_tunnel = "https://existing-nb-tunnel.trycloudflare.com"

    with patch.object(recovery_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, target_tunnel]):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.tunnel_url == target_tunnel
        assert recovery_gateway.observed_state.kernel_slug == "avidok/audiogen"


# ==============================================================================
# 4. Start when the notebook is already running
# ==============================================================================
def test_scenario_04_start_when_notebook_already_running(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 4: Notebook execution status is RUNNING; gateway awaits tunnel readiness."""
    target_tunnel = "https://active-nb.trycloudflare.com"

    with patch.object(recovery_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, None, target_tunnel]):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.observed_state.kaggle_gpu_state == "RUNNING"
        assert recovery_gateway.observed_state.api_service_readiness == "READY"


# ==============================================================================
# 5. Start twice simultaneously (concurrency lock)
# ==============================================================================
def test_scenario_05_start_twice_simultaneously_concurrency_lock(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 5: Two simultaneous POST /session/start calls only trigger push once."""
    push_call_count = 0

    def slow_push():
        nonlocal push_call_count
        push_call_count += 1
        time.sleep(0.05)

    with patch.object(recovery_gateway, "_call_kaggle_push", side_effect=slow_push), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", return_value=None):
        resp1 = client.post("/session/start")
        resp2 = client.post("/session/start")

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Both return STARTING with the same operation/session
        d1 = resp1.json()
        d2 = resp2.json()
        assert d1["status"] == "STARTING"
        assert d2["status"] == "STARTING"
        assert d1["operation_id"] == d2["operation_id"]

        time.sleep(0.1)
        assert push_call_count == 1


# ==============================================================================
# 6. Close/reopen browser during startup (state preserved)
# ==============================================================================
def test_scenario_06_close_reopen_browser_during_startup(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 6: Browser closure does not terminate session; reopened tab recovers state."""
    target_tunnel = "https://persistent-tunnel.trycloudflare.com"

    with patch.object(recovery_gateway, "_call_kaggle_push"), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, None, target_tunnel]):
        # Browser tab 1 initiates start
        resp = client.post("/session/start")
        assert resp.status_code == 200
        initial_sess_id = resp.json()["session_id"]

        # Tab 1 closes without calling /session/terminate (pagehide/beforeunload hooks removed)
        # Background task continues running.
        time.sleep(0.08)

        # Tab 2 opens and queries /session/status
        status_resp = client.get("/session/status")
        assert status_resp.status_code == 200
        current_data = status_resp.json()
        # Session was NOT terminated
        assert current_data["status"] in ("STARTING", "READY")
        assert current_data["session_id"] == initial_sess_id


# ==============================================================================
# 7. Restart local backend while Kaggle remains running (reconciles to READY)
# ==============================================================================
def test_scenario_07_restart_local_backend_while_kaggle_remains_running(
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 7: Restarting local backend reconciles persisted session state with running Kaggle tunnel."""
    existing_tunnel = "https://surviving-tunnel.trycloudflare.com"

    # Simulate persisted state from prior backend run
    prior_session = {
        "session_id": "audiogen-sess-prior999",
        "desired_state": "READY",
        "status": "READY",
        "tunnel_url": existing_tunnel,
        "kaggle_kernel_slug": "avidok/audiogen",
        "observed_state": {
            "kaggle_gpu_state": "RUNNING",
            "tunnel_url": existing_tunnel,
            "api_service_readiness": "READY",
        },
    }
    write_session_runtime_file(prior_session)

    try:
        # Create a new gateway instance simulating new backend process
        new_gateway = SessionGateway(
            poll_interval_seconds=0.01,
            startup_timeout_seconds=0.5,
            registry_url="https://mock-registry.example.com",
        )

        disc_result = session_manager.TunnelDiscoveryResult(
            status=session_manager.DiscoveryStatus.READY,
            tunnel_url=existing_tunnel,
            message="Active tunnel discovered in registry",
        )

        with patch.object(new_gateway, "_call_check_tunnel_discovery", return_value=disc_result):
            # Load persisted state
            loaded = new_gateway.load_persisted_state()
            assert loaded is True
            assert new_gateway.session_id == "audiogen-sess-prior999"

            # Reconcile with external state
            reconcile_result = asyncio.run(new_gateway.reconcile_with_external_state())
            assert reconcile_result["status"] == "READY"
            assert reconcile_result["tunnel_url"] == existing_tunnel
            assert new_gateway.observed_state.resource_ownership == "reused_session"
    finally:
        remove_session_runtime_file()


# ==============================================================================
# 8. Recover after partial initialization
# ==============================================================================
def test_scenario_08_recover_after_partial_initialization(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 8: Remote kernel is RUNNING but tunnel not ready; gateway enters RECOVERING then reaches READY."""
    target_tunnel = "https://eventual-tunnel.trycloudflare.com"

    disc_not_ready = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.NO_TUNNEL_REGISTERED,
        message="Tunnel key not published yet",
    )
    disc_ready = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.READY,
        tunnel_url=target_tunnel,
    )

    with patch.object(recovery_gateway, "_call_check_tunnel_discovery", side_effect=[disc_not_ready, disc_ready]), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", return_value=target_tunnel):
        # Trigger reconciliation
        resp = client.post("/session/reconcile")
        assert resp.status_code == 200
        # Kernel is running so it adopts and enters RECOVERING while polling
        assert resp.json()["status"] in ("RECOVERING", "READY")

        # Background poller transitions to READY
        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.tunnel_url == target_tunnel


# ==============================================================================
# 9. Retry after Kaggle timeout
# ==============================================================================
def test_scenario_09_retry_after_kaggle_timeout(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 9: After timeout ERROR, user can click retry to reset and successfully start."""
    recovery_gateway.startup_timeout_seconds = 0.05
    target_tunnel = "https://retry-success.trycloudflare.com"

    # Step 1: Let it time out into ERROR
    with patch.object(recovery_gateway, "_call_kaggle_push"), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", return_value=None):
        client.post("/session/start")
        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "ERROR":
                break

        assert recovery_gateway.state == "ERROR"
        assert recovery_gateway.failure_stage == "timeout"

    # Step 2: Retry with working tunnel
    recovery_gateway.startup_timeout_seconds = 1.0
    with patch.object(recovery_gateway, "_call_kaggle_push") as retry_push, \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, target_tunnel]):
        resp = client.post("/session/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "STARTING"

        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.failure_stage is None
        retry_push.assert_called_once()


# ==============================================================================
# 10. Existing SaveKernel 409 reconciles to RUNNING -> READY
# ==============================================================================
def test_scenario_10_savekernel_409_reconciles_to_running_then_ready(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 10: KagglePushConflictError (409) is NOT treated as fatal; reconciles to RUNNING and READY."""
    target_tunnel = "https://409-resolved-tunnel.trycloudflare.com"

    conflict_err = session_manager.KagglePushConflictError(
        "Kaggle push failed with conflict: 409 Client Error: Conflict for url: https://www.kaggle.com/api/v1/SaveKernel"
    )

    with patch.object(recovery_gateway, "_call_kaggle_push", side_effect=conflict_err), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, target_tunnel]):
        resp = client.post("/session/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "STARTING"

        # The poller catches 409, logs that kernel is already running, and awaits tunnel
        for _ in range(40):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.tunnel_url == target_tunnel
        assert recovery_gateway.failure_stage is None
        assert recovery_gateway.observed_state.resource_ownership == "reused_session"


# ==============================================================================
# 11. Stale persisted state vs actual Kaggle state (reconciles to IDLE)
# ==============================================================================
def test_scenario_11_stale_persisted_state_reconciles_to_idle(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 11: Persisted state claimed READY, but remote kernel is terminated and no tunnel."""
    recovery_gateway._state = "READY"
    recovery_gateway._tunnel_url = "https://stale-dead-tunnel.trycloudflare.com"

    disc_dead = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.NO_TUNNEL_REGISTERED,
        message="No tunnel found",
    )

    with patch.object(recovery_gateway, "_call_check_tunnel_discovery", return_value=disc_dead), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="COMPLETE"):
        resp = client.post("/session/reconcile")
        assert resp.status_code == 200
        data = resp.json()
        # Should reconcile to IDLE because remote kernel completed
        assert data["status"] == "IDLE"
        assert recovery_gateway.state == "IDLE"
        assert recovery_gateway.tunnel_url is None


# ==============================================================================
# 12. Stop followed immediately by Start (serialized cleanly)
# ==============================================================================
def test_scenario_12_stop_followed_immediately_by_start(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 12: Stopping an active session followed immediately by start is properly serialized."""
    recovery_gateway._state = "READY"
    recovery_gateway._tunnel_url = "https://stopping-tunnel.trycloudflare.com"
    sess_id = recovery_gateway.session_id
    new_tunnel = "https://new-started-tunnel.trycloudflare.com"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("orchestrator.session_manager.cancel_kaggle_kernel", return_value=True), \
         patch.object(recovery_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", side_effect=[None, new_tunnel]):
        # 1. Stop
        term_resp = client.post("/session/terminate", json={"session_id": sess_id, "reason": "user_restart"})
        assert term_resp.status_code == 200
        assert term_resp.json()["status"] == "SHUTDOWN"

        # 2. Immediately Start
        start_resp = client.post("/session/start")
        assert start_resp.status_code == 200
        assert start_resp.json()["status"] == "STARTING"

        for _ in range(30):
            time.sleep(0.02)
            if client.get("/session/status").json()["status"] == "READY":
                break

        assert recovery_gateway.state == "READY"
        assert recovery_gateway.tunnel_url == new_tunnel
        mock_push.assert_called_once()


# ==============================================================================
# 13. Browser reconnect while operation in progress
# ==============================================================================
def test_scenario_13_browser_reconnect_while_operation_in_progress(
    client: TestClient,
    recovery_gateway: SessionGateway,
) -> None:
    """Scenario 13: Client reconnecting while startup is in progress gets accurate state and operation ID."""
    with patch.object(recovery_gateway, "_call_kaggle_push"), \
         patch.object(recovery_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(recovery_gateway, "_call_is_tunnel_healthy", return_value=None):
        start_resp = client.post("/session/start")
        assert start_resp.status_code == 200
        op_id = start_resp.json()["operation_id"]
        sess_id = start_resp.json()["session_id"]

        # Reconnecting browser tab queries status
        status_resp = client.get(f"/session/status?session_id={sess_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "STARTING"
        assert data["operation_id"] == op_id
        assert data["session_id"] == sess_id
        assert "startup initiated" in data["message"].lower()
