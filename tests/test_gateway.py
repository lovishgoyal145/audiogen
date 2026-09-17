"""Integration and unit tests for orchestrator/gateway.py and API endpoints."""

from __future__ import annotations

import time
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from starlette.testclient import TestClient

from orchestrator.gateway import SessionGateway, app, create_app
import orchestrator.session_manager as session_manager
import voices.registry


@pytest.fixture
def test_gateway() -> SessionGateway:
    """Provide an isolated SessionGateway instance configured for rapid testing."""
    return SessionGateway(
        poll_interval_seconds=0.01,
        startup_timeout_seconds=0.1,
    )


@pytest.fixture
def client(test_gateway: SessionGateway) -> Generator[TestClient, None, None]:
    """TestClient configured with test_gateway."""
    test_app = create_app(gateway=test_gateway)
    with TestClient(test_app) as tc:
        yield tc


# ==============================================================================
# 1. State Machine & Session Startup Tests
# ==============================================================================


def test_initial_state_is_idle(client: TestClient) -> None:
    """Test gateway starts in IDLE state with informative status message."""
    resp = client.get("/session/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "IDLE"
    assert data["message"] == "Idle"


def test_start_session_from_idle_calls_push_once_and_transitions_to_starting(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    POST /session/start from IDLE calls _kaggle_push() exactly once (mock-verified)
    and transitions to STARTING.
    """
    with patch.object(test_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        resp = client.post("/session/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "STARTING"
        assert "startup initiated" in data["message"]
        # Allow background task to execute
        for _ in range(20):
            if mock_push.called:
                break
            time.sleep(0.01)
        mock_push.assert_called_once()
        assert test_gateway.state == "STARTING"


def test_start_session_while_already_starting_is_noop(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    POST /session/start while already STARTING does not call _kaggle_push() again
    (mock call count still 1) — test this explicitly with two calls in sequence
    before the first completes.
    """
    with patch.object(test_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        # First call transitions to STARTING
        resp1 = client.post("/session/start")
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "STARTING"
        for _ in range(20):
            if mock_push.called:
                break
            time.sleep(0.01)
        assert mock_push.call_count == 1

        # Second call while already STARTING
        resp2 = client.post("/session/start")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "STARTING"
        time.sleep(0.02)
        # Mock push must still have been called exactly once
        assert mock_push.call_count == 1


def test_start_session_while_already_ready_is_noop(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    POST /session/start while already READY returns READY immediately without calling _kaggle_push().
    """
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://active.trycloudflare.com"

    with patch.object(test_gateway, "_call_kaggle_push") as mock_push:
        resp = client.post("/session/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "READY"
        mock_push.assert_not_called()


def test_polling_transitions_to_error_immediately_on_terminal_failure(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    Background polling transitions to ERROR immediately (not after waiting out the timeout)
    when get_kaggle_status() returns a terminal failure status — verify via mocked fast polling,
    no real multi-minute test waits.
    """
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 10.0  # long timeout, must NOT wait it out

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="COMPLETE"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        start_time = time.time()
        resp = client.post("/session/start")
        assert resp.status_code == 200

        # Wait briefly for background task iteration
        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Must have transitioned fast, not waiting for 10.0s
        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert "terminal failure" in data["message"]


def test_polling_transitions_to_ready_the_moment_tunnel_healthy(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    Background polling transitions to READY the moment is_tunnel_healthy() returns a URL.
    """
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 10.0
    tunnel_target = "https://ready-tunnel.trycloudflare.com"

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=tunnel_target):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "READY":
                break

        data = client.get("/session/status").json()
        assert data["status"] == "READY"
        assert test_gateway.tunnel_url == tunnel_target


def test_polling_timeout_results_in_error(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    A timeout with no terminal failure and no healthy tunnel results in ERROR
    after the configured timeout (mocked, fast).
    """
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 0.05  # Fast 50ms timeout

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert "timed out" in data["message"]


def test_retry_after_error_resets_and_retries(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    After ERROR, a subsequent POST /session/start successfully resets and retries
    (verify _kaggle_push() gets called again this time).
    """
    test_gateway._state = "ERROR"
    test_gateway._status_message = "Previous failure"

    with patch.object(test_gateway, "_call_kaggle_push") as mock_push, \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        resp = client.post("/session/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "STARTING"
        for _ in range(20):
            if mock_push.called:
                break
            time.sleep(0.01)
        mock_push.assert_called_once()
        assert test_gateway.state == "STARTING"


def test_polling_transitions_to_error_immediately_on_kaggle_status_error(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify poller transitions to ERROR immediately on non-transient KaggleStatusError."""
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 10.0

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(
             test_gateway,
             "_call_get_kaggle_status",
             side_effect=session_manager.KaggleStatusError("403 - Forbidden: Invalid credentials"),
         ), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value=None):
        start_time = time.time()
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Must fail fast, not wait 10s
        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert "403 - Forbidden" in data["message"]


def test_background_push_failure_transitions_to_error(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify background push failure transitions state to ERROR."""
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 10.0

    with patch.object(
        test_gateway,
        "_call_kaggle_push",
        side_effect=session_manager.KagglePushError("Push failed: missing CLI"),
    ):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert "Kaggle push failed" in data["message"]


# ==============================================================================
# 2. Pre-Ready Guard Gate & Forwarding (/generate)
# ==============================================================================


@pytest.mark.parametrize("non_ready_state", ["IDLE", "STARTING", "ERROR"])
def test_generate_returns_409_when_state_not_ready(
    client: TestClient,
    test_gateway: SessionGateway,
    non_ready_state: str,
) -> None:
    """Acceptance criterion:

    POST /generate returns HTTP 409 with the specified body shape when state is
    IDLE, STARTING, or ERROR.
    Expected shape: {"error": "session not ready", "status": "<current_state>"}
    """
    test_gateway._state = non_ready_state
    test_gateway._status_message = f"In {non_ready_state} state"

    payload = {
        "text": "नमस्ते दुनिया",
        "language": "hi",
        "speaker_ref_name": "anchor_female_calm",
    }
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 409
    assert resp.headers["content-type"] == "application/json"
    data = resp.json()
    assert data == {
        "error": "session not ready",
        "status": non_ready_state,
    }


def test_generate_forwards_correctly_when_ready(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Acceptance criterion:

    POST /generate forwards correctly (mocked) and returns the Kaggle server's response
    when state is READY.
    """
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"
    test_gateway.bearer_token = "secret-bearer-123"

    payload = {
        "text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
        "language": "pa",
        "speaker_ref_name": "anchor_male_energetic",
    }

    mock_audio_bytes = b"RIFFmockwavdata12345678"

    async def mock_post(url: str, json: dict, headers: dict, **kwargs):
        assert url == "https://tunnel.trycloudflare.com/generate"
        assert json == payload
        assert headers.get("Authorization") == "Bearer secret-bearer-123"
        return httpx.Response(
            status_code=200,
            content=mock_audio_bytes,
            headers={"content-type": "audio/wav"},
        )

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 200
        assert resp.content == mock_audio_bytes
        assert resp.headers["content-type"] == "audio/wav"


def test_generate_forwards_remote_error_when_ready(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Test remote 400 Bad Request is returned to client as-is."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"

    payload = {
        "text": "bad request text",
        "language": "hi",
        "speaker_ref_name": "invalid_voice",
    }

    async def mock_post(url: str, json: dict, headers: dict, **kwargs):
        return httpx.Response(
            status_code=400,
            json={"detail": "Unknown voice name: invalid_voice"},
            headers={"content-type": "application/json"},
        )

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 400
        assert resp.json() == {"detail": "Unknown voice name: invalid_voice"}


# ==============================================================================
# 3. Voice Listing & UI Route Tests
# ==============================================================================


def test_get_voices_filtered(client: TestClient) -> None:
    """Acceptance criterion:

    GET /voices?language=hi and ?language=pa return correctly filtered results
    from a mocked voices.registry.list_voices.
    """
    with patch("voices.registry.list_voices") as mock_list:
        mock_list.side_effect = lambda lang: (
            ["anchor_female_calm", "anchor_male_energetic"]
            if lang == "hi"
            else ["punjabi_voice_1"]
            if lang == "pa"
            else ["voice_all"]
        )

        resp_hi = client.get("/voices?language=hi")
        assert resp_hi.status_code == 200
        assert resp_hi.json() == {"voices": ["anchor_female_calm", "anchor_male_energetic"]}
        mock_list.assert_called_with("hi")

        resp_pa = client.get("/voices?language=pa")
        assert resp_pa.status_code == 200
        assert resp_pa.json() == {"voices": ["punjabi_voice_1"]}
        mock_list.assert_called_with("pa")


def test_get_voices_all(client: TestClient) -> None:
    """Test GET /voices without language query parameter returns all voices."""
    with patch("voices.registry.list_voices", return_value=["voice1", "voice2"]) as mock_list:
        resp = client.get("/voices")
        assert resp.status_code == 200
        assert resp.json() == {"voices": ["voice1", "voice2"]}
        mock_list.assert_called_with(None)


def test_get_root_serves_ui_index_html(client: TestClient) -> None:
    """Acceptance criterion:

    Verify GET / returns HTTP 200 with text/html content from ui/index.html.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "AudioGen Session Control" in resp.text
    assert "Start Session" in resp.text
    assert "id=\"start-btn\"" in resp.text
    assert "id=\"synthesis-card\"" in resp.text
