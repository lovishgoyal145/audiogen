"""Integration and unit tests for orchestrator/gateway.py and API endpoints."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
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
        registry_url=os.environ.get("TUNNEL_REGISTRY_WEBHOOK_URL", "https://mock-registry.example.com"),
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


def test_get_voices_english(client: TestClient) -> None:
    """Verify GET /voices?language=en queries voice registry for English voices."""
    with patch("voices.registry.list_voices", return_value=["narrator_english_neutral"]) as mock_list:
        resp = client.get("/voices?language=en")
        assert resp.status_code == 200
        assert resp.json() == {"voices": ["narrator_english_neutral"]}
        mock_list.assert_called_with("en")


@pytest.mark.parametrize("non_ready_state", ["IDLE", "STARTING", "ERROR"])
def test_gateway_clone_voice_rejects_when_not_ready_returns_409(
    client: TestClient,
    test_gateway: SessionGateway,
    non_ready_state: str,
) -> None:
    """Verify POST /voices/clone returns HTTP 409 when session is not READY."""
    test_gateway._state = non_ready_state
    test_gateway._status_message = f"In {non_ready_state} state"

    resp = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone",
            "language": "en",
            "ref_text": "Sample text",
        },
        files={"file": ("test.wav", b"RIFFfakebytes", "audio/wav")},
    )
    assert resp.status_code == 409
    data = resp.json()
    assert data["error"] == "session not ready"
    assert data["status"] == non_ready_state


def test_gateway_clone_voice_proxies_when_ready_returns_201(
    client: TestClient,
    test_gateway: SessionGateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /voices/clone proxies to remote worker when READY and saves locally."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"
    test_gateway.bearer_token = "secret-token-gw"

    # Mock audio bytes
    mock_wav = b"RIFFvalidwavbytes"

    mock_resp_json = {
        "status": "success",
        "voice": {
            "id": "gw_cloned_voice",
            "name": "Gw Cloned Voice",
            "language": ["en"],
            "ref_text": "Spoken words transcript",
        },
    }

    async def mock_post(url: str, files: dict, data: dict, headers: dict, **kwargs):
        assert url == "https://tunnel.trycloudflare.com/voices/clone"
        assert headers.get("Authorization") == "Bearer secret-token-gw"
        assert data["voice_id"] == "gw_cloned_voice"
        assert data["language"] == "en"
        return httpx.Response(
            status_code=201,
            json=mock_resp_json,
            headers={"content-type": "application/json"},
        )

    try:
        with patch("httpx.AsyncClient.post", side_effect=mock_post), \
             patch("voices.registry.register_voice") as mock_reg:
            resp = client.post(
                "/voices/clone",
                data={
                    "voice_id": "gw_cloned_voice",
                    "language": "en",
                    "ref_text": "Spoken words transcript",
                    "name": "Gw Cloned Voice",
                },
                files={"file": ("sample.wav", mock_wav, "audio/wav")},
            )
            assert resp.status_code == 201
            assert resp.json() == mock_resp_json
            mock_reg.assert_called_once()
    finally:
        f = voices.registry.REPO_ROOT / "voices" / "refs" / "gw_cloned_voice.wav"
        if f.exists():
            f.unlink()


def test_gateway_clone_voice_remote_error_returns_502(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify network failure during clone forwarding returns 502 Bad Gateway."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        resp = client.post(
            "/voices/clone",
            data={
                "voice_id": "gw_cloned_fail",
                "language": "en",
                "ref_text": "Text",
            },
            files={"file": ("sample.wav", b"123", "audio/wav")},
        )
        assert resp.status_code == 502
        assert "Failed to forward clone request" in resp.json()["error"]


def test_gateway_generate_accepts_english(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify POST /generate with language 'en' is proxied cleanly to remote server."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"
    test_gateway.bearer_token = "secret-token-gw"

    payload = {
        "text": "Hello world in English",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }

    mock_audio = b"RIFFaudioeng"

    async def mock_post(url: str, json: dict, headers: dict, **kwargs):
        assert json["language"] == "en"
        assert headers.get("Authorization") == "Bearer secret-token-gw"
        assert headers.get("X-Server-Secret") == "secret-token-gw"
        assert headers.get("Content-Type") == "application/json"
        return httpx.Response(
            status_code=200,
            content=mock_audio,
            headers={"content-type": "audio/wav"},
        )

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 200
        assert resp.content == mock_audio


def test_gateway_clone_voice_rejects_empty_or_invalid_voice_id(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify POST /voices/clone with invalid or empty voice_id returns 400."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"

    resp = client.post(
        "/voices/clone",
        data={
            "voice_id": "???///",
            "language": "en",
            "ref_text": "Sample text",
        },
        files={"file": ("sample.wav", b"RIFFbytes", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "Invalid voice_id" in resp.json()["error"]


def test_gateway_clone_voice_normalizes_persisted_audio_to_24k_mono(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify audio persisted by gateway is normalized to 24 kHz mono 16-bit PCM WAV."""
    import io
    import soundfile as sf
    import numpy as np

    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://tunnel.trycloudflare.com"
    test_gateway.bearer_token = "secret-token-gw"

    # Create a 16 kHz stereo audio buffer
    sr_input = 16000
    t = np.linspace(0, 1.5, int(sr_input * 1.5), endpoint=False)
    stereo_data = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 880 * t)])
    buf = io.BytesIO()
    sf.write(buf, stereo_data, sr_input, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    mock_resp_json = {
        "status": "success",
        "voice": {
            "id": "gw_norm_voice",
            "name": "Gw Norm Voice",
            "language": ["en"],
            "ref_text": "Test transcript",
        },
    }

    async def mock_post(url: str, files: dict, data: dict, headers: dict, **kwargs):
        return httpx.Response(
            status_code=201,
            json=mock_resp_json,
            headers={"content-type": "application/json"},
        )

    target_path = voices.registry.REPO_ROOT / "voices" / "refs" / "gw_norm_voice.wav"
    try:
        with patch("httpx.AsyncClient.post", side_effect=mock_post), \
             patch("voices.registry.register_voice"):
            resp = client.post(
                "/voices/clone",
                data={
                    "voice_id": "gw_norm_voice",
                    "language": "en",
                    "ref_text": "Test transcript",
                },
                files={"file": ("sample.wav", audio_bytes, "audio/wav")},
            )
            assert resp.status_code == 201
            assert target_path.exists()
            info = sf.info(str(target_path))
            assert info.samplerate == 24000
            assert info.channels == 1
            assert info.subtype == "PCM_16"
    finally:
        if target_path.exists():
            target_path.unlink()


def test_heartbeat_endpoint_updates_timestamp_and_returns_ok(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify POST /session/heartbeat refreshes last_heartbeat and returns 200."""
    initial_hb = test_gateway.last_heartbeat
    resp = client.post("/session/heartbeat", json={"session_id": test_gateway.session_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["session_id"] == test_gateway.session_id
    assert test_gateway.last_heartbeat >= initial_hb


def test_terminate_endpoint_transitions_state_to_shutdown_and_rejects_requests(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify POST /session/terminate transitions state to SHUTDOWN and /generate returns 410."""
    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_is_tunnel_healthy", return_value="https://test.trycloudflare.com"):
        client.post("/session/start")
        # Give background task time to set READY
        for _ in range(50):
            if test_gateway.state == "READY":
                break
            time.sleep(0.01)

    assert test_gateway.state == "READY"

    # Terminate session
    term_resp = client.post(
        "/session/terminate",
        json={"session_id": test_gateway.session_id, "reason": "user_exit"},
    )
    assert term_resp.status_code == 200
    assert term_resp.json()["status"] == "SHUTDOWN"
    assert test_gateway.state == "SHUTDOWN"

    # Subsequent /generate must return HTTP 410 Gone
    gen_resp = client.post(
        "/generate",
        json={
            "text": "Hello world",
            "language": "en",
            "speaker_ref_name": "narrator_english_neutral",
        },
    )
    assert gen_resp.status_code == 410
    assert gen_resp.json()["error"] == "session terminated"


def test_terminate_endpoint_is_idempotent(
    client: TestClient,
    test_gateway: SessionGateway,
) -> None:
    """Verify multiple POST /session/terminate calls succeed safely."""
    resp1 = client.post("/session/terminate", json={"reason": "tab_close"})
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "SHUTDOWN"

    resp2 = client.post("/session/terminate", json={"reason": "tab_close"})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "SHUTDOWN"


# ==============================================================================
# 4. TICKET-012 Regression Tests: Config, Pre-Flight, Discovery, & Stale Tabs
# ==============================================================================


def test_config_contract_webhook_url_env_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_app resolves registry URL from webhook_url_env."""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(
        "registry:\n"
        "  webhook_url_env: 'CUSTOM_TUNNEL_URL'\n"
        "  webhook_url: null\n"
    )
    monkeypatch.setenv("CUSTOM_TUNNEL_URL", "https://custom-tunnel.example.com")
    custom_app = create_app(config_path=cfg_file)
    gw: SessionGateway = custom_app.state.gateway
    assert gw.registry_url == "https://custom-tunnel.example.com"


def test_config_contract_webhook_url_fallback_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_app falls back to webhook_url if env var is unset."""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(
        "registry:\n"
        "  webhook_url_env: 'NONEXISTENT_ENV_VAR'\n"
        "  webhook_url: 'https://fallback-tunnel.example.com'\n"
    )
    monkeypatch.delenv("NONEXISTENT_ENV_VAR", raising=False)
    custom_app = create_app(config_path=cfg_file)
    gw: SessionGateway = custom_app.state.gateway
    assert gw.registry_url == "https://fallback-tunnel.example.com"


def test_preflight_validation_fails_fast_on_missing_registry_url() -> None:
    """Verify start_session fails fast (< 50ms) without calling push if registry URL is missing."""
    gw = SessionGateway(registry_url=None)
    mock_push = MagicMock()
    gw._push_fn = mock_push

    start_time = time.time()
    res = asyncio.run(gw.start_session())
    elapsed = time.time() - start_time

    assert elapsed < 0.05
    assert res["status"] == "ERROR"
    assert res["failure_stage"] == "local_configuration"
    assert "Registry URL is not configured" in res["message"]
    mock_push.assert_not_called()
    assert gw.state == "ERROR"


def test_preflight_validation_fails_fast_on_malformed_registry_url() -> None:
    """Verify start_session fails fast on malformed registry URL."""
    gw = SessionGateway(registry_url="ftp://invalid-scheme.example.com")
    mock_push = MagicMock()
    gw._push_fn = mock_push

    res = asyncio.run(gw.start_session())
    assert res["status"] == "ERROR"
    assert res["failure_stage"] == "local_configuration"
    assert "Invalid registry URL structure" in res["message"]
    mock_push.assert_not_called()


def test_preflight_validation_fails_fast_on_missing_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify start_session fails fast when TUNNEL_REGISTRY_AUTH_TOKEN is missing."""
    monkeypatch.delenv("TUNNEL_REGISTRY_AUTH_TOKEN", raising=False)
    gw = SessionGateway(registry_url="https://mock-registry.example.com")
    mock_push = MagicMock()
    gw._push_fn = mock_push

    res = asyncio.run(gw.start_session())
    assert res["status"] == "ERROR"
    assert res["failure_stage"] == "local_configuration"
    assert "TUNNEL_REGISTRY_AUTH_TOKEN is missing" in res["message"]
    mock_push.assert_not_called()


def test_preflight_validation_fails_fast_on_missing_kaggle_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify start_session fails fast when Kaggle credentials are missing."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with patch("pathlib.Path.is_file", return_value=False):
        gw = SessionGateway(registry_url="https://mock-registry.example.com")
        mock_push = MagicMock()
        gw._push_fn = mock_push

        res = asyncio.run(gw.start_session())
        assert res["status"] == "ERROR"
        assert res["failure_stage"] == "local_configuration"
        assert "Kaggle credentials not found" in res["message"]
        mock_push.assert_not_called()


def test_polling_transitions_to_error_on_registry_auth_error(test_gateway: SessionGateway, client: TestClient) -> None:
    """Verify polling loop transitions to ERROR when discovery reports REGISTRY_AUTH_ERROR."""
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 5.0

    mock_discovery = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.REGISTRY_AUTH_ERROR,
        message="HTTP 401 Unauthorized",
        http_status=401,
    )

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_check_tunnel_discovery", return_value=mock_discovery):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert data["failure_stage"] == "registry_communication"
        assert "Registry authentication failed" in data["message"]


def test_polling_retries_on_no_tunnel_registered(test_gateway: SessionGateway, client: TestClient) -> None:
    """Verify polling loop continues polling when NO_TUNNEL_REGISTERED is returned."""
    test_gateway.poll_interval_seconds = 0.01
    test_gateway.startup_timeout_seconds = 0.05

    mock_no_tunnel = session_manager.TunnelDiscoveryResult(
        status=session_manager.DiscoveryStatus.NO_TUNNEL_REGISTERED,
        message="No tunnel registered yet.",
    )

    with patch.object(test_gateway, "_call_kaggle_push"), \
         patch.object(test_gateway, "_call_get_kaggle_status", return_value="RUNNING"), \
         patch.object(test_gateway, "_call_check_tunnel_discovery", return_value=mock_no_tunnel):
        resp = client.post("/session/start")
        assert resp.status_code == 200

        # Wait until timeout
        for _ in range(20):
            time.sleep(0.01)
            status_resp = client.get("/session/status")
            if status_resp.json()["status"] == "ERROR":
                break

        data = client.get("/session/status").json()
        assert data["status"] == "ERROR"
        assert data["failure_stage"] == "timeout"
        assert "timed out" in data["message"]


def test_terminate_rejects_mismatched_session_id_when_ready(test_gateway: SessionGateway, client: TestClient) -> None:
    """Verify POST /session/terminate with mismatched session_id returns 400 and preserves READY."""
    test_gateway._state = "READY"
    resp = client.post(
        "/session/terminate",
        json={"session_id": "stale-session-id-123", "reason": "stale_tab_close"},
    )
    assert resp.status_code == 400
    assert "mismatched session_id" in resp.json()["message"]
    assert test_gateway.state == "READY"


def test_terminate_rejects_missing_session_id_when_ready(test_gateway: SessionGateway, client: TestClient) -> None:
    """Verify POST /session/terminate with empty session_id returns 400 when session is active."""
    test_gateway._state = "READY"
    resp = client.post(
        "/session/terminate",
        json={"reason": "anonymous_request"},
    )
    assert resp.status_code == 400
    assert "session_id is required" in resp.json()["message"]
    assert test_gateway.state == "READY"


def test_terminate_succeeds_with_matching_session_id(test_gateway: SessionGateway, client: TestClient) -> None:
    """Verify POST /session/terminate with matching session_id succeeds and transitions to SHUTDOWN."""
    test_gateway._state = "READY"
    test_gateway._tunnel_url = "https://ready.trycloudflare.com"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("orchestrator.session_manager.cancel_kaggle_kernel", return_value=True), \
         patch("orchestrator.session_manager.delete_tunnel_url", return_value=True):
        mock_post.return_value = MagicMock(status_code=200)
        resp = client.post(
            "/session/terminate",
            json={"session_id": test_gateway.session_id, "reason": "user_exit"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "SHUTDOWN"
        assert test_gateway.state == "SHUTDOWN"



