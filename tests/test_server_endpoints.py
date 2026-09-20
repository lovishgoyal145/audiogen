"""Unit tests for FastAPI endpoints, tunnel launcher, and URL publisher (Ticket 002)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Generator, Tuple
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from audiogen.engine import Synthesizer
import voices.registry
from server.app import (
    FALLBACK_SECRET_ENV_VAR,
    DEFAULT_BEARER_ENV_VAR,
    cleanup_generated_files,
    create_app,
)
from server.registry import URLPublisher, URLPublisherError, publish_tunnel_url
from server.tunnel import (
    CloudflaredNotFoundError,
    CloudflareTunnel,
    TunnelStartupError,
    TunnelTimeoutError,
    start_tunnel,
)
from server.watchdog import IdleWatchdog


class MockTTSBackend:
    """Mock TTS backend simulating inference with concurrency tracking."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.calls = []
        self.active_count = 0
        self.max_concurrent = 0
        self.lock = threading.Lock()

    def __call__(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, int]:
        with self.lock:
            self.active_count += 1
            if self.active_count > self.max_concurrent:
                self.max_concurrent = self.active_count
            self.calls.append((text, ref_audio_path, ref_text))

        time.sleep(0.06)  # Simulate non-instantaneous inference

        with self.lock:
            self.active_count -= 1

        t = np.linspace(0, 0.2, int(self.sample_rate * 0.2), endpoint=False)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return waveform, self.sample_rate


@pytest.fixture
def mock_backend() -> MockTTSBackend:
    return MockTTSBackend()


@pytest.fixture
def mock_synthesizer(mock_backend: MockTTSBackend) -> Synthesizer:
    return Synthesizer(backend=mock_backend)


@pytest.fixture
def mock_watchdog() -> IdleWatchdog:
    return IdleWatchdog(idle_timeout_seconds=600.0, shutdown_action=MagicMock())


@pytest.fixture
def app_and_client(
    mock_synthesizer: Synthesizer,
    mock_watchdog: IdleWatchdog,
) -> Generator[Tuple[Any, TestClient], None, None]:
    app = create_app(
        synthesizer=mock_synthesizer,
        watchdog=mock_watchdog,
        auth_token="test-secret-token",
    )
    with TestClient(app) as client:
        yield app, client


# ==============================================================================
# 1. Health Endpoint Tests
# ==============================================================================


def test_health_returns_status_and_gpu_without_auth(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify GET /health succeeds unauthenticated and returns status, gpu, and idle countdown."""
    _, client = app_and_client
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert isinstance(data["gpu"], bool)
    assert isinstance(data["idle_seconds_remaining"], int)
    assert data["idle_seconds_remaining"] <= 600


# ==============================================================================
# 2. Authentication Tests
# ==============================================================================


def test_protected_endpoint_rejects_missing_auth(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /generate rejects request missing authentication."""
    _, client = app_and_client
    payload = {
        "text": "नमस्ते",
        "language": "hi",
        "speaker_ref_name": "anchor_male_energetic",
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized: Missing authentication token"


def test_protected_endpoint_rejects_invalid_token(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /generate rejects request with invalid bearer token."""
    _, client = app_and_client
    payload = {
        "text": "नमस्ते",
        "language": "hi",
        "speaker_ref_name": "anchor_male_energetic",
    }
    response = client.post(
        "/generate",
        json=payload,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized: Invalid authentication token"


def test_protected_endpoint_accepts_secret_header(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /generate accepts X-Server-Secret header."""
    _, client = app_and_client
    payload = {
        "text": "नमस्ते दुनिया",
        "language": "hi",
        "speaker_ref_name": "anchor_male_energetic",
    }
    response = client.post(
        "/generate",
        json=payload,
        headers={"X-Server-Secret": "test-secret-token"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_auth_resolves_from_environment_variables(mock_synthesizer: Synthesizer, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify authentication token falls back to SERVER_BEARER_TOKEN or SHARED_SECRET env vars."""
    monkeypatch.setenv(DEFAULT_BEARER_ENV_VAR, "env-bearer-token")
    app = create_app(synthesizer=mock_synthesizer, auth_token=None)
    with TestClient(app) as client:
        payload = {
            "text": "नमस्ते",
            "language": "hi",
            "speaker_ref_name": "anchor_male_energetic",
        }
        res = client.post("/generate", json=payload, headers={"Authorization": "Bearer env-bearer-token"})
        assert res.status_code == 200

    monkeypatch.delenv(DEFAULT_BEARER_ENV_VAR, raising=False)
    monkeypatch.setenv(FALLBACK_SECRET_ENV_VAR, "env-shared-secret")
    app2 = create_app(synthesizer=mock_synthesizer, auth_token=None)
    with TestClient(app2) as client2:
        payload = {
            "text": "नमस्ते",
            "language": "hi",
            "speaker_ref_name": "anchor_male_energetic",
        }
        res2 = client2.post("/generate", json=payload, headers={"X-Server-Secret": "env-shared-secret"})
        assert res2.status_code == 200


# ==============================================================================
# 3. Payload Validation Tests
# ==============================================================================


def test_generate_validation_empty_text_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify empty or whitespace-only text returns 400 Bad Request."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    for bad_text in ["", "   ", "\t\n"]:
        res = client.post(
            "/generate",
            json={"text": bad_text, "language": "hi", "speaker_ref_name": "anchor_male_energetic"},
            headers=headers,
        )
        assert res.status_code == 400
        assert res.json()["detail"] == "Text cannot be empty or whitespace-only."


def test_generate_validation_unsupported_language_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify unsupported language code returns 400 Bad Request."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    for bad_lang in ["es", "fr", "de"]:
        res = client.post(
            "/generate",
            json={"text": "Hello", "language": bad_lang, "speaker_ref_name": "anchor_male_energetic"},
            headers=headers,
        )
        assert res.status_code == 400
        assert f"Unsupported language: {bad_lang}" in res.json()["detail"]


def test_generate_validation_unknown_voice_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify non-existent voice returns 400 Bad Request."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    res = client.post(
        "/generate",
        json={"text": "नमस्ते", "language": "hi", "speaker_ref_name": "nonexistent_voice_id"},
        headers=headers,
    )
    assert res.status_code == 400
    assert "Unknown voice name: nonexistent_voice_id" in res.json()["detail"]


def test_generate_validation_missing_ref_on_disk_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify voice in manifest but missing from disk returns 500 Internal Server Error."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    with patch(
        "voices.registry.get_voice_ref",
        side_effect=FileNotFoundError("Reference audio file for voice 'ghost_voice' does not exist on disk"),
    ):
        res = client.post(
            "/generate",
            json={"text": "नमस्ते", "language": "hi", "speaker_ref_name": "ghost_voice"},
            headers=headers,
        )
        assert res.status_code == 500
        assert "does not exist on disk" in res.json()["detail"]


def test_generate_validation_missing_ref_on_disk_returns_500(app_and_client: Tuple[Any, TestClient]) -> None:
    """Alias for test_generate_validation_missing_ref_on_disk_returns_400 with updated contract name."""
    test_generate_validation_missing_ref_on_disk_returns_400(app_and_client)


def test_generate_language_mismatch_with_voice_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify requesting language not supported by the resolved voice returns 400 Bad Request."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    # anchor_female_calm only supports Hindi ('hi'), requesting 'pa' must return 400
    res = client.post(
        "/generate",
        json={"text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "language": "pa", "speaker_ref_name": "anchor_female_calm"},
        headers=headers,
    )
    assert res.status_code == 400
    assert "does not support language 'pa'" in res.json()["detail"]

    # storyteller_punjabi_elder only supports Punjabi ('pa'), requesting 'hi' must return 400
    res2 = client.post(
        "/generate",
        json={"text": "नमस्ते", "language": "hi", "speaker_ref_name": "storyteller_punjabi_elder"},
        headers=headers,
    )
    assert res2.status_code == 400
    assert "does not support language 'hi'" in res2.json()["detail"]


# ==============================================================================
# 4. Successful Synthesis & Return URI Tests
# ==============================================================================


def test_generate_valid_request_returns_wav_binary(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify valid request returns 200 OK, audio/wav Content-Type, and RIFF header."""
    app, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    # Mock watchdog touch spy
    watchdog = app.state.watchdog
    with patch.object(watchdog, "touch", wraps=watchdog.touch) as touch_spy:
        res = client.post(
            "/generate",
            json={"text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "language": "pa", "speaker_ref_name": "storyteller_punjabi_elder"},
            headers=headers,
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content[:4] == b"RIFF"

        # Verify WAV decodes properly
        audio_data, sr = sf.read(io.BytesIO(res.content))
        assert len(audio_data) > 0
        assert sr == 24000

        # Verify watchdog touch was invoked
        assert touch_spy.call_count >= 1


def test_generate_valid_request_returns_file_uri(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify return_uri=True returns JSON with valid file_uri pointing to WAV file on disk."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    res = client.post(
        "/generate",
        json={
            "text": "नमस्ते भारत",
            "language": "hi",
            "speaker_ref_name": "anchor_male_energetic",
            "return_uri": True,
        },
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "file_uri" in data
    file_uri = data["file_uri"]
    assert file_uri.startswith("file://")

    # Verify file exists on disk
    file_path = Path(file_uri[7:])  # strip file://
    assert file_path.exists()
    assert file_path.is_file()

    # Verify content
    audio_data, sr = sf.read(str(file_path))
    assert len(audio_data) > 0
    assert sr == 24000


def test_generate_inference_failure_returns_500(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify unexpected inference engine failure returns 500 Internal Server Error."""
    app, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    # Mock synthesizer to raise RuntimeError
    with patch.object(app.state.synthesizer, "synthesize", side_effect=RuntimeError("Synthesizer GPU out of memory")):
        res = client.post(
            "/generate",
            json={"text": "नमस्ते", "language": "hi", "speaker_ref_name": "anchor_male_energetic"},
            headers=headers,
        )
        assert res.status_code == 500
        assert "Inference failure: Synthesizer GPU out of memory" in res.json()["detail"]


def test_generate_calls_synthesizer_with_indicf5_signature(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /generate calls Synthesizer.synthesize with IndicF5 signature (text, ref_audio_path, ref_text)."""
    app, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    voice_name = "anchor_male_energetic"
    voice_rec = voices.registry.get_voice_ref(voice_name)

    synthesizer = app.state.synthesizer
    with patch.object(synthesizer, "synthesize", wraps=synthesizer.synthesize) as synth_spy:
        res = client.post(
            "/generate",
            json={"text": "नमस्ते भारत", "language": "hi", "speaker_ref_name": voice_name},
            headers=headers,
        )
        assert res.status_code == 200
        synth_spy.assert_called_once()
        call_args, call_kwargs = synth_spy.call_args
        assert call_args[0] == "नमस्ते भारत"
        assert call_kwargs["ref_audio_path"] == voice_rec.path
        assert call_kwargs["ref_text"] == voice_rec.ref_text


def test_generate_synthesizer_raises_file_not_found_returns_500(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify Synthesizer.synthesize raising FileNotFoundError returns 500 Internal Server Error."""
    app, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    with patch.object(
        app.state.synthesizer,
        "synthesize",
        side_effect=FileNotFoundError("Reference audio file not found on disk"),
    ):
        res = client.post(
            "/generate",
            json={"text": "नमस्ते", "language": "hi", "speaker_ref_name": "anchor_male_energetic"},
            headers=headers,
        )
        assert res.status_code == 500
        assert "Server configuration error: Reference audio file not found" in res.json()["detail"]


# ==============================================================================
# 5. Serialized Inference Concurrency Tests
# ==============================================================================


@pytest.mark.anyio
async def test_generate_concurrency_serialization(
    mock_backend: MockTTSBackend,
    mock_synthesizer: Synthesizer,
    mock_watchdog: IdleWatchdog,
) -> None:
    """Verify concurrent requests are serialized through asyncio.Lock (max_concurrent == 1)."""
    app = create_app(
        synthesizer=mock_synthesizer,
        watchdog=mock_watchdog,
        auth_token="test-secret-token",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as async_client:
        headers = {"Authorization": "Bearer test-secret-token"}
        payload = {
            "text": "नमस्ते दुनिया",
            "language": "hi",
            "speaker_ref_name": "anchor_male_energetic",
        }

        # Fire 2 concurrent requests simultaneously
        req1 = async_client.post("/generate", json=payload, headers=headers)
        req2 = async_client.post("/generate", json=payload, headers=headers)

        res1, res2 = await asyncio.gather(req1, req2)

        assert res1.status_code == 200
        assert res2.status_code == 200
        assert res1.headers["content-type"] == "audio/wav"
        assert res2.headers["content-type"] == "audio/wav"

        # Crucial assertion: max concurrent executions inside backend must never exceed 1
        assert mock_backend.max_concurrent == 1
        assert len(mock_backend.calls) == 2


# ==============================================================================
# 6. Cloudflare Tunnel Tests
# ==============================================================================


def test_tunnel_cloudflared_not_found() -> None:
    """Verify missing cloudflared binary raises CloudflaredNotFoundError."""
    tunnel = CloudflareTunnel(binary_path="/path/does/not/exist/cloudflared")
    with pytest.raises(CloudflaredNotFoundError, match="cloudflared executable not found"):
        tunnel.start()


def test_tunnel_startup_premature_exit() -> None:
    """Verify premature exit with non-zero status raises TunnelStartupError."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1
    mock_proc.stderr = io.StringIO("Fatal error: invalid configuration\n")
    mock_proc.stdout = io.StringIO("")

    with patch("shutil.which", return_value="/bin/cloudflared"), patch(
        "subprocess.Popen", return_value=mock_proc
    ):
        tunnel = CloudflareTunnel()
        with pytest.raises(TunnelStartupError, match="cloudflared exited unexpectedly with code 1"):
            tunnel.start()


def test_tunnel_startup_timeout() -> None:
    """Verify timeout waiting for URL raises TunnelTimeoutError."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stderr = io.StringIO("Connecting to Cloudflare...\nStill waiting...\n")
    mock_proc.stdout = io.StringIO("")

    with patch("shutil.which", return_value="/bin/cloudflared"), patch(
        "subprocess.Popen", return_value=mock_proc
    ):
        tunnel = CloudflareTunnel(startup_timeout_seconds=0.08)
        with pytest.raises(TunnelTimeoutError, match="Timed out after 0.08s"):
            tunnel.start()


def test_tunnel_success_url_extraction() -> None:
    """Verify successful spawn parses trycloudflare.com URL."""
    log_stream = (
        "2026-09-14T10:00:00Z INF +-----------------------------------------------------------------------+\n"
        "2026-09-14T10:00:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time  |\n"
        "2026-09-14T10:00:00Z INF |  https://audio-gen-speed.trycloudflare.com                               |\n"
        "2026-09-14T10:00:00Z INF +-----------------------------------------------------------------------+\n"
    )

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stderr = io.StringIO(log_stream)
    mock_proc.stdout = io.StringIO("")

    with patch("shutil.which", return_value="/bin/cloudflared"), patch(
        "subprocess.Popen", return_value=mock_proc
    ):
        tunnel, url = start_tunnel(startup_timeout_seconds=2.0)
        assert url == "https://audio-gen-speed.trycloudflare.com"
        assert tunnel.tunnel_url == "https://audio-gen-speed.trycloudflare.com"
        assert tunnel.is_running is True

        tunnel.stop()
        mock_proc.terminate.assert_called_once()


# ==============================================================================
# 7. URL Publisher Tests
# ==============================================================================


def test_url_publisher_validation_errors() -> None:
    """Verify invalid URL or secret raises ValueError."""
    pub = URLPublisher()
    with pytest.raises(ValueError, match="must be non-empty strings"):
        pub.publish("", "secret")

    with pytest.raises(ValueError, match="must be non-empty strings"):
        pub.publish("https://test.trycloudflare.com", "")


def test_url_publisher_missing_endpoint_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify publishing without configured webhook endpoint raises URLPublisherError."""
    monkeypatch.delenv("TUNNEL_REGISTRY_WEBHOOK_URL", raising=False)
    pub = URLPublisher()
    with pytest.raises(URLPublisherError, match="No registry webhook URL configured"):
        pub.publish("https://test.trycloudflare.com", "secret")


def test_url_publisher_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URLPublisher sends POST to /set/tunnel_url and parses JSON response."""
    monkeypatch.delenv("TUNNEL_REGISTRY_AUTH_TOKEN", raising=False)
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": "OK"}
    mock_client.post.return_value = mock_resp

    pub = URLPublisher(endpoint_url="https://api.example.com/set/tunnel_url", client=mock_client)
    res = pub.publish("https://test.trycloudflare.com", "secret-token", metadata={"env": "test"})

    assert res == {"result": "OK"}
    mock_client.post.assert_called_once_with(
        "https://api.example.com/set/tunnel_url",
        json={
            "tunnel_url": "https://test.trycloudflare.com",
            "secret": "secret-token",
            "env": "test",
        },
        timeout=10.0,
    )


def test_url_publisher_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URLPublisher sends exact path and Authorization: Bearer <token> when TUNNEL_REGISTRY_AUTH_TOKEN is set."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "mock-auth-token-12345")
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": "OK"}
    mock_client.post.return_value = mock_resp

    pub = URLPublisher(endpoint_url="https://api.example.com/set/tunnel_url", client=mock_client)
    res = pub.publish("https://test.trycloudflare.com", "secret-token", metadata={"env": "test"})

    assert res == {"result": "OK"}
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://api.example.com/set/tunnel_url"
    assert call_args.kwargs.get("headers") == {"Authorization": "Bearer mock-auth-token-12345"}
    assert call_args.kwargs["json"]["secret"] == "secret-token"
    assert call_args.kwargs["json"]["tunnel_url"] == "https://test.trycloudflare.com"


def test_url_publisher_upstash_domain_auto_appends_set_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URLPublisher automatically normalizes bare Upstash domain or /get path to /set/tunnel_url."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "upstash-secret")
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": "OK"}
    mock_client.post.return_value = mock_resp

    # Test 1: Bare Upstash domain
    pub_base = URLPublisher(endpoint_url="https://large-pup-282364.upstash.io", client=mock_client)
    pub_base.publish("https://test.trycloudflare.com", "secret-token")
    assert mock_client.post.call_args[0][0] == "https://large-pup-282364.upstash.io/set/tunnel_url"
    assert mock_client.post.call_args.kwargs.get("headers") == {"Authorization": "Bearer upstash-secret"}

    mock_client.reset_mock()

    # Test 2: Upstash read path (/get/tunnel_url) normalized to write path (/set/tunnel_url)
    pub_get = URLPublisher(endpoint_url="https://large-pup-282364.upstash.io/get/tunnel_url", client=mock_client)
    pub_get.publish("https://test.trycloudflare.com", "secret-token")
    assert mock_client.post.call_args[0][0] == "https://large-pup-282364.upstash.io/set/tunnel_url"
    assert mock_client.post.call_args.kwargs.get("headers") == {"Authorization": "Bearer upstash-secret"}


def test_url_publisher_with_empty_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URLPublisher sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is whitespace/empty."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "   ")
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_client.post.return_value = mock_resp

    pub = URLPublisher(endpoint_url="https://api.example.com/set/tunnel_url", client=mock_client)
    pub.publish("https://test.trycloudflare.com", "secret-token")

    mock_client.post.assert_called_once()
    assert mock_client.post.call_args[0][0] == "https://api.example.com/set/tunnel_url"
    assert "headers" not in mock_client.post.call_args.kwargs


def test_url_publisher_without_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URLPublisher sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is unset."""
    monkeypatch.delenv("TUNNEL_REGISTRY_AUTH_TOKEN", raising=False)
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_client.post.return_value = mock_resp

    pub = URLPublisher(endpoint_url="https://api.example.com/set/tunnel_url", client=mock_client)
    pub.publish("https://test.trycloudflare.com", "secret-token")

    mock_client.post.assert_called_once()
    assert mock_client.post.call_args[0][0] == "https://api.example.com/set/tunnel_url"
    assert "headers" not in mock_client.post.call_args.kwargs


def test_url_publisher_http_failure() -> None:
    """Verify HTTP error raises URLPublisherError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=mock_resp)
    mock_client.post.return_value = mock_resp

    pub = URLPublisher(endpoint_url="https://api.example.com/set/tunnel_url", client=mock_client)
    with pytest.raises(URLPublisherError, match="Failed to publish tunnel URL"):
        pub.publish("https://test.trycloudflare.com", "secret-token")


def test_cleanup_generated_files_retention_and_age(tmp_path: Path) -> None:
    """Verify cleanup_generated_files removes expired files and caps retention count."""
    target_dir = tmp_path / "audiogen_generated"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 dummy files
    f1 = target_dir / "file1.wav"
    f2 = target_dir / "file2.wav"
    f3 = target_dir / "file3.wav"
    f4 = target_dir / "file4.wav"
    f5 = target_dir / "file5.wav"

    for f in [f1, f2, f3, f4, f5]:
        f.write_text("dummy audio")

    # Set mtime for f1 to be 2 hours ago
    old_time = time.time() - 7200.0
    import os
    os.utime(str(f1), (old_time, old_time))

    # Clean with max_age = 3600 (f1 should be deleted)
    cleanup_generated_files(target_dir, max_files=10, max_age_seconds=3600.0)
    assert not f1.exists()
    assert f2.exists()
    assert f3.exists()
    assert f4.exists()
    assert f5.exists()

    # Now enforce max_files=2 (should keep 2 files and delete oldest remaining)
    cleanup_generated_files(target_dir, max_files=2, max_age_seconds=86400.0)
    remaining = list(target_dir.glob("*.wav"))
    assert len(remaining) <= 2


def _create_test_audio_bytes(duration_sec: float = 2.0, sample_rate: int = 24000, silent: bool = False) -> bytes:
    import io
    import math
    import struct
    import wave

    sr = int(sample_rate)
    n = int(sr * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if silent:
            data = struct.pack(f"<{n}h", *(0 for _ in range(n)))
        else:
            data = struct.pack(
                f"<{n}h",
                *(int(10000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)),
            )
        wf.writeframes(data)
    return buf.getvalue()


def test_generate_accepts_english_language(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /generate with language 'en' returns 200 WAV binary."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    res = client.post(
        "/generate",
        json={
            "text": "Hello world, this is a test.",
            "language": "en",
            "speaker_ref_name": "narrator_english_neutral",
        },
        headers=headers,
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) > 0


def test_clone_voice_missing_auth_returns_401(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /voices/clone returns 401 without bearer token."""
    _, client = app_and_client
    audio_bytes = _create_test_audio_bytes(2.0)
    res = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone_1",
            "language": "en",
            "ref_text": "This is test transcript.",
        },
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
    )
    assert res.status_code == 401


def test_clone_voice_short_audio_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /voices/clone returns 400 when audio duration is < 1.0s."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(0.5)
    res = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone_short",
            "language": "en",
            "ref_text": "This is test transcript.",
        },
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
        headers=headers,
    )
    assert res.status_code == 400
    assert "too short" in res.json()["detail"]


def test_clone_voice_silent_audio_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /voices/clone returns 400 when audio is silent."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(2.0, silent=True)
    res = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone_silent",
            "language": "en",
            "ref_text": "This is test transcript.",
        },
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
        headers=headers,
    )
    assert res.status_code == 400
    assert "silent" in res.json()["detail"]


def test_clone_voice_empty_ref_text_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /voices/clone returns 400 when ref_text is empty or whitespace."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(2.0)
    res = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone_empty_text",
            "language": "en",
            "ref_text": "   ",
        },
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
        headers=headers,
    )
    assert res.status_code == 400
    assert "ref_text" in res.json()["detail"]


def test_clone_voice_unsupported_language_returns_400(app_and_client: Tuple[Any, TestClient]) -> None:
    """Verify POST /voices/clone returns 400 for unsupported language."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(2.0)
    res = client.post(
        "/voices/clone",
        data={
            "voice_id": "test_clone_bad_lang",
            "language": "es",
            "ref_text": "Hola mundo.",
        },
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
        headers=headers,
    )
    assert res.status_code == 400
    assert "Unsupported language" in res.json()["detail"]


def test_clone_voice_valid_request_returns_201_and_can_generate(
    app_and_client: Tuple[Any, TestClient],
) -> None:
    """Verify POST /voices/clone returns 201 Created and voice can be synthesized."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(3.5)
    try:
        res = client.post(
            "/voices/clone",
            data={
                "voice_id": "cloned_test_voice",
                "language": "en",
                "ref_text": "This is the transcript of the cloned voice.",
                "name": "Cloned Test Voice",
                "description": "A cloned test voice for unit testing",
            },
            files={"file": ("cloned.wav", audio_bytes, "audio/wav")},
            headers=headers,
        )
        assert res.status_code == 201
        data = res.json()
        assert data["status"] == "success"
        assert data["voice"]["id"] == "cloned_test_voice"
        assert data["voice"]["name"] == "Cloned Test Voice"
        assert "en" in data["voice"]["language"]

        # Verify new voice can now be used in /generate
        gen_res = client.post(
            "/generate",
            json={
                "text": "Hello synthesized using cloned voice!",
                "language": "en",
                "speaker_ref_name": "cloned_test_voice",
            },
            headers=headers,
        )
        assert gen_res.status_code == 200
        assert gen_res.headers["content-type"] == "audio/wav"
    finally:
        schema_file = voices.registry.DEFAULT_MANIFEST_PATH
        if schema_file.is_file():
            import json
            with open(schema_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "cloned_test_voice" in d:
                del d["cloned_test_voice"]
                with open(schema_file, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
        ref_file = voices.registry.REPO_ROOT / "voices" / "refs" / "cloned_test_voice.wav"
        if ref_file.exists():
            ref_file.unlink()
        voices.registry.clear_registry_cache()


def test_server_voices_sync_and_list_endpoints(
    app_and_client: Tuple[Any, TestClient],
) -> None:
    """Verify POST /voices/sync and GET /voices endpoints on server app."""
    _, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}

    # Sync
    sync_res = client.post("/voices/sync", headers=headers)
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert sync_data["status"] == "synced"
    assert "narrator_english_neutral" in sync_data["voices"]

    # List English voices
    list_res = client.get("/voices?language=en")
    assert list_res.status_code == 200
    voices_en = list_res.json()["voices"]
    assert "narrator_english_neutral" in voices_en


def test_clone_voice_gpu_verification_failure_returns_500(
    app_and_client: Tuple[Any, TestClient],
) -> None:
    """Verify POST /voices/clone raises 500 when synthesizer verification fails."""
    app, client = app_and_client
    headers = {"Authorization": "Bearer test-secret-token"}
    audio_bytes = _create_test_audio_bytes(2.0)

    with patch.object(
        app.state.synthesizer, "synthesize", side_effect=RuntimeError("CUDA out of memory")
    ):
        res = client.post(
            "/voices/clone",
            data={
                "voice_id": "cloned_oom_voice",
                "language": "en",
                "ref_text": "This is test transcript.",
            },
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
            headers=headers,
        )
        assert res.status_code == 500
        assert "GPU verification failed: CUDA out of memory" in res.json()["detail"]
        out_file = voices.registry.REPO_ROOT / "voices" / "refs" / "cloned_oom_voice.wav"
        assert not out_file.exists()

