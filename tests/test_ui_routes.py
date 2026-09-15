"""Integration and unit tests for AudioGen minimalist dark web UI and API routes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from starlette.testclient import TestClient

from audiogen.config import get_settings
from audiogen.main import REPO_ROOT, app
import voices.registry


class MockUIBackend:
    """In-memory test synthesizer backend providing valid 16-bit PCM WAV audio for UI integration tests."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        duration = 0.2
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        waveform = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        return waveform, self.sample_rate


@pytest.fixture(autouse=True)
def clean_test_state() -> Generator[None, None, None]:
    """Reset app state and configuration singleton between test cases with test synthesizer backend."""
    app.dependency_overrides.clear()
    app.state.custom_voices = {}
    app.state.synthesizer = MockUIBackend()
    get_settings.cache_clear()
    yield
    app.dependency_overrides.clear()
    app.state.custom_voices = {}
    app.state.synthesizer = None
    get_settings.cache_clear()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_ui_root_route_returns_200_and_html(client: TestClient) -> None:
    """Test 1: GET / returns 200 OK with valid HTML content and required DOM elements."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text

    # Verify structural DOM markers required by ticket & plan
    assert "AUDIOGEN" in body
    assert "app-container" in body
    assert "Select Language" in body
    assert "Select Voice" in body
    assert "Script Input" in body
    assert "Synthesizing Audio" in body
    assert "audio-player" in body
    assert "btn-download" in body
    assert "voice-modal" in body
    assert "new-voice-sample" in body


def test_static_assets_served(client: TestClient) -> None:
    """Test 2: GET /static/styles.css and /static/app.js are served correctly."""
    # Stylesheet check
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]
    css_content = css_res.text
    assert "#09090b" in css_content
    assert "max-width: 520px" in css_content
    assert "margin-left: 0" in css_content

    # JavaScript check
    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert any(
        mime in js_res.headers["content-type"]
        for mime in ["application/javascript", "text/javascript", "text/plain"]
    )
    js_content = js_res.text
    assert "selectLanguage" in js_content
    assert "startGeneration" in js_content


def test_api_voices_listing_and_filtering(client: TestClient) -> None:
    """Test 3: GET /api/voices returns voices and filters by language."""
    # All voices
    all_res = client.get("/api/voices")
    assert all_res.status_code == 200
    voices_list = all_res.json()
    assert isinstance(voices_list, list)
    assert len(voices_list) >= 3

    # Hindi voices filter
    hi_res = client.get("/api/voices?language=hi")
    assert hi_res.status_code == 200
    hi_voices = hi_res.json()
    assert isinstance(hi_voices, list)
    assert len(hi_voices) >= 1
    for v in hi_voices:
        assert "hi" in [lang.lower() for lang in v["language"]]

    # Punjabi voices filter
    pa_res = client.get("/api/voices?language=pa")
    assert pa_res.status_code == 200
    pa_voices = pa_res.json()
    assert isinstance(pa_voices, list)
    assert len(pa_voices) >= 1
    for v in pa_voices:
        assert "pa" in [lang.lower() for lang in v["language"]]

    # English voices filter
    en_res = client.get("/api/voices?language=en")
    assert en_res.status_code == 200
    en_voices = en_res.json()
    assert isinstance(en_voices, list)
    assert len(en_voices) >= 1
    for v in en_voices:
        assert "en" in [lang.lower() for lang in v["language"]]


def test_api_voices_register_custom_voice(client: TestClient) -> None:
    """Test 4: POST /api/voices registers custom profile and persists in subsequent queries."""
    payload = {
        "id": "custom_reporter_voice",
        "name": "Custom Reporter Voice",
        "language": ["en"],
        "description": "Specialized English broadcast voice",
        "path": "voices/refs/anchor_female_calm.wav",
        "ref_text": "Good morning, here is the automated broadcast report.",
    }
    create_res = client.post("/api/voices", json=payload)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["id"] == "custom_reporter_voice"
    assert created["name"] == "Custom Reporter Voice"
    assert created["language"] == ["en"]
    assert created["path"] == "voices/refs/anchor_female_calm.wav"

    # Verify query returns new voice
    query_res = client.get("/api/voices?language=en")
    assert query_res.status_code == 200
    voices_found = query_res.json()
    voice_ids = [v["id"] for v in voices_found]
    assert "custom_reporter_voice" in voice_ids


def test_api_voices_path_sanitization_and_traversal_rejection(client: TestClient) -> None:
    """Test path sanitization: arbitrary path traversal and non-wav paths are rejected."""
    # 1. Path traversal attempt outside voices/
    res = client.post(
        "/api/voices",
        json={
            "id": "evil_voice",
            "language": ["en"],
            "path": "../../etc/passwd",
            "ref_text": "Sample text",
        },
    )
    assert res.status_code == 400
    assert "path traversal" in res.json().get("detail", "").lower()

    # 2. Non-.wav extension
    res = client.post(
        "/api/voices",
        json={
            "id": "text_voice",
            "language": ["en"],
            "path": "voices/registry_schema.json",
            "ref_text": "Sample text",
        },
    )
    assert res.status_code == 400
    assert ".wav" in res.json().get("detail", "").lower()

    # 3. Non-existent WAV file inside voices/
    res = client.post(
        "/api/voices",
        json={
            "id": "missing_wav_voice",
            "language": ["en"],
            "path": "voices/refs/non_existent_audio_sample.wav",
            "ref_text": "Sample text",
        },
    )
    assert res.status_code == 400
    assert "does not exist on disk" in res.json().get("detail", "").lower()


def test_api_voices_validation_errors(client: TestClient) -> None:
    """Test validation errors on POST /api/voices for empty fields."""
    # Empty ID
    res = client.post(
        "/api/voices",
        json={"id": "   ", "language": ["hi"], "ref_text": "Sample"},
    )
    assert res.status_code == 400

    # Empty reference text
    res = client.post(
        "/api/voices",
        json={"id": "test_id", "language": ["hi"], "ref_text": "   "},
    )
    assert res.status_code == 400

    # Empty language list
    res = client.post(
        "/api/voices",
        json={"id": "test_id", "language": [], "ref_text": "Sample"},
    )
    assert res.status_code == 400


def test_api_generate_valid_request(client: TestClient) -> None:
    """Test 5: POST /api/generate produces valid WAV binary starting with RIFF header."""
    payload = {
        "text": "Hello world from AudioGen speech synthesis engine.",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    content = response.content
    assert len(content) > 44  # WAV header is 44 bytes minimum
    assert content.startswith(b"RIFF")
    assert b"WAVE" in content[:12]


def test_api_generate_with_hindi_manifest_voice(client: TestClient) -> None:
    """Test POST /api/generate with registered Hindi voice."""
    payload = {
        "text": "नमस्ते! यह एक परीक्षण संदेश है।",
        "language": "hi",
        "speaker_ref_name": "anchor_female_calm",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


def test_api_generate_with_punjabi_manifest_voice(client: TestClient) -> None:
    """Test POST /api/generate with registered Punjabi voice."""
    payload = {
        "text": "ਇੱਕ ਵਾਰ ਦੀ ਗੱਲ ਹੈ, ਪੁਰਾਣੇ ਪਿੰਡ ਵਿੱਚ ਇੱਕ ਬਜ਼ੁਰਗ ਕਹਾਣੀਕਾਰ ਰਹਿੰਦਾ ਸੀ।",
        "language": "pa",
        "speaker_ref_name": "storyteller_punjabi_elder",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


def test_api_generate_empty_text_returns_400(client: TestClient) -> None:
    """Test 6: POST /api/generate rejects whitespace-only or empty text with 400."""
    payload = {
        "text": "   ",
        "language": "hi",
        "speaker_ref_name": "anchor_male_energetic",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "empty" in detail.lower() or "whitespace" in detail.lower()


def test_api_generate_unknown_voice_returns_400(client: TestClient) -> None:
    """Test POST /api/generate rejects unknown voice identifier."""
    payload = {
        "text": "Test text for synthesis.",
        "language": "en",
        "speaker_ref_name": "non_existent_voice_profile",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "unknown voice" in detail.lower()


def test_api_generate_unsupported_language_for_voice_returns_400(client: TestClient) -> None:
    """Test POST /api/generate rejects mismatch between selected language and voice profile."""
    # anchor_female_calm supports only 'hi'
    payload = {
        "text": "English text requested with Hindi-only voice.",
        "language": "en",
        "speaker_ref_name": "anchor_female_calm",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "does not support language" in detail.lower()


def test_api_generate_with_mock_synthesizer_signature_verification(client: TestClient) -> None:
    """Test POST /api/generate invokes Synthesizer with genuine IndicF5 signature arguments."""
    mock_engine = MagicMock()
    # 0.2s of sine wave at 24000Hz
    t = np.linspace(0, 0.2, 4800, endpoint=False)
    synthetic_wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mock_engine.synthesize.return_value = (synthetic_wave, 24000)

    app.state.synthesizer = mock_engine

    target_text = "Synthesized with mock engine."
    payload = {
        "text": target_text,
        "language": "hi",
        "speaker_ref_name": "anchor_female_calm",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")

    # Assert genuine signature arguments passed to engine.synthesize
    mock_engine.synthesize.assert_called_once()
    call_args, call_kwargs = mock_engine.synthesize.call_args
    assert call_args[0] == target_text
    assert "anchor_female_calm.wav" in call_kwargs.get("ref_audio_path", "")
    assert Path(call_kwargs["ref_audio_path"]).is_file()  # Verifies the path actually exists on disk
    expected_ref_text = voices.registry.get_voice_metadata("anchor_female_calm")["ref_text"]
    assert call_kwargs.get("ref_text") == expected_ref_text


def test_api_generate_missing_reference_file_returns_500(client: TestClient) -> None:
    """Test POST /api/generate returns 500 when reference audio file does not exist on disk."""
    # Force a custom voice entry with a missing reference audio path
    app.state.custom_voices["broken_ref_voice"] = {
        "id": "broken_ref_voice",
        "speaker_ref_name": "broken_ref_voice",
        "name": "Broken Reference Voice",
        "language": ["en"],
        "description": "Voice with missing reference audio",
        "ref_text": "Some sample reference text",
        "path": "voices/refs/missing_file_on_disk.wav",
    }

    payload = {
        "text": "Hello world.",
        "language": "en",
        "speaker_ref_name": "broken_ref_voice",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 500
    detail = response.json().get("detail", "")
    assert "does not exist on disk" in detail.lower()


def test_manifest_loading_internal_error_returns_500(client: TestClient) -> None:
    """Test eliminate silent exception swallowing: corrupt manifest raises explicit 500."""
    with patch("voices.registry.load_manifest", side_effect=RuntimeError("Corrupt manifest disk failure")):
        # GET /api/voices returns 500
        voices_res = client.get("/api/voices")
        assert voices_res.status_code == 500
        assert "manifest loading failed" in voices_res.json().get("detail", "").lower()

        # POST /api/generate returns 500
        gen_res = client.post(
            "/api/generate",
            json={
                "text": "Sample text",
                "language": "hi",
                "speaker_ref_name": "anchor_female_calm",
            },
        )
        assert gen_res.status_code == 500
        assert "manifest loading failed" in gen_res.json().get("detail", "").lower()


def test_healthz_endpoint_preserved(client: TestClient) -> None:
    """Test 7: Verify /healthz contract remains functional and accurate."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AudioGen"
    assert data["port"] == 17000
