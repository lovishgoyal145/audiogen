"""End-to-end integration tests for zero-shot voice cloning system (TICKET-010)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import struct
from typing import Any, Generator, Tuple
from unittest.mock import patch
import wave

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from orchestrator.gateway import SessionGateway, create_app as create_gateway_app
from server.app import create_app as create_server_app, process_and_resample_audio
from audiogen.engine import Synthesizer
import voices.registry


def _create_wav(duration_sec: float = 2.0, sample_rate: int = 24000, channels: int = 1, silent: bool = False) -> bytes:
    buf = io.BytesIO()
    n_frames = int(sample_rate * duration_sec)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if silent:
            data = struct.pack(f"<{n_frames * channels}h", *(0 for _ in range(n_frames * channels)))
        else:
            data = struct.pack(
                f"<{n_frames * channels}h",
                *(int(8000 * math.sin(2 * math.pi * 440 * (i // channels) / sample_rate)) for i in range(n_frames * channels)),
            )
        wf.writeframes(data)
    return buf.getvalue()


class MockSynthesizerBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, text: str, ref_audio_path: str = "", ref_text: str = "", **kwargs):
        self.calls.append((text, ref_audio_path, ref_text))
        samples = np.zeros(2400, dtype=np.float32)
        return samples, 24000


@pytest.fixture(autouse=True)
def clean_registry():
    voices.registry.clear_registry_cache()
    yield
    voices.registry.clear_registry_cache()


def test_audio_processing_resampling_and_validation():
    """Verify audio processor converts stereo 44.1kHz to 24kHz mono and validates duration/silence."""
    stereo_44k = _create_wav(duration_sec=2.5, sample_rate=44100, channels=2)
    samples, sr = process_and_resample_audio(stereo_44k)
    assert sr == 24000
    assert len(samples) == int(2.5 * 24000)
    assert np.max(np.abs(samples)) > 0.01

    # Short audio (< 1.0s)
    short_audio = _create_wav(duration_sec=0.7)
    with pytest.raises(Exception) as exc:
        process_and_resample_audio(short_audio)
    assert "too short" in str(exc.value)

    # Silent audio
    silent_audio = _create_wav(duration_sec=2.0, silent=True)
    with pytest.raises(Exception) as exc:
        process_and_resample_audio(silent_audio)
    assert "silent" in str(exc.value)


def test_gateway_session_gate_enforcement():
    """Verify gateway rejects /voices/clone with HTTP 409 unless session state is READY."""
    gw = SessionGateway()
    gw_app = create_gateway_app(gateway=gw)
    with TestClient(gw_app) as client:
        # IDLE
        res_idle = client.post(
            "/voices/clone",
            data={"voice_id": "test_v", "language": "en", "ref_text": "Sample text"},
            files={"file": ("test.wav", _create_wav(2.0), "audio/wav")},
        )
        assert res_idle.status_code == 409
        assert res_idle.json()["status"] == "IDLE"

        # STARTING
        gw._state = "STARTING"
        res_starting = client.post(
            "/voices/clone",
            data={"voice_id": "test_v", "language": "en", "ref_text": "Sample text"},
            files={"file": ("test.wav", _create_wav(2.0), "audio/wav")},
        )
        assert res_starting.status_code == 409
        assert res_starting.json()["status"] == "STARTING"

        # ERROR
        gw._state = "ERROR"
        res_error = client.post(
            "/voices/clone",
            data={"voice_id": "test_v", "language": "en", "ref_text": "Sample text"},
            files={"file": ("test.wav", _create_wav(2.0), "audio/wav")},
        )
        assert res_error.status_code == 409
        assert res_error.json()["status"] == "ERROR"


def test_end_to_end_voice_cloning_and_synthesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify full flow: clone voice on server -> register -> query list -> synthesize."""
    mock_backend = MockSynthesizerBackend()
    synth = Synthesizer(backend=mock_backend)
    server_app = create_server_app(synthesizer=synth, auth_token="test-token")

    with TestClient(server_app) as server_client:
        audio_bytes = _create_wav(duration_sec=3.5)
        try:
            clone_res = server_client.post(
                "/voices/clone",
                data={
                    "voice_id": "e2e_cloned_voice",
                    "name": "E2E Cloned Voice",
                    "language": "en",
                    "ref_text": "This is the transcript of the reference audio.",
                    "description": "Voice cloned for integration test",
                },
                files={"file": ("sample.wav", audio_bytes, "audio/wav")},
                headers={"Authorization": "Bearer test-token"},
            )
            assert clone_res.status_code == 201
            voice_data = clone_res.json()["voice"]
            assert voice_data["id"] == "e2e_cloned_voice"
            assert voice_data["language"] == ["en"]
            assert Path(voice_data["path"]).is_file()

            # Query voices
            list_res = server_client.get("/voices?language=en")
            assert list_res.status_code == 200
            assert "e2e_cloned_voice" in list_res.json()["voices"]

            # Synthesize with the cloned voice
            gen_res = server_client.post(
                "/generate",
                json={
                    "text": "Speech synthesized with freshly cloned voice.",
                    "language": "en",
                    "speaker_ref_name": "e2e_cloned_voice",
                },
                headers={"Authorization": "Bearer test-token"},
            )
            assert gen_res.status_code == 200
            assert gen_res.headers["content-type"] == "audio/wav"
            assert len(mock_backend.calls) >= 1
            last_call = mock_backend.calls[-1]
            assert "Speech synthesized" in last_call[0]
            assert last_call[2] == "This is the transcript of the reference audio."
        finally:
            schema_file = voices.registry.DEFAULT_MANIFEST_PATH
            if schema_file.is_file():
                with open(schema_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if "e2e_cloned_voice" in d:
                    del d["e2e_cloned_voice"]
                    with open(schema_file, "w", encoding="utf-8") as f:
                        json.dump(d, f, indent=2, ensure_ascii=False)
            ref_file = voices.registry.REPO_ROOT / "voices" / "refs" / "e2e_cloned_voice.wav"
            if ref_file.exists():
                ref_file.unlink()
            voices.registry.clear_registry_cache()


def test_gateway_clone_proxy_and_local_persistence(tmp_path: Path):
    """Verify gateway proxies clone to remote worker and persists locally when READY."""
    gw = SessionGateway()
    gw._state = "READY"
    gw._tunnel_url = "https://tunnel.example.com"
    gw.bearer_token = "token-123"

    audio_bytes = _create_wav(2.0)
    mock_remote_voice = {
        "id": "gw_persisted_voice",
        "name": "Gw Persisted Voice",
        "language": ["en"],
        "ref_text": "This is transcript",
        "path": "voices/refs/gw_persisted_voice.wav",
        "duration_seconds": 2.0,
    }

    async def mock_remote_clone(url: str, files: dict, data: dict, headers: dict, **kwargs):
        assert "voices/clone" in url
        return httpx.Response(
            status_code=201,
            json={"status": "success", "voice": mock_remote_voice},
            headers={"content-type": "application/json"},
        )

    gw_app = create_gateway_app(gateway=gw)
    persisted_path = Path("voices/refs/gw_persisted_voice.wav")
    try:
        with patch("httpx.AsyncClient.post", side_effect=mock_remote_clone), \
             patch("voices.registry.register_voice") as mock_reg, \
             TestClient(gw_app) as client:
            res = client.post(
                "/voices/clone",
                data={
                    "voice_id": "gw_persisted_voice",
                    "name": "Gw Persisted Voice",
                    "language": "en",
                    "ref_text": "This is transcript",
                },
                files={"file": ("ref.wav", audio_bytes, "audio/wav")},
            )
            assert res.status_code == 201
            assert res.json()["status"] == "success"
            mock_reg.assert_called_once()
    finally:
        if persisted_path.exists():
            persisted_path.unlink()

