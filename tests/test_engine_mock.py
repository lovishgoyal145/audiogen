"""Unit tests for model-agnostic speech synthesis core and mock backend integration."""

from pathlib import Path
from typing import Any
import sys
import numpy as np
import pytest

from core.engine import Synthesizer


class MockTTSBackend:
    """Mock inference backend simulating speech synthesis."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.calls = []

    def __call__(self, normalized_text: str, language: str, speaker_ref: Any = None):
        self.calls.append((normalized_text, language, speaker_ref))
        duration_sec = 0.5
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return waveform, self.sample_rate


@pytest.fixture
def mock_backend() -> MockTTSBackend:
    return MockTTSBackend()


@pytest.fixture
def dummy_model_file(tmp_path) -> Path:
    model_file = tmp_path / "model_weights.pt"
    model_file.write_bytes(b"dummy_weights_content")
    return model_file


def test_cpu_fallback_when_cuda_unavailable(monkeypatch, mock_backend):
    """Verify requesting CUDA when unavailable safely logs warning and falls back to CPU."""
    if "torch" in sys.modules:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    synthesizer = Synthesizer(device="cuda", backend=mock_backend)
    assert synthesizer.device == "cpu"


def test_cuda_detection_unexpected_exception_trapping(monkeypatch, capsys, mock_backend):
    """Verify unexpected errors during CUDA detection emit structured error manifests."""
    class BrokenTorchModule:
        class cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("Unexpected CUDA driver mismatch fault")

    monkeypatch.setitem(sys.modules, "torch", BrokenTorchModule)

    synth = Synthesizer(device="cuda", backend=mock_backend)
    assert synth.device == "cpu"

    captured = capsys.readouterr()
    assert "STRUCTURED_ERROR_MANIFEST:" in captured.err
    assert "Unexpected CUDA driver mismatch fault" in captured.err
    assert "Traceback (most recent call last):" in captured.err


def test_model_path_resolution_and_precedence(tmp_path, monkeypatch, mock_backend):
    """Verify environment variable injection and constructor override precedence."""
    monkeypatch.delenv("INDIC_TTS_MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    p_env = tmp_path / "indic_tts_model.pt"
    p_env.touch()
    p_fallback = tmp_path / "fallback_model.pt"
    p_fallback.touch()
    p_custom = tmp_path / "custom_model.pt"
    p_custom.touch()

    # 1. Primary env var
    monkeypatch.setenv("INDIC_TTS_MODEL_PATH", str(p_env))
    synth_env = Synthesizer(backend=mock_backend)
    assert synth_env.model_path == p_env

    # 2. Fallback env var
    monkeypatch.delenv("INDIC_TTS_MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(p_fallback))
    synth_fallback = Synthesizer(backend=mock_backend)
    assert synth_fallback.model_path == p_fallback

    # 3. Constructor precedence over env vars
    monkeypatch.setenv("INDIC_TTS_MODEL_PATH", str(p_env))
    synth_explicit = Synthesizer(model_path=str(p_custom), backend=mock_backend)
    assert synth_explicit.model_path == p_custom


def test_nonexistent_model_path_raises_file_not_found(tmp_path):
    """Verify providing a non-existent model path raises FileNotFoundError."""
    non_existent = tmp_path / "ghost_model.pt"
    with pytest.raises(FileNotFoundError, match="Model path does not exist:"):
        Synthesizer(model_path=non_existent)


def test_missing_model_path_raises_value_error(monkeypatch):
    """Verify instantiating without model_path, env var, or backend raises explicit ValueError."""
    monkeypatch.delenv("INDIC_TTS_MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(
        ValueError,
        match="Model path must be provided via constructor or INDIC_TTS_MODEL_PATH / MODEL_PATH environment variable.",
    ):
        Synthesizer()


def test_invalid_sample_rate_raises(dummy_model_file):
    """Verify non-positive sample rate raises ValueError."""
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        Synthesizer(model_path=dummy_model_file, sample_rate=-1)


def test_synthesizer_language_contracts(mock_backend):
    """Verify strict language gating rejecting unsupported languages."""
    synth = Synthesizer(backend=mock_backend)

    with pytest.raises(ValueError, match="Unsupported language: en"):
        synth.synthesize("Hello", "en")

    with pytest.raises(ValueError, match="Unsupported language: fr"):
        synth.synthesize("Bonjour", "fr")

    # Supported languages succeed
    wav_hi, sr_hi = synth.synthesize("नमस्ते", "hi")
    assert sr_hi == 24000
    assert isinstance(wav_hi, np.ndarray)

    wav_pa, sr_pa = synth.synthesize("ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ", "pa")
    assert sr_pa == 24000
    assert isinstance(wav_pa, np.ndarray)


def test_empty_and_invalid_text_handling(mock_backend):
    """Verify whitespace-only or noise-only text is rejected."""
    synth = Synthesizer(backend=mock_backend)

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("", "hi")

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("   ", "hi")

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("😀 🔥 🎉", "hi")

    with pytest.raises(TypeError, match="Expected text to be str"):
        synth.synthesize(None, "hi")  # type: ignore

    with pytest.raises(TypeError, match="Expected language to be str"):
        synth.synthesize("नमस्ते", 123)  # type: ignore


def test_normalizer_pre_processing_integration(mock_backend):
    """Verify text is normalized (e.g. digits expanded) before reaching inference backend."""
    synth = Synthesizer(backend=mock_backend)

    synth.synthesize("100 रुपये", "hi")
    last_call = mock_backend.calls[-1]
    assert last_call[0] == "एक सौ रुपये"
    assert last_call[1] == "hi"

    synth.synthesize("100 ਰੁਪਏ", "pa")
    last_call_pa = mock_backend.calls[-1]
    assert last_call_pa[0] == "ਇੱਕ ਸੌ ਰੁਪਏ"
    assert last_call_pa[1] == "pa"


def test_fallback_inference_without_backend(dummy_model_file):
    """Verify genuine fallback acoustic synthesis when backend is None and model_path is provided."""
    synth = Synthesizer(model_path=dummy_model_file, backend=None, sample_rate=24000)

    # 1. Synthesize short Hindi text
    short_text = "नमस्ते भारत"
    wav_short, sr = synth.synthesize(short_text, "hi")
    assert sr == 24000
    assert isinstance(wav_short, np.ndarray)
    assert wav_short.ndim == 1
    assert wav_short.dtype == np.float32
    assert len(wav_short) > 0

    # Waveform is non-silent and within safe headroom
    peak_val = np.max(np.abs(wav_short))
    assert 0.05 < peak_val <= 0.89

    # 2. Duration scaling: longer text produces proportionally longer waveform
    long_text = "नमस्ते भारत आप कैसे हैं मैं बहुत अच्छा हूँ"
    wav_long, _ = synth.synthesize(long_text, "hi")
    assert len(wav_long) > len(wav_short)

    # 3. Text variation produces distinct waveforms
    wav_pa, _ = synth.synthesize("ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ", "pa")
    min_len = min(len(wav_short), len(wav_pa))
    assert not np.allclose(wav_short[:min_len], wav_pa[:min_len])
