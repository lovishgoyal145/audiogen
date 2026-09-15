"""Unit tests for IndicF5 speech synthesis engine and mock backend integration."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import struct
import sys
import wave
import numpy as np
import pytest

from audiogen.engine import Synthesizer
from voices.registry import VoiceRecord


def _create_dummy_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 24000) -> None:
    """Helper to create a minimal valid 16-bit PCM mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_sec)
    frames = struct.pack(f"<{num_frames}h", *(0 for _ in range(num_frames)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


class MockTTSBackend:
    """Mock inference backend simulating IndicF5 speech synthesis."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.calls = []
        self.to_called_with = None
        self.eval_called = False

    def to(self, device: str) -> "MockTTSBackend":
        self.to_called_with = device
        return self

    def eval(self) -> "MockTTSBackend":
        self.eval_called = True
        return self

    def __call__(self, text: str, ref_audio_path: Any = None, ref_text: str = "", **kwargs: Any):
        if "ref_audio_path" in kwargs:
            ref_audio_path = kwargs["ref_audio_path"]
        if "ref_text" in kwargs:
            ref_text = kwargs["ref_text"]
        self.calls.append((text, ref_audio_path, ref_text))
        duration_sec = 0.5
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return waveform, self.sample_rate

    def synthesize(self, text: str, ref_audio_path: Any = None, ref_text: str = "", **kwargs: Any):
        return self(text, ref_audio_path=ref_audio_path, ref_text=ref_text, **kwargs)


@pytest.fixture
def mock_backend() -> MockTTSBackend:
    return MockTTSBackend()


@pytest.fixture
def dummy_model_file(tmp_path: Path) -> Path:
    model_file = tmp_path / "model_weights.pt"
    model_file.write_bytes(b"dummy_weights_content")
    return model_file


@pytest.fixture
def dummy_ref_audio(tmp_path: Path) -> Path:
    wav_file = tmp_path / "ref_speaker.wav"
    _create_dummy_wav(wav_file)
    return wav_file


def test_cpu_fallback_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch, mock_backend: MockTTSBackend) -> None:
    """Verify requesting CUDA when unavailable safely logs warning and falls back to CPU."""
    if "torch" in sys.modules:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    synthesizer = Synthesizer(device="cuda", backend=mock_backend)
    assert synthesizer.device == "cpu"


def test_cuda_detection_unexpected_exception_trapping(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, mock_backend: MockTTSBackend
) -> None:
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


def test_model_path_resolution_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_backend: MockTTSBackend
) -> None:
    """Verify environment variable injection and constructor override precedence for model_path."""
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


def test_repo_id_resolution_and_precedence(monkeypatch: pytest.MonkeyPatch, mock_backend: MockTTSBackend) -> None:
    """Verify repo_id resolution precedence: constructor > INDIC_F5_MODEL_REPO > default."""
    monkeypatch.delenv("INDIC_F5_MODEL_REPO", raising=False)
    monkeypatch.delenv("INDIC_TTS_MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    # 1. Default repo id
    synth_default = Synthesizer(backend=mock_backend)
    assert synth_default.repo_id == "ai4bharat/IndicF5"

    # 2. Environment variable
    monkeypatch.setenv("INDIC_F5_MODEL_REPO", "custom/indicf5-repo")
    synth_env = Synthesizer(backend=mock_backend)
    assert synth_env.repo_id == "custom/indicf5-repo"

    # 3. Constructor argument override
    synth_override = Synthesizer(repo_id="override/indicf5", backend=mock_backend)
    assert synth_override.repo_id == "override/indicf5"


def test_revision_pinning(monkeypatch: pytest.MonkeyPatch, mock_backend: MockTTSBackend) -> None:
    """Verify revision pinning via constructor argument and INDIC_F5_MODEL_REVISION env var."""
    monkeypatch.delenv("INDIC_F5_MODEL_REVISION", raising=False)

    # 1. Default revision is None
    synth_default = Synthesizer(backend=mock_backend)
    assert synth_default.revision is None

    # 2. Env var resolution
    monkeypatch.setenv("INDIC_F5_MODEL_REVISION", "git_sha_abc123")
    synth_env = Synthesizer(backend=mock_backend)
    assert synth_env.revision == "git_sha_abc123"

    # 3. Constructor override
    synth_override = Synthesizer(revision="git_sha_xyz789", backend=mock_backend)
    assert synth_override.revision == "git_sha_xyz789"


def test_device_placement_and_eval_mode(mock_backend: MockTTSBackend) -> None:
    """Verify Synthesizer explicitly calls .to(device) and .eval() on the backend."""
    synth = Synthesizer(device="cpu", backend=mock_backend)
    assert mock_backend.to_called_with == "cpu"
    assert mock_backend.eval_called is True


def test_nonexistent_model_path_raises_file_not_found(tmp_path: Path) -> None:
    """Verify providing a non-existent model path raises FileNotFoundError."""
    non_existent = tmp_path / "ghost_model.pt"
    with pytest.raises(FileNotFoundError, match="Model path does not exist:"):
        Synthesizer(model_path=non_existent)


def test_invalid_sample_rate_raises(dummy_model_file: Path, mock_backend: MockTTSBackend) -> None:
    """Verify non-positive sample rate raises ValueError."""
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        Synthesizer(model_path=dummy_model_file, sample_rate=-1, backend=mock_backend)


def test_synthesizer_indicf5_synthesis_success(dummy_ref_audio: Path, mock_backend: MockTTSBackend) -> None:
    """Verify synthesize against mocked IndicF5 model returns 1D float32 waveform at 24000 Hz."""
    synth = Synthesizer(backend=mock_backend)

    # Hindi synthesis
    text_hi = "नमस्ते भारत, आज का मौसम बहुत अच्छा है।"
    ref_text_hi = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है।"
    wav_hi, sr_hi = synth.synthesize(text_hi, dummy_ref_audio, ref_text_hi)

    assert sr_hi == 24000
    assert isinstance(wav_hi, np.ndarray)
    assert wav_hi.dtype == np.float32
    assert wav_hi.ndim == 1
    assert len(wav_hi) > 0
    assert np.max(np.abs(wav_hi)) <= 1.0

    assert len(mock_backend.calls) == 1
    assert mock_backend.calls[0] == (text_hi, str(dummy_ref_audio), ref_text_hi)

    # Punjabi synthesis
    text_pa = "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ, ਤੁਹਾਡਾ ਕੀ ਹਾਲ ਹੈ?"
    ref_text_pa = "ਇੱਕ ਵਾਰ ਦੀ ਗੱਲ ਹੈ, ਪੁਰਾਣੇ ਪਿੰਡ ਵਿੱਚ ਇੱਕ ਬਜ਼ੁਰਗ ਕਹਾਣੀਕਾਰ ਰਹਿੰਦਾ ਸੀ।"
    wav_pa, sr_pa = synth.synthesize(text_pa, str(dummy_ref_audio), ref_text_pa)

    assert sr_pa == 24000
    assert isinstance(wav_pa, np.ndarray)
    assert wav_pa.dtype == np.float32
    assert wav_pa.ndim == 1
    assert len(wav_pa) > 0
    assert np.max(np.abs(wav_pa)) <= 1.0

    assert len(mock_backend.calls) == 2
    assert mock_backend.calls[1] == (text_pa, str(dummy_ref_audio), ref_text_pa)


def test_synthesizer_backward_compatibility_shims(dummy_ref_audio: Path, mock_backend: MockTTSBackend) -> None:
    """Verify synthesize supports legacy positional and keyword parameters seamlessly."""
    synth = Synthesizer(backend=mock_backend)

    # 1. Legacy positional: synthesize(text, language, speaker_ref)
    wav, sr = synth.synthesize("नमस्ते दुनिया", "hi", str(dummy_ref_audio))
    assert sr == 24000
    assert isinstance(wav, np.ndarray)

    # 2. Legacy keyword: synthesize(text=..., language=..., speaker_ref=...)
    voice_rec = VoiceRecord(
        path=str(dummy_ref_audio),
        ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਵੇਰਵੇ।",
        language=["hi", "pa"],
    )
    wav2, sr2 = synth.synthesize(text="ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ", language="pa", speaker_ref=voice_rec)
    assert sr2 == 24000
    assert isinstance(wav2, np.ndarray)

    # 3. Passing VoiceRecord as positional speaker_ref
    wav3, sr3 = synth.synthesize("नमस्ते भारत", "hi", voice_rec)
    assert sr3 == 24000
    assert isinstance(wav3, np.ndarray)


def test_synthesizer_nonexistent_ref_audio_raises_file_not_found(
    tmp_path: Path, mock_backend: MockTTSBackend
) -> None:
    """Verify calling synthesize with nonexistent ref_audio_path raises FileNotFoundError with zero mock calls."""
    synth = Synthesizer(backend=mock_backend)
    ghost_audio = tmp_path / "non_existent_ref.wav"

    with pytest.raises(FileNotFoundError, match="Reference audio file not found:"):
        synth.synthesize("नमस्ते भारत", ghost_audio, "नमूनों का पाठ")

    # Critical invariant: mock backend must NOT have been called
    assert len(mock_backend.calls) == 0


def test_empty_and_invalid_text_handling(dummy_ref_audio: Path, mock_backend: MockTTSBackend) -> None:
    """Verify whitespace-only, emoji-only, or invalid text types are rejected before inference."""
    synth = Synthesizer(backend=mock_backend)
    valid_ref_text = "वैध संदर्भ पाठ"

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("", dummy_ref_audio, valid_ref_text)

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("   ", dummy_ref_audio, valid_ref_text)

    with pytest.raises(ValueError, match="Text cannot be empty or whitespace-only."):
        synth.synthesize("😀 🔥 🎉", dummy_ref_audio, valid_ref_text)

    with pytest.raises(TypeError, match="Expected text to be str"):
        synth.synthesize(None, dummy_ref_audio, valid_ref_text)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected text to be str"):
        synth.synthesize(12345, dummy_ref_audio, valid_ref_text)  # type: ignore[arg-type]

    # Model call count should remain zero for all rejected calls
    assert len(mock_backend.calls) == 0


def test_empty_and_invalid_ref_text_handling(dummy_ref_audio: Path, mock_backend: MockTTSBackend) -> None:
    """Verify missing, empty, or whitespace-only ref_text is rejected before inference."""
    synth = Synthesizer(backend=mock_backend)
    valid_text = "नमस्ते भारत"

    with pytest.raises(ValueError, match="Reference text cannot be empty or whitespace-only."):
        synth.synthesize(valid_text, dummy_ref_audio, "")

    with pytest.raises(ValueError, match="Reference text cannot be empty or whitespace-only."):
        synth.synthesize(valid_text, dummy_ref_audio, "   ")

    with pytest.raises(TypeError, match="Expected ref_text to be str"):
        synth.synthesize(valid_text, dummy_ref_audio, None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected ref_text to be str"):
        synth.synthesize(valid_text, dummy_ref_audio, 999)  # type: ignore[arg-type]

    assert len(mock_backend.calls) == 0


def test_invalid_ref_audio_path_type(mock_backend: MockTTSBackend) -> None:
    """Verify non-str and non-Path ref_audio_path raises TypeError."""
    synth = Synthesizer(backend=mock_backend)

    with pytest.raises(TypeError, match="Expected ref_audio_path to be str or Path"):
        synth.synthesize("नमस्ते भारत", 12345, "संदर्भ पाठ")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected ref_audio_path to be str or Path"):
        synth.synthesize("नमस्ते भारत", None, "संदर्भ पाठ")  # type: ignore[arg-type]

    assert len(mock_backend.calls) == 0


def test_auto_model_loaded_when_backend_none() -> None:
    """Verify AutoModel.from_pretrained is called with trust_remote_code=True when backend is None."""
    mock_auto_model = MagicMock()
    mock_backend = MockTTSBackend()
    mock_auto_model.from_pretrained.return_value = mock_backend

    with patch("transformers.AutoModel", mock_auto_model):
        synth = Synthesizer(backend=None, repo_id="ai4bharat/IndicF5", revision="git_rev_123")
        mock_auto_model.from_pretrained.assert_called_once_with(
            "ai4bharat/IndicF5", revision="git_rev_123", trust_remote_code=True, cache_dir=None
        )
        assert synth.repo_id == "ai4bharat/IndicF5"
        assert synth._backend == mock_backend


def test_auto_model_loading_failure_raises_runtime_error() -> None:
    """Verify fail-fast exception: AutoModel.from_pretrained failure raises descriptive RuntimeError."""
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = RuntimeError("Checkpoint corrupted or blocked")

    with patch("transformers.AutoModel", mock_auto_model):
        with pytest.raises(RuntimeError, match="Failed to load IndicF5 model"):
            Synthesizer(backend=None, repo_id="ai4bharat/IndicF5")


def test_waveform_normalization_clipping(dummy_ref_audio: Path) -> None:
    """Verify waveforms exceeding [-1.0, 1.0] are safely peak-normalized."""

    class HotBackend:
        def __call__(self, text: str, ref_audio_path: str, ref_text: str):
            # Return high-amplitude waveform with peak 2.5
            return np.array([0.0, 2.5, -2.5, 1.0], dtype=np.float32), 24000

    synth = Synthesizer(backend=HotBackend())
    wav, sr = synth.synthesize("परीक्षण", dummy_ref_audio, "संदर्भ")

    assert sr == 24000
    assert np.max(np.abs(wav)) <= 1.0
    assert np.isclose(np.max(np.abs(wav)), 1.0)


def test_fallback_acoustic_synthesis_with_local_weights(
    dummy_model_file: Path, dummy_ref_audio: Path
) -> None:
    """Verify deterministic reference-based synthesis when local weights file is provided without backend."""
    synth = Synthesizer(model_path=dummy_model_file, sample_rate=24000)
    assert synth._backend is None

    wav_hi, sr_hi = synth.synthesize("नमस्ते भारत", dummy_ref_audio, "संदर्भ पाठ")
    assert sr_hi == 24000
    assert isinstance(wav_hi, np.ndarray)
    assert wav_hi.ndim == 1
    assert wav_hi.dtype == np.float32
    assert len(wav_hi) > 0

    # Duration scaling with text length
    wav_long, _ = synth.synthesize(
        "नमस्ते भारत आप कैसे हैं मैं बहुत अच्छा हूँ और यह लम्बा वाक्य है",
        dummy_ref_audio,
        "संदर्भ पाठ",
    )
    assert len(wav_long) > len(wav_hi)


def test_synthesizer_delegation_to_kaggle_bridge_when_backend_none(
    dummy_ref_audio: Path,
) -> None:
    """Verify Synthesizer delegates to KaggleExecutionBridge when no local backend/model is loaded."""
    synth = Synthesizer.__new__(Synthesizer)
    synth._backend = None
    synth._model_path = None
    synth._sample_rate = 24000

    test_wave = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    with patch("audiogen.engine.KaggleExecutionBridge.synthesize", return_value=(test_wave, 24000)) as mock_bridge:
        wav, sr = synth.synthesize("नमस्ते भारत", dummy_ref_audio, "संदर्भ पाठ")
        assert sr == 24000
        assert np.array_equal(wav, test_wave)
        mock_bridge.assert_called_once()
        call_args, call_kwargs = mock_bridge.call_args
        assert call_args[0] == "नमस्ते भारत"
        assert str(dummy_ref_audio) in str(call_kwargs.get("ref_audio_path"))
        assert call_kwargs.get("ref_text") == "संदर्भ पाठ"

