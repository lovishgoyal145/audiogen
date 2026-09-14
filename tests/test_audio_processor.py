"""Unit tests for audio mastering, loudness normalization, filtering, and WAV export."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from audiogen.audio_processor import (
    ALLOWED_EXPORT_SAMPLE_RATES,
    HIGH_PASS_CUTOFF_HZ,
    TARGET_LUFS,
    TRUE_PEAK_CEILING_DB,
    AudioProcessor,
    apply_highpass_filter,
    export_wav,
    master_audio,
    measure_loudness,
    normalize_loudness,
    resample_audio,
)


@pytest.fixture
def synthetic_sine_100hz() -> np.ndarray:
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 100 * t).astype(np.float32)


@pytest.fixture
def synthetic_rumble_30hz() -> np.ndarray:
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 30 * t).astype(np.float32)


@pytest.fixture
def synthetic_silent_audio() -> np.ndarray:
    return np.zeros(24000, dtype=np.float32)


@pytest.fixture
def synthetic_clipped_audio() -> np.ndarray:
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # High amplitude sine wave clamped to hard clipping threshold
    raw = 3.0 * np.sin(2 * np.pi * 220 * t)
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def synthetic_quiet_speech_proxy() -> np.ndarray:
    sr = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Multi-harmonic low-amplitude signal simulating quiet speech (~ -32 LUFS)
    proxy = (
        0.03 * np.sin(2 * np.pi * 150 * t)
        + 0.02 * np.sin(2 * np.pi * 300 * t)
        + 0.015 * np.sin(2 * np.pi * 600 * t)
    )
    return proxy.astype(np.float32)


def test_highpass_filter_attenuation(synthetic_rumble_30hz):
    """Verify 80 Hz high-pass filter reduces 30 Hz power by >= 15 dB while preserving 1000 Hz within 0.5 dB."""
    sr = 24000
    filtered_30 = apply_highpass_filter(synthetic_rumble_30hz, sr)

    # Calculate power after filter settling time (second half of signal)
    half = sr // 2
    p_in_30 = np.mean(synthetic_rumble_30hz[half:] ** 2)
    p_out_30 = np.mean(filtered_30[half:] ** 2)
    att_30_db = 10.0 * np.log10(p_out_30 / p_in_30)
    assert att_30_db <= -15.0  # Reduced by at least 15 dB

    # 1000 Hz preservation
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_1000 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    filtered_1000 = apply_highpass_filter(sig_1000, sr)

    p_in_1000 = np.mean(sig_1000[half:] ** 2)
    p_out_1000 = np.mean(filtered_1000[half:] ** 2)
    att_1000_db = 10.0 * np.log10(p_out_1000 / p_in_1000)
    assert abs(att_1000_db) < 0.5


def test_highpass_filter_error_contracts():
    """Verify highpass filter raises ValueError on empty input or invalid sample rate."""
    with pytest.raises(ValueError, match="Audio array cannot be empty."):
        apply_highpass_filter(np.array([], dtype=np.float32), 24000)

    with pytest.raises(ValueError, match="must be greater than twice cutoff"):
        apply_highpass_filter(np.ones(100, dtype=np.float32), 100, cutoff_hz=80.0)


def test_loudness_measurement_silence(synthetic_silent_audio):
    """Verify pure silence returns -inf LUFS."""
    assert measure_loudness(synthetic_silent_audio, 24000) == float("-inf")
    assert measure_loudness(np.array([], dtype=np.float32), 24000) == float("-inf")


def test_loudness_normalization_target_and_ceiling(synthetic_quiet_speech_proxy):
    """Verify normalization targets -14.0 ± 0.5 LUFS and never breaches true peak ceiling."""
    sr = 24000
    normalized = normalize_loudness(
        synthetic_quiet_speech_proxy,
        sr,
        target_lufs=-14.0,
        true_peak_ceiling_db=-1.0,
    )

    measured_lufs = measure_loudness(normalized, sr)
    assert abs(measured_lufs - (-14.0)) <= 0.5

    ceiling_linear = 10.0 ** (-1.0 / 20.0)
    max_peak = np.max(np.abs(normalized))
    assert max_peak <= ceiling_linear + 1e-5


def test_graceful_degradation_on_silent_and_clipped(
    synthetic_silent_audio, synthetic_clipped_audio, caplog
):
    """Verify silent audio and clipped audio degrade gracefully with warnings."""
    sr = 24000

    # 1. Silent audio does not throw and returns unmodified
    out_silent = normalize_loudness(synthetic_silent_audio, sr)
    assert np.all(out_silent == 0.0)
    assert any("Input audio is silent" in record.message for record in caplog.records)

    # 2. Clipped audio logs warning and respects peak ceiling
    caplog.clear()
    out_clipped = normalize_loudness(synthetic_clipped_audio, sr)
    assert any("Clipped audio detected" in record.message for record in caplog.records)
    ceiling_linear = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(out_clipped)) <= ceiling_linear + 1e-5


def test_resample_audio():
    """Verify resampling transforms array length accurately."""
    sr_orig = 24000
    sr_target = 44100
    audio = np.ones(sr_orig, dtype=np.float32)
    resampled = resample_audio(audio, sr_orig, sr_target)
    assert len(resampled) == sr_target
    assert resampled.dtype == np.float32

    # Same sample rate returns copy
    same = resample_audio(audio, sr_orig, sr_orig)
    assert len(same) == sr_orig

    with pytest.raises(ValueError, match="Sample rates must be positive"):
        resample_audio(audio, -1, 44100)


def test_master_audio_pipeline(synthetic_quiet_speech_proxy):
    """Verify master_audio pipeline executes filter, resample, and loudness normalization."""
    sr = 24000
    mastered, out_sr = master_audio(
        synthetic_quiet_speech_proxy,
        sample_rate=sr,
        target_sample_rate=44100,
        target_lufs=-14.0,
        peak_ceiling_db=-1.0,
    )
    assert out_sr == 44100
    assert len(mastered) == int(len(synthetic_quiet_speech_proxy) * 44100 / sr)
    assert np.max(np.abs(mastered)) <= (10.0 ** (-1.0 / 20.0)) + 1e-5

    with pytest.raises(ValueError, match="Unsupported target sample rate"):
        master_audio(synthetic_quiet_speech_proxy, sr, target_sample_rate=16000)


def test_export_wav_and_audio_processor(tmp_path, synthetic_sine_100hz):
    """Verify 16-bit PCM WAV export at 24000 Hz and 44100 Hz, plus error rejection."""
    out_path_24k = tmp_path / "test_24000.wav"
    export_wav(synthetic_sine_100hz, 24000, out_path_24k)

    assert out_path_24k.exists()
    info = sf.info(str(out_path_24k))
    assert info.samplerate == 24000
    assert info.subtype == "PCM_16"
    assert info.channels == 1

    # 44100 Hz export
    out_path_44k = tmp_path / "test_44100.wav"
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    audio_44k = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    export_wav(audio_44k, 44100, out_path_44k)
    info_44k = sf.info(str(out_path_44k))
    assert info_44k.samplerate == 44100
    assert info_44k.subtype == "PCM_16"

    # Unsupported export sample rates
    with pytest.raises(ValueError, match="Unsupported sample rate: 16000"):
        export_wav(synthetic_sine_100hz, 16000, tmp_path / "invalid.wav")

    with pytest.raises(ValueError, match="Unsupported sample rate: 48000"):
        export_wav(synthetic_sine_100hz, 48000, tmp_path / "invalid.wav")

    # AudioProcessor OOP interface
    proc = AudioProcessor(target_sample_rate=24000)
    out_proc = tmp_path / "proc_mastered.wav"
    result_path = proc.process_and_export(synthetic_sine_100hz, 24000, out_proc)
    assert result_path.exists()
    proc_info = sf.info(str(result_path))
    assert proc_info.samplerate == 24000
    assert proc_info.subtype == "PCM_16"


def test_2d_audio_rejected(tmp_path):
    """Verify 2D audio arrays are strictly rejected across all public DSP functions."""
    audio_2d = np.zeros((2, 1000), dtype=np.float32)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        apply_highpass_filter(audio_2d, 24000)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        measure_loudness(audio_2d, 24000)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        normalize_loudness(audio_2d, 24000)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        resample_audio(audio_2d, 24000, 44100)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        master_audio(audio_2d, 24000)

    with pytest.raises(ValueError, match="Audio array must be 1-dimensional."):
        export_wav(audio_2d, 24000, tmp_path / "2d.wav")

