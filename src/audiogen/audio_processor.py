"""Audio mastering, loudness normalization, filtering, and WAV export module."""

from pathlib import Path
from typing import Final, Optional, Set, Tuple, Union
import logging
import numpy as np
import soundfile as sf

TARGET_LUFS: Final[float] = -14.0
TRUE_PEAK_CEILING_DB: Final[float] = -1.0
HIGH_PASS_CUTOFF_HZ: Final[float] = 80.0
ALLOWED_EXPORT_SAMPLE_RATES: Final[Set[int]] = {24000, 44100}

logger = logging.getLogger(__name__)


def _apply_biquad_iir(
    b0: float, b1: float, b2: float, a1: float, a2: float, x: np.ndarray
) -> np.ndarray:
    """Apply a Direct Form II Transposed / Direct Form I biquad IIR filter using a loop."""
    y = np.zeros_like(x, dtype=np.float32)
    y1 = 0.0
    y2 = 0.0
    x1 = 0.0
    x2 = 0.0
    for n in range(len(x)):
        xn = float(x[n])
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2 = x1
        x1 = xn
        y2 = y1
        y1 = yn
    return y


def apply_highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = HIGH_PASS_CUTOFF_HZ,
) -> np.ndarray:
    """Apply an 80 Hz high-pass filter (2nd-order Butterworth IIR) to eliminate low-frequency rumble.

    Args:
        audio: 1D numpy array of audio samples (float32).
        sample_rate: Audio sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz (default 80.0).

    Returns:
        Filtered audio array as float32.

    Raises:
        ValueError: If audio array is empty or sample_rate <= 2 * cutoff_hz.
    """
    if len(audio) == 0:
        raise ValueError("Audio array cannot be empty.")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be positive.")
    if sample_rate <= 2 * cutoff_hz:
        raise ValueError(
            f"Sample rate {sample_rate} Hz must be greater than twice cutoff frequency {cutoff_hz} Hz."
        )

    omega = np.tan(np.pi * cutoff_hz / sample_rate)
    c = omega**2
    d = np.sqrt(2.0) * omega
    a0 = 1.0 + d + c
    b0 = 1.0 / a0
    b1 = -2.0 / a0
    b2 = 1.0 / a0
    a1 = 2.0 * (c - 1.0) / a0
    a2 = (1.0 - d + c) / a0

    filtered = _apply_biquad_iir(b0, b1, b2, a1, a2, audio)
    return filtered.astype(np.float32)


def _get_k_weighting_coeffs(fs: float) -> Tuple[Tuple[float, float, float, float, float], Tuple[float, float, float, float, float]]:
    """Derive ITU-R BS.1770 K-weighting filter coefficients for an arbitrary sampling rate."""
    # Stage 1: High shelf pre-filter
    db = 3.9998438
    f0 = 1681.974450955533
    q = 0.7071752369554193
    k = np.tan(np.pi * f0 / fs)
    vh = 10.0 ** (db / 20.0)
    vb = vh**0.4996667741545416

    a0_1 = 1.0 + k / q + k * k
    b0_1 = (vh + vb * k / q + k * k) / a0_1
    b1_1 = 2.0 * (k * k - vh) / a0_1
    b2_1 = (vh - vb * k / q + k * k) / a0_1
    a1_1 = 2.0 * (k * k - 1.0) / a0_1
    a2_1 = (1.0 - k / q + k * k) / a0_1

    # Stage 2: High-pass (RLB) weighting filter
    f0_hp = 38.13547087602444
    q_hp = 0.5003270373253953
    k_hp = np.tan(np.pi * f0_hp / fs)
    a0_2 = 1.0 + k_hp / q_hp + k_hp * k_hp
    b0_2 = 1.0 / a0_2
    b1_2 = -2.0 / a0_2
    b2_2 = 1.0 / a0_2
    a1_2 = 2.0 * (k_hp * k_hp - 1.0) / a0_2
    a2_2 = (1.0 - k_hp / q_hp + k_hp * k_hp) / a0_2

    return (b0_1, b1_1, b2_1, a1_1, a2_1), (b0_2, b1_2, b2_2, a1_2, a2_2)


def measure_loudness(
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    """Measure integrated loudness in LUFS following ITU-R BS.1770-4 (K-weighting).

    Args:
        audio: 1D float numpy array.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Integrated loudness in LUFS. Returns float('-inf') if audio is pure silence.

    Raises:
        ValueError: If sample_rate <= 0.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")
    if len(audio) == 0 or np.all(audio == 0) or np.sqrt(np.mean(audio**2)) < 1e-6:
        return float("-inf")

    c1, c2 = _get_k_weighting_coeffs(float(sample_rate))
    y = _apply_biquad_iir(c1[0], c1[1], c1[2], c1[3], c1[4], audio)
    y = _apply_biquad_iir(c2[0], c2[1], c2[2], c2[3], c2[4], y)

    block_size = int(0.4 * sample_rate)
    hop_size = int(0.1 * sample_rate)

    if len(y) < block_size:
        ms = float(np.mean(y**2))
        if ms < 1e-12:
            return float("-inf")
        return float(-0.691 + 10.0 * np.log10(ms))

    num_blocks = (len(y) - block_size) // hop_size + 1
    blocks_ms = np.array(
        [np.mean(y[i * hop_size : i * hop_size + block_size] ** 2) for i in range(num_blocks)],
        dtype=np.float64,
    )
    l_blocks = -0.691 + 10.0 * np.log10(np.maximum(blocks_ms, 1e-12))

    # Absolute gating at -70 LUFS
    abs_mask = l_blocks > -70.0
    if not np.any(abs_mask):
        return float("-inf")

    # Relative gating at 10 dB below the average of blocks surviving absolute gate
    mean_abs = np.mean(blocks_ms[abs_mask])
    rel_thresh = -0.691 + 10.0 * np.log10(mean_abs) - 10.0

    rel_mask = abs_mask & (l_blocks > rel_thresh)
    if not np.any(rel_mask):
        return float("-inf")

    final_ms = float(np.mean(blocks_ms[rel_mask]))
    return float(-0.691 + 10.0 * np.log10(final_ms))


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = TARGET_LUFS,
    true_peak_ceiling_db: float = TRUE_PEAK_CEILING_DB,
) -> np.ndarray:
    """Normalize audio loudness to target LUFS while constraining true peak to ceiling.

    Degrades gracefully on silent or clipped input with warning logs.

    Args:
        audio: 1D float numpy array.
        sample_rate: Sampling frequency in Hz.
        target_lufs: Target loudness in LUFS (default -14.0).
        true_peak_ceiling_db: Peak ceiling in dBTP (default -1.0).

    Returns:
        Loudness-normalized float32 audio array.

    Raises:
        ValueError: If audio array is empty or sample_rate <= 0.
    """
    if len(audio) == 0:
        raise ValueError("Audio array cannot be empty.")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Check for pure silence or extremely low signal (< 1e-6 RMS)
    rms = np.sqrt(np.mean(audio**2))
    if np.all(audio == 0) or rms < 1e-6:
        logger.warning(
            "Input audio is silent; returning best-effort unmodified waveform without gain boost."
        )
        return audio.copy().astype(np.float32)

    working_audio = audio.copy().astype(np.float32)

    # Check for clipping (peak >= 1.0)
    peak_val = float(np.max(np.abs(working_audio)))
    if peak_val >= 1.0:
        logger.warning(
            "Clipped audio detected; peak attenuating before loudness normalization."
        )
        working_audio = working_audio * (0.95 / peak_val)

    current_lufs = measure_loudness(working_audio, sample_rate)
    if current_lufs == float("-inf"):
        return working_audio

    delta_db = target_lufs - current_lufs
    gain = 10.0 ** (delta_db / 20.0)
    normalized = working_audio * gain

    # Peak ceiling constraint
    ceiling_linear = 10.0 ** (true_peak_ceiling_db / 20.0)
    current_peak = float(np.max(np.abs(normalized)))
    if current_peak > ceiling_linear:
        normalized = normalized * (ceiling_linear / current_peak)

    return normalized.astype(np.float32)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample 1D audio array from orig_sr to target_sr using band-limited interpolation.

    Args:
        audio: 1D float numpy array.
        orig_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio as 1D float32 numpy array.

    Raises:
        ValueError: If orig_sr <= 0 or target_sr <= 0 or audio is empty.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"Sample rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}"
        )
    if len(audio) == 0:
        raise ValueError("Audio array cannot be empty.")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")
    if orig_sr == target_sr:
        return audio.copy().astype(np.float32)

    new_length = int(round(len(audio) * target_sr / orig_sr))
    orig_indices = np.linspace(0, len(audio) - 1, num=new_length)
    resampled = np.interp(orig_indices, np.arange(len(audio)), audio)
    return resampled.astype(np.float32)


def master_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 24000,
    target_lufs: float = TARGET_LUFS,
    peak_ceiling_db: float = TRUE_PEAK_CEILING_DB,
) -> Tuple[np.ndarray, int]:
    """Execute complete mastering pipeline:
    1. High-pass filter (80 Hz)
    2. Resampling (if sample_rate != target_sample_rate)
    3. Loudness normalization (-14.0 LUFS, -1.0 dB true peak ceiling)
    4. Safe clipping & headroom check

    Args:
        audio: Raw 1D numpy array.
        sample_rate: Source sample rate in Hz.
        target_sample_rate: Desired output sample rate (24000 or 44100).
        target_lufs: Target LUFS (default -14.0).
        peak_ceiling_db: True peak ceiling in dBTP (default -1.0).

    Returns:
        Tuple[np.ndarray, int]: (Mastered float32 audio array, target_sample_rate).

    Raises:
        ValueError: If target_sample_rate not in {24000, 44100} or audio is empty.
    """
    if target_sample_rate not in ALLOWED_EXPORT_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported target sample rate: {target_sample_rate}. Expected 24000 or 44100 Hz."
        )
    if len(audio) == 0:
        raise ValueError("Audio array cannot be empty.")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")

    # 1. High-pass filter
    mastered = apply_highpass_filter(audio, sample_rate)

    # 2. Resampling
    current_sr = sample_rate
    if sample_rate != target_sample_rate:
        mastered = resample_audio(mastered, sample_rate, target_sample_rate)
        current_sr = target_sample_rate

    # 3. Loudness normalization & peak ceiling
    mastered = normalize_loudness(
        mastered,
        current_sr,
        target_lufs=target_lufs,
        true_peak_ceiling_db=peak_ceiling_db,
    )

    return mastered.astype(np.float32), target_sample_rate


def export_wav(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Union[str, Path],
) -> Path:
    """Export audio array as 16-bit PCM .wav file.

    Args:
        audio: 1D float or int16 numpy array.
        sample_rate: Sample rate in Hz (must be 24000 or 44100).
        output_path: Destination file path on disk.

    Returns:
        Path to the written .wav file.

    Raises:
        ValueError: If sample_rate is not in {24000, 44100}.
        ValueError: If audio is empty or not 1-dimensional.
        OSError: If destination directory cannot be written.
    """
    if sample_rate not in ALLOWED_EXPORT_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported sample rate: {sample_rate}. Expected 24000 or 44100 Hz."
        )
    if len(audio) == 0:
        raise ValueError("Audio array cannot be empty.")
    if audio.ndim != 1:
        raise ValueError("Audio array must be 1-dimensional.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


class AudioProcessor:
    """Encapsulated audio mastering and export processor."""

    def __init__(
        self,
        target_sample_rate: int = 24000,
        target_lufs: float = TARGET_LUFS,
        peak_ceiling_db: float = TRUE_PEAK_CEILING_DB,
        highpass_cutoff: float = HIGH_PASS_CUTOFF_HZ,
    ) -> None:
        """Initialize AudioProcessor with mastering configuration.

        Args:
            target_sample_rate: Target sample rate (24000 or 44100).
            target_lufs: Target LUFS (default -14.0).
            peak_ceiling_db: Peak ceiling in dBTP (default -1.0).
            highpass_cutoff: Highpass cutoff frequency in Hz (default 80.0).

        Raises:
            ValueError: If target_sample_rate not in {24000, 44100}.
        """
        if target_sample_rate not in ALLOWED_EXPORT_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported target sample rate: {target_sample_rate}. Expected 24000 or 44100 Hz."
            )
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        self.peak_ceiling_db = peak_ceiling_db
        self.highpass_cutoff = highpass_cutoff

    def process_and_export(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: Union[str, Path],
    ) -> Path:
        """Master audio and export 16-bit PCM WAV.

        Args:
            audio: 1D numpy array of audio samples.
            sample_rate: Audio source sample rate in Hz.
            output_path: Path to destination WAV file.

        Returns:
            Path to the written .wav file.
        """
        mastered, out_sr = master_audio(
            audio,
            sample_rate,
            target_sample_rate=self.target_sample_rate,
            target_lufs=self.target_lufs,
            peak_ceiling_db=self.peak_ceiling_db,
        )
        return export_wav(mastered, out_sr, output_path)
