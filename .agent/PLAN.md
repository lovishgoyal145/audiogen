# Implementation Plan: Ticket 001 — Indic TTS Core Engine & Audio Post-Processing

## 1. Executive Summary & Objective

The goal of **Ticket 001** is to construct an isolated, model-agnostic Python inference core for Hindi (`hi`) and Punjabi (`pa`). This core handles:
1. **Text Normalization (`core/normalizer.py`):** Script cleaning, symbol/emoji removal, preservation of phonetically critical Indic diacritics (`virama`, `bindi`, `tippi`, `addak`, `danda`), and number-to-words expansion for Devanagari and Gurmukhi.
2. **Inference Synthesizer (`core/engine.py`):** A model-agnostic `Synthesizer` interface with dynamic model weight path resolution (injectable via constructor or environment variables), device resolution with graceful CPU fallback when CUDA is unavailable, strict language gating (`hi`, `pa`), and seamless integration with the text normalizer.
3. **Audio Mastering & Export (`core/audio_processor.py`):** Digital signal processing pipeline providing an 80 Hz high-pass filter (sub-bass / DC offset attenuation), ITU-R BS.1770-4 loudness normalization to `-14.0 LUFS` with a `-1.0 dB` true peak ceiling, graceful degradation on silent or clipped audio, and 16-bit PCM `.wav` export at `24000 Hz` or `44100 Hz`.
4. **Verification & Testing (`tests/`):** Full unit test coverage (`test_normalizer.py`, `test_audio_processor.py`, `test_engine_mock.py`) executing inside `.venv/bin/pytest` with zero external network calls and zero reliance on heavy pre-trained weights.

---

## 2. Allowed Files & Repository Scope

### Allowed Files
- `pyproject.toml`
- `requirements.txt`
- `core/engine.py`
- `core/normalizer.py`
- `core/audio_processor.py`
- `tests/test_normalizer.py`
- `tests/test_audio_processor.py`
- `tests/test_engine_mock.py`
- `.agent/PLAN.md`
- `.agent/tickets/TICKET-001.md`

### Strict Off-Limits
- `server/*`, `batch/*`, `voices/*` — strictly off-limits.
- Existing unit tests (`tests/test_config_and_health.py`) must not be weakened, deleted, or muted.
- Zero hardcoded absolute paths to model weights.
- No network calls or downloads during automated testing.

---

## 3. Exact Function & Class Signatures

### 3.1. `core/normalizer.py`

```python
"""Indic text normalization module for Hindi (hi) and Punjabi (pa)."""

from typing import Final, Optional, Set

SUPPORTED_LANGUAGES: Final[Set[str]] = {"hi", "pa"}

# Character sets and Unicode ranges
# Devanagari: U+0900 - U+097F
# Gurmukhi:   U+0A00 - U+0A7F

def normalize_text(text: str, language: str) -> str:
    """Normalize raw text for Indic TTS synthesis.

    Strips emojis, URLs, and unsupported symbols while preserving required diacritics
    (virama/halant, bindi, tippi, addak, danda) and expanding digits into spoken words.

    Args:
        text: Raw input string in Devanagari, Gurmukhi, or mixed script with digits.
        language: ISO language code ('hi' or 'pa').

    Returns:
        Cleaned, normalized string with spoken word representations for all numerals.

    Raises:
        ValueError: If language is not in {'hi', 'pa'}.
        TypeError: If text is not a string or language is not a string.
    """


def expand_digits(text: str, language: str) -> str:
    """Detect all numeric tokens (ASCII '0-9', Devanagari '०-९', Gurmukhi '੦-੯')

    and expand them into spoken words in the target language script.

    Args:
        text: Input text containing numeric digits.
        language: Target language code ('hi' or 'pa').

    Returns:
        Text with all digits converted to spelled-out spoken words.

    Raises:
        ValueError: If language is not in {'hi', 'pa'}.
        TypeError: If text is not a string.
    """


def number_to_words(number: int, language: str) -> str:
    """Convert an integer to its spelled-out Indic word representation.

    Supports numbers from 0 to 99,99,99,999 (crores) following Indic numbering.

    Args:
        number: Non-negative integer to convert.
        language: Target language code ('hi' for Hindi, 'pa' for Punjabi).

    Returns:
        Word representation in Devanagari or Gurmukhi script.

    Raises:
        ValueError: If number < 0 or language is unsupported.
        TypeError: If number is not an integer.
    """


def strip_unsupported_characters(text: str, language: str) -> str:
    """Strip emojis, control codes, and symbols outside the allowed phonetic alphabet.

    Preserves:
    - Hindi: Devanagari letters, matras, nukta (\u093C), virama (\u094D),
             bindi/anusvara (\u0902), chandrabindu (\u0901), danda (\u0964), double danda (\u0965).
    - Punjabi: Gurmukhi letters, matras, nukta (\u0A3C), virama (\u0A4D),
               bindi (\u0A02), tippi (\u0A70), addak (\u0A71), danda (\u0964), double danda (\u0965).
    - Permitted prosodic punctuation: whitespace, comma (,), question mark (?), exclamation (!).

    Args:
        text: Raw text.
        language: Language code ('hi' or 'pa').

    Returns:
        Cleaned text string containing only valid characters.
    """


class IndicNormalizer:
    """Stateful or object-oriented interface for Indic text normalization."""

    def __init__(self, language: str) -> None:
        """Initialize normalizer for a specific language.

        Args:
            language: Language code ('hi' or 'pa').

        Raises:
            ValueError: If language is not supported.
        """
        ...

    def normalize(self, text: str) -> str:
        """Normalize input text using the configured language."""
        ...
```

---

### 3.2. `core/engine.py`

```python
"""Model-agnostic speech synthesis core for Indic languages."""

from pathlib import Path
from typing import Any, Callable, Final, Optional, Set, Tuple, Union
import numpy as np

SUPPORTED_LANGUAGES: Final[Set[str]] = {"hi", "pa"}
DEFAULT_SAMPLE_RATE: Final[int] = 24000
ENV_MODEL_PATH_KEY: Final[str] = "INDIC_TTS_MODEL_PATH"
FALLBACK_ENV_MODEL_PATH_KEY: Final[str] = "MODEL_PATH"


class Synthesizer:
    """Model-agnostic TTS inference synthesizer for Hindi and Punjabi."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        backend: Optional[Any] = None,
    ) -> None:
        """Initialize the Synthesizer with model path, execution device, and sample rate.

        Args:
            model_path: Path to model weights or checkpoint. If None, resolves from
                        INDIC_TTS_MODEL_PATH or MODEL_PATH environment variables.
            device: Requested device ('cuda', 'cpu', or None for auto-detect).
                    If 'cuda' is requested but unavailable, falls back to 'cpu'.
            sample_rate: Audio sample rate in Hz (default 24000).
            backend: Optional pluggable/mocked inference backend callable or object.

        Raises:
            ValueError: If model_path is None and no environment variable is configured
                        (and backend is not injected).
            ValueError: If sample_rate <= 0.
        """
        ...

    @property
    def device(self) -> str:
        """Return the resolved active device ('cpu' or 'cuda')."""
        ...

    @property
    def sample_rate(self) -> int:
        """Return the synthesis sample rate in Hz."""
        ...

    @property
    def model_path(self) -> Optional[Path]:
        """Return the resolved model path, if configured."""
        ...

    def synthesize(
        self,
        text: str,
        language: str,
        speaker_ref: Optional[Union[str, Path, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Normalize input text and synthesize speech waveform.

        Args:
            text: Input text in Hindi or Punjabi (raw or normalized).
            language: Language code ('hi' or 'pa').
            speaker_ref: Optional audio file path or numpy waveform for voice cloning.

        Returns:
            Tuple[np.ndarray, int]: (waveform as 1D float32 numpy array in [-1.0, 1.0], sample_rate).

        Raises:
            ValueError: If language is not in {'hi', 'pa'}:
                        Exact message: f"Unsupported language: {language}"
            ValueError: If text is empty or contains only whitespace/unsupported symbols.
            TypeError: If text or language is not a string.
        """
        ...
```

---

### 3.3. `core/audio_processor.py`

```python
"""Audio mastering, loudness normalization, filtering, and WAV export module."""

from pathlib import Path
from typing import Final, Optional, Set, Tuple, Union
import numpy as np

TARGET_LUFS: Final[float] = -14.0
TRUE_PEAK_CEILING_DB: Final[float] = -1.0
HIGH_PASS_CUTOFF_HZ: Final[float] = 80.0
ALLOWED_EXPORT_SAMPLE_RATES: Final[Set[int]] = {24000, 44100}


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
    ...


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
    """
    ...


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
    """
    ...


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
        ValueError: If orig_sr <= 0 or target_sr <= 0.
    """
    ...


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
        ValueError: If target_sample_rate not in {24000, 44100}.
    """
    ...


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
    ...


class AudioProcessor:
    """Encapsulated audio mastering and export processor."""

    def __init__(
        self,
        target_sample_rate: int = 24000,
        target_lufs: float = TARGET_LUFS,
        peak_ceiling_db: float = TRUE_PEAK_CEILING_DB,
        highpass_cutoff: float = HIGH_PASS_CUTOFF_HZ,
    ) -> None:
        """Initialize AudioProcessor with mastering configuration."""
        ...

    def process_and_export(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: Union[str, Path],
    ) -> Path:
        """Master audio and export 16-bit PCM WAV."""
        ...
```

---

### 3.4. `requirements.txt`

```
numpy>=1.26.0
soundfile>=0.12.1
```

*(Self-contained: DSP filtering, ITU-R BS.1770 K-weighting, and 16-bit PCM WAV serialization implemented via standard numpy and soundfile to ensure 100% offline verification in `.venv`).*

---

## 4. Explicit Error Contracts

| Component | Function / Method | Condition | Exception Raised | Error Message / Contract |
|---|---|---|---|---|
| `normalizer.py` | `normalize_text(text, language)` | `language not in ['hi', 'pa']` | `ValueError` | `f"Unsupported language: {language}"` |
| `normalizer.py` | `normalize_text(text, language)` | `not isinstance(text, str)` | `TypeError` | `f"Expected text to be str, got {type(text).__name__}"` |
| `normalizer.py` | `normalize_text(text, language)` | `not isinstance(language, str)` | `TypeError` | `f"Expected language to be str, got {type(language).__name__}"` |
| `normalizer.py` | `number_to_words(number, language)` | `number < 0` | `ValueError` | `f"Negative numbers not supported: {number}"` |
| `normalizer.py` | `number_to_words(number, language)` | `language not in ['hi', 'pa']` | `ValueError` | `f"Unsupported language: {language}"` |
| `engine.py` | `Synthesizer.__init__` | `model_path is None` and no env var set and `backend is None` | `ValueError` | `"Model path must be provided via constructor or INDIC_TTS_MODEL_PATH / MODEL_PATH environment variable."` |
| `engine.py` | `Synthesizer.__init__` | `device="cuda"` requested but CUDA unavailable | **No exception** | Emits `logging.warning`, falls back to `self._device = "cpu"` |
| `engine.py` | `Synthesizer.synthesize` | `language not in ['hi', 'pa']` | `ValueError` | `f"Unsupported language: {language}"` |
| `engine.py` | `Synthesizer.synthesize` | `text.strip() == ""` | `ValueError` | `"Text cannot be empty or whitespace-only."` |
| `engine.py` | `Synthesizer.synthesize` | `not isinstance(text, str)` | `TypeError` | `f"Expected text to be str, got {type(text).__name__}"` |
| `audio_processor.py` | `apply_highpass_filter` | `len(audio) == 0` | `ValueError` | `"Audio array cannot be empty."` |
| `audio_processor.py` | `normalize_loudness` | Silent input (all zeros or RMS < 1e-6) | **No exception** | Logs warning: `"Input audio is silent; returning best-effort unmodified waveform without gain boost."` Returns zeros array. |
| `audio_processor.py` | `normalize_loudness` | Clipped input (`np.max(np.abs(audio)) >= 1.0`) | **No exception** | Logs warning: `"Clipped audio detected; peak attenuating before loudness normalization."` Soft-limits/attenuates gracefully. |
| `audio_processor.py` | `export_wav` | `sample_rate not in {24000, 44100}` | `ValueError` | `f"Unsupported sample rate: {sample_rate}. Expected 24000 or 44100 Hz."` |
| `audio_processor.py` | `export_wav` | `len(audio) == 0` | `ValueError` | `"Audio array cannot be empty."` |

---

## 5. Test Fixtures & Mock Strategy

### 5.1. `tests/test_normalizer.py`

#### Test Fixtures
- `hindi_clean_sample`: Pure Devanagari text with virama and bindi: `"नमस्ते भारत! आप कैसे हैं?"`
- `punjabi_clean_sample`: Pure Gurmukhi text with tippi, addak, bindi: `"ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"`
- `diacritics_sample_hi`: Devanagari string explicitly testing virama (`्`, `\u094D`), bindi (`ं`, `\u0902`), chandrabindu (`ँ`, `\u0901`), and danda (`।`, `\u0964`): `"संसार में कर्म ही प्रधान है।"`.
- `diacritics_sample_pa`: Gurmukhi string testing bindi (`ਂ`, `\u0A02`), tippi (`ੰ`, `\u0A70`), addak (`ੱ`, `\u0A71`), and danda (`।`, `\u0964`): `"ਪੰਜਾਬੀ ਵਿੱਚ ਪਿੰਡ ਅਤੇ ਕੁੱਤਾ ਲਿਖੋ।"`.
- `dirty_emoji_sample`: Indic text containing emojis (`😀`, `🔥`, `🎉`), Twitter tags (`#speech`, `@user`), URLs, and unapproved symbols (`%`, `^`, `&`, `*`).
- `number_conversion_table_hi`: Test cases:
  - `0` → `"शून्य"`
  - `100` → `"एक सौ"`
  - `105` → `"एक सौ पाँच"`
  - `1000` → `"एक हज़ार"`
  - `250000` → `"दो लाख पचास हज़ार"`
  - Sentence: `"मेरे पास 100 रुपये हैं।"` → `"मेरे पास एक सौ रुपये हैं।"`
- `number_conversion_table_pa`: Test cases:
  - `0` → `"ਸਿਫ਼ਰ"`
  - `100` → `"ਇੱਕ ਸੌ"`
  - `105` → `"ਇੱਕ ਸੌ ਪੰਜ"`
  - `1000` → `"ਇੱਕ ਹਜ਼ਾਰ"`
  - `250000` → `"ਦੋ ਲੱਖ ਪੰਜਾਹ ਹਜ਼ਾਰ"`
  - Sentence: `"ਮੇਰੇ ਕੋਲ 100 ਰੁਪਏ ਹਨ।"` → `"ਮੇਰੇ ਕੋਲ ਇੱਕ ਸੌ ਰੁਪਏ ਹਨ।"`

#### Mock Strategy
- Zero mocks needed: Normalizer is pure algorithmic text transformation.
- Deterministic property tests verifying character whitelists and regex filters.

---

### 5.2. `tests/test_audio_processor.py`

#### Test Fixtures
- `synthetic_sine_100hz`: 1-second 100 Hz sine wave at 24000 Hz (`np.sin(2 * np.pi * 100 * t)`).
- `synthetic_rumble_30hz`: 1-second 30 Hz sub-bass sine wave at 24000 Hz.
- `synthetic_silent_audio`: `np.zeros(24000, dtype=np.float32)`.
- `synthetic_clipped_audio`: High-gain sine wave clamped to `[-1.0, 1.0]` with flattened peaks simulating hard clipping.
- `synthetic_quiet_speech_proxy`: Audio with low integrated loudness (~ -30 LUFS).
- `tmp_export_dir`: `pytest` `tmp_path` fixture for generating temporary `.wav` files.

#### Mock Strategy
- Zero external audio mocks: Uses mathematically synthesized numpy arrays.
- Assertions verify:
  1. `apply_highpass_filter` reduces 30 Hz signal power by at least 15 dB while preserving 1000 Hz signal power within 0.5 dB.
  2. `normalize_loudness` brings normal speech proxy to `-14.0 ± 0.5 LUFS`.
  3. True peak ceiling: `np.max(np.abs(normalized_audio))` never exceeds `10 ** (-1.0 / 20) ≈ 0.89125`.
  4. Graceful degradation:
     - Passing `synthetic_silent_audio` logs warning, does not throw, and returns clean array.
     - Passing `synthetic_clipped_audio` logs warning, does not throw, and returns valid audio within `[-0.89125, 0.89125]`.
  5. `export_wav`:
     - Creates valid 16-bit PCM WAV at 24000 Hz; verified by reading back with `soundfile.read()`.
     - Creates valid 16-bit PCM WAV at 44100 Hz.
     - Exporting at 16000 Hz or 48000 Hz raises `ValueError`.

---

### 5.3. `tests/test_engine_mock.py`

#### Test Fixtures & Mock Strategy
- **Mock Inference Backend:** A callable / class simulating model synthesis:
  ```python
  class MockTTSBackend:
      def __init__(self, sample_rate: int = 24000):
          self.sample_rate = sample_rate
          self.calls = []

      def __call__(self, normalized_text: str, language: str, speaker_ref: Any):
          self.calls.append((normalized_text, language, speaker_ref))
          duration_sec = 0.5
          t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
          waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
          return waveform, self.sample_rate
  ```
- **CPU Fallback Fixture:**
  - Mock `torch.cuda.is_available` returning `False`.
  - Instantiate `Synthesizer(device="cuda", backend=mock_backend)`.
  - Assert `synthesizer.device == "cpu"`.
- **Environment Variable Injection Fixture:**
  - Use `monkeypatch.setenv("INDIC_TTS_MODEL_PATH", "/tmp/mock_model.pt")`.
  - Instantiate `Synthesizer()`.
  - Assert `synthesizer.model_path == Path("/tmp/mock_model.pt")`.
- **Constructor Injection Precedence:**
  - With `INDIC_TTS_MODEL_PATH` set to `/tmp/env_model.pt`, pass `model_path="/custom/path.pt"`.
  - Assert `synthesizer.model_path == Path("/custom/path.pt")`.
- **Language Contract Assertions:**
  - `.synthesize("Hello", "en")` raises `ValueError("Unsupported language: en")`.
  - `.synthesize("Bonjour", "fr")` raises `ValueError("Unsupported language: fr")`.
  - `.synthesize("नमस्ते", "hi")` succeeds and records normalized text in backend mock.
  - `.synthesize("ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ", "pa")` succeeds.
- **Normalizer Pre-processing Verification:**
  - `.synthesize("100 रुपये", "hi")` passes `"एक सौ रुपये"` to the backend.

---

## 6. Open Questions & Ambiguities

Before proceeding to code implementation, the following architectural details and ticket ambiguities are identified for stakeholder awareness:

1. **Number Normalization Ceiling & Ordinal/Decimal Support:**
   - *Question:* Does the acceptance criteria require support beyond cardinal numbers (e.g. decimals like `10.5`, ordinals like `1st` / `पहला`, currency symbols `₹`), or is the scope strictly integer cardinal conversion (0 to crores) as demonstrated in the example `100 → एक सौ` / `ਇੱਕ ਸੌ`?
   - *Working Assumption in Plan:* We implement integer cardinal expansion up to crores (`100` → `एक सौ`, `1000` → `एक हज़ार`, etc.) and strip unmatched symbols (e.g. `₹` stripped or mapped to `रुपये` / `ਰੁਪਏ`).
2. **Prosodic Punctuation Whitelist:**
   - *Question:* In addition to `virama/bindi/tippi/addak/danda (।)` explicitly requested, TTS prosody depends on sentence break marks (`,`, `?`, `!`, `॥`).
   - *Working Assumption in Plan:* We preserve standard prosodic punctuation (danda `।`, double danda `॥`, `,`, `?`, `!`) and strip arbitrary emojis, math symbols, and URL noise.
3. **Pluggable Backend Interface vs Direct Model Loader:**
   - *Question:* Will Ticket 002 inject a specific deep learning model checkpoint (e.g. F5-TTS, VITS, or FastSpeech2), or should `Synthesizer` support loading weights from Torch checkpoint directly when torch is installed?
   - *Working Assumption in Plan:* `Synthesizer` will feature a clean `backend` parameter allowing arbitrary model backends or mock callables, alongside lazy loading of PyTorch weights if a checkpoint path is provided.
4. **Resampling Policy for Arbitrary Model Sample Rates:**
   - *Question:* If an underlying TTS model outputs audio at 22050 Hz or 16000 Hz, should `Synthesizer` output the native rate or automatically pass through `audio_processor` to achieve the target 24000 Hz or 44100 Hz?
   - *Working Assumption in Plan:* `Synthesizer.synthesize` outputs `(waveform, sample_rate)` reflecting its synthesis rate, and `audio_processor.master_audio` / `export_wav` performs resampling to the target 24000 Hz or 44100 Hz.

---

## 7. Verification Execution Roadmap

The implementation will be verified with 100% pass rate using:

```bash
.venv/bin/pytest tests/test_normalizer.py tests/test_audio_processor.py tests/test_engine_mock.py -v
```

All tests will run in the local isolated environment without downloading external weights, invoking network sockets, or mutating existing baseline tests (`test_config_and_health.py`).
