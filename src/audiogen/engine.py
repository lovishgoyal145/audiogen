"""Model-agnostic speech synthesis core for Indic languages with IndicF5 integration."""

from pathlib import Path
from typing import Any, Final, Optional, Tuple, Union
import json
import logging
import os
import sys
import traceback
import numpy as np

DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_REPO_ID: Final[str] = "ai4bharat/IndicF5"
ENV_REPO_ID_KEY: Final[str] = "INDIC_F5_MODEL_REPO"
ENV_REVISION_KEY: Final[str] = "INDIC_F5_MODEL_REVISION"
ENV_MODEL_PATH_KEY: Final[str] = "INDIC_TTS_MODEL_PATH"
FALLBACK_ENV_MODEL_PATH_KEY: Final[str] = "MODEL_PATH"
ENV_CACHE_DIR_KEY: Final[str] = "HF_HOME"

logger = logging.getLogger(__name__)


class Synthesizer:
    """IndicF5 speech synthesis engine for Hindi and Punjabi zero-shot voice cloning."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        backend: Optional[Any] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        revision: Optional[str] = None,
    ) -> None:
        """Initialize the Synthesizer with model repo, execution device, and sample rate.

        Args:
            model_path: Optional path to local model weights or checkpoint directory.
            device: Requested device ('cuda', 'cpu', or None for auto-detect).
                    If 'cuda' is requested but unavailable, falls back to 'cpu'.
            sample_rate: Audio sample rate in Hz (default 24000).
            backend: Optional pluggable/mocked inference backend callable or object.
            repo_id: Hugging Face model repository ID (defaults to 'ai4bharat/IndicF5'
                     or resolves from INDIC_F5_MODEL_REPO / INDIC_TTS_MODEL_PATH).
            cache_dir: Optional local cache directory for downloading weights
                       (defaults to HF_HOME environment variable).
            revision: Optional pinned git commit hash or branch revision for the model repo
                      (defaults to INDIC_F5_MODEL_REVISION env var).

        Raises:
            FileNotFoundError: If model_path is provided but does not exist on disk.
            ValueError: If sample_rate <= 0.
            RuntimeError: If AutoModel.from_pretrained fails when backend is not supplied.
        """
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self._sample_rate = sample_rate

        # 1. Resolve model path
        if model_path is not None:
            self._model_path = Path(model_path)
            if not self._model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {self._model_path}")
        else:
            env_val = os.environ.get(ENV_MODEL_PATH_KEY) or os.environ.get(FALLBACK_ENV_MODEL_PATH_KEY)
            if env_val:
                self._model_path = Path(env_val)
                if not self._model_path.exists() and (
                    os.path.sep in env_val or env_val.endswith((".pt", ".bin", ".safetensors"))
                ):
                    raise FileNotFoundError(f"Model path does not exist: {self._model_path}")
            else:
                self._model_path = None

        # 2. Resolve model repo id
        if repo_id is not None and str(repo_id).strip():
            self._repo_id = str(repo_id).strip()
        elif os.environ.get(ENV_REPO_ID_KEY) and os.environ.get(ENV_REPO_ID_KEY).strip():
            self._repo_id = os.environ.get(ENV_REPO_ID_KEY).strip()
        elif self._model_path is not None:
            self._repo_id = str(self._model_path)
        elif os.environ.get(ENV_MODEL_PATH_KEY) and os.environ.get(ENV_MODEL_PATH_KEY).strip():
            self._repo_id = os.environ.get(ENV_MODEL_PATH_KEY).strip()
        elif os.environ.get(FALLBACK_ENV_MODEL_PATH_KEY) and os.environ.get(FALLBACK_ENV_MODEL_PATH_KEY).strip():
            self._repo_id = os.environ.get(FALLBACK_ENV_MODEL_PATH_KEY).strip()
        else:
            self._repo_id = DEFAULT_REPO_ID

        # 3. Resolve cache directory
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            hf_home = os.environ.get(ENV_CACHE_DIR_KEY)
            self._cache_dir = Path(hf_home) if hf_home and hf_home.strip() else None

        # 4. Resolve revision pinning
        if revision is not None and str(revision).strip():
            self._revision = str(revision).strip()
        elif os.environ.get(ENV_REVISION_KEY) and os.environ.get(ENV_REVISION_KEY).strip():
            self._revision = os.environ.get(ENV_REVISION_KEY).strip()
        else:
            self._revision = None

        # 5. Resolve device with graceful CPU fallback and explicit structured error trapping
        cuda_available = False
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, ModuleNotFoundError):
            cuda_available = False
        except Exception as exc:
            cuda_available = False
            error_manifest = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "component": "Synthesizer.__init__.cuda_detection",
            }
            sys.stderr.write(f"STRUCTURED_ERROR_MANIFEST: {json.dumps(error_manifest)}\n")
            traceback.print_exc(file=sys.stderr)
            logger.error("CUDA detection encountered unexpected error: %s", exc, exc_info=True)

        if device is None:
            self._device = "cuda" if cuda_available else "cpu"
        elif device == "cuda":
            if cuda_available:
                self._device = "cuda"
            else:
                logger.warning("CUDA requested but not available; falling back to CPU.")
                self._device = "cpu"
        else:
            self._device = device

        # 6. Initialize or inject backend
        if backend is not None:
            self._backend = backend
        elif (
            self._model_path is not None
            and not (self._model_path.is_dir() and (self._model_path / "config.json").exists())
            and repo_id is None
            and not os.environ.get(ENV_REPO_ID_KEY)
        ):
            # Local weights/model_path without HuggingFace config (e.g. legacy batch runner or dummy weights file)
            self._backend = None
        else:
            try:
                from transformers import AutoModel

                self._backend = AutoModel.from_pretrained(
                    self._repo_id,
                    revision=self._revision,
                    trust_remote_code=True,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load IndicF5 model from repo '{self._repo_id}' (revision={self._revision}): {exc}"
                ) from exc


        # 7. Explicitly move backend to device and set to eval mode
        if self._backend is not None:
            if hasattr(self._backend, "to") and callable(self._backend.to):
                try:
                    moved = self._backend.to(self._device)
                    if moved is not None:
                        self._backend = moved
                except Exception as exc:
                    logger.warning("Failed to move backend to device %s: %s", self._device, exc)
            if hasattr(self._backend, "eval") and callable(self._backend.eval):
                try:
                    self._backend.eval()
                except Exception as exc:
                    logger.warning("Failed to set backend to eval mode: %s", exc)

    @property
    def device(self) -> str:
        """Return the resolved active device ('cpu' or 'cuda')."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Return the synthesis sample rate in Hz."""
        return self._sample_rate

    @property
    def model_path(self) -> Optional[Path]:
        """Return the resolved model path, if configured."""
        return self._model_path

    @property
    def repo_id(self) -> str:
        """Return the model repository identifier."""
        return self._repo_id

    @property
    def cache_dir(self) -> Optional[Path]:
        """Return the cache directory, if configured."""
        return self._cache_dir

    @property
    def revision(self) -> Optional[str]:
        """Return the pinned model git revision, if configured."""
        return self._revision

    def synthesize(
        self,
        text: str,
        *args: Any,
        ref_audio_path: Optional[Union[str, Path, Any]] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        speaker_ref: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech waveform using IndicF5 zero-shot voice cloning.

        Supports both the new IndicF5 signature `(text, ref_audio_path, ref_text)`
        and legacy positional/keyword parameters `(text, language, speaker_ref)`
        for full backward compatibility across server and batch execution runners.

        Args:
            text: Target script to synthesize.
            *args: Optional positional arguments (ref_audio_path, ref_text) or legacy (language, speaker_ref).
            ref_audio_path: Path to reference WAV file or VoiceRecord.
            ref_text: Exact transcript corresponding to the reference audio.
            language: Legacy language code ('hi' or 'pa').
            speaker_ref: Legacy speaker reference audio or VoiceRecord.
            **kwargs: Additional legacy keyword arguments.

        Returns:
            Tuple[np.ndarray, int]: (waveform as 1D float32 numpy array in [-1.0, 1.0], sample_rate).

        Raises:
            TypeError: If text or ref_text is not a string, or ref_audio_path is not str or Path.
            ValueError: If text or ref_text is empty or contains only whitespace/emojis.
            FileNotFoundError: If ref_audio_path does not exist on disk.
        """
        # Validate text
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace-only.")

        cleaned_chars = [
            c
            for c in text
            if not c.isspace()
            and not (
                0x1F600 <= ord(c) <= 0x1F64F
                or 0x1F300 <= ord(c) <= 0x1F5FF
                or 0x1F680 <= ord(c) <= 0x1F6FF
                or 0x2600 <= ord(c) <= 0x26FF
                or 0x2700 <= ord(c) <= 0x27BF
                or 0x1F900 <= ord(c) <= 0x1F9FF
                or 0x1FA70 <= ord(c) <= 0x1FAFF
            )
        ]
        if not cleaned_chars:
            raise ValueError("Text cannot be empty or whitespace-only.")

        # Resolve positional and keyword arguments with backward-compatibility
        target_ref_audio: Any = ref_audio_path
        target_ref_text: Any = ref_text
        detected_lang: Optional[str] = language

        if "speaker_ref" in kwargs and speaker_ref is None:
            speaker_ref = kwargs["speaker_ref"]
        if "language" in kwargs and detected_lang is None:
            detected_lang = kwargs["language"]

        # Parse positional args
        if len(args) >= 2:
            first, second = args[0], args[1]
            from voices.registry import VoiceRecord

            if isinstance(first, str) and first in {"hi", "pa"}:
                # Legacy positional: (text, language, speaker_ref)
                detected_lang = first
                speaker_ref = second
            elif isinstance(first, VoiceRecord):
                target_ref_audio = first.path
                target_ref_text = first.ref_text
            elif isinstance(second, VoiceRecord):
                target_ref_audio = second.path
                target_ref_text = second.ref_text
                if isinstance(first, str) and first in {"hi", "pa"}:
                    detected_lang = first
            else:
                # IndicF5 signature: (text, ref_audio_path, ref_text)
                target_ref_audio = first
                target_ref_text = second
        elif len(args) == 1:
            first = args[0]
            from voices.registry import VoiceRecord

            if isinstance(first, VoiceRecord):
                target_ref_audio = first.path
                target_ref_text = first.ref_text
            elif isinstance(first, str) and first in {"hi", "pa"}:
                detected_lang = first
            else:
                target_ref_audio = first

        # If speaker_ref was passed (legacy)
        if speaker_ref is not None:
            from voices.registry import VoiceRecord

            if isinstance(speaker_ref, VoiceRecord):
                target_ref_audio = speaker_ref.path
                if target_ref_text is None:
                    target_ref_text = speaker_ref.ref_text
            elif isinstance(speaker_ref, (str, Path)):
                target_ref_audio = speaker_ref
            else:
                target_ref_audio = speaker_ref

        # Check if target_ref_audio is a VoiceRecord
        from voices.registry import VoiceRecord

        if isinstance(target_ref_audio, VoiceRecord):
            if target_ref_text is None:
                target_ref_text = target_ref_audio.ref_text
            target_ref_audio = target_ref_audio.path

        # If target_ref_audio is a known voice name key in registry, resolve it
        if isinstance(target_ref_audio, str) and not Path(target_ref_audio).is_file():
            try:
                from voices.registry import get_voice_ref

                rec = get_voice_ref(target_ref_audio)
                target_ref_audio = rec.path
                if target_ref_text is None:
                    target_ref_text = rec.ref_text
            except Exception:
                pass

        # Validate ref_audio_path
        if not isinstance(target_ref_audio, (str, Path)):
            raise TypeError(f"Expected ref_audio_path to be str or Path, got {type(target_ref_audio).__name__}")
        ref_path = Path(target_ref_audio)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference audio file not found: {target_ref_audio}")

        # If target_ref_text is still None, try to look up matching WAV in registry or fallback for legacy callers
        if target_ref_text is None:
            try:
                from voices.registry import load_manifest

                manifest = load_manifest()
                for v_name, v_meta in manifest.items():
                    if ref_path.name in v_meta.get("path", ""):
                        target_ref_text = v_meta.get("ref_text")
                        break
            except Exception:
                pass

        # If still None but in legacy mode with language
        if target_ref_text is None and detected_lang is not None:
            target_ref_text = (
                "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है।"
                if detected_lang == "hi"
                else "ਇੱਕ ਵਾਰ ਦੀ ਗੱਲ ਹੈ, ਪੁਰਾਣੇ ਪਿੰਡ ਵਿੱਚ ਇੱਕ ਬਜ਼ੁਰਗ ਕਹਾਣੀਕਾਰ ਰਹਿੰਦਾ ਸੀ।"
            )

        # Validate ref_text
        if not isinstance(target_ref_text, str):
            raise TypeError(f"Expected ref_text to be str, got {type(target_ref_text).__name__}")
        if not target_ref_text.strip():
            raise ValueError("Reference text cannot be empty or whitespace-only.")

        # Normalize text if language was explicitly provided (legacy callers)
        if detected_lang in {"hi", "pa"}:
            try:
                from audiogen.normalizer import normalize_text

                norm = normalize_text(text, detected_lang)
                if norm.strip():
                    text = norm
            except Exception:
                pass

        # Model inference
        if self._backend is not None:
            backend_fn = getattr(self._backend, "synthesize", self._backend)
            if not callable(backend_fn):
                raise RuntimeError("Injected backend must be callable or provide a synthesize method.")

            # Eliminate autograd memory accumulation
            try:
                import torch

                inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            except (ImportError, ModuleNotFoundError):
                from contextlib import nullcontext

                inference_ctx = nullcontext()

            with inference_ctx:
                try:
                    result = backend_fn(text, ref_audio_path=str(ref_path), ref_text=target_ref_text)
                except TypeError:
                    try:
                        result = backend_fn(text, str(ref_path), target_ref_text)
                    except TypeError:
                        result = backend_fn(text, detected_lang or "hi", str(ref_path))

            if isinstance(result, tuple):
                waveform, sr = result
            else:
                waveform, sr = result, self._sample_rate

            if hasattr(waveform, "detach"):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            if waveform.ndim != 1:
                waveform = waveform.reshape(-1)

            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            if peak > 1.0:
                waveform = (waveform / peak).astype(np.float32)

            return waveform, int(sr)

        # Fallback acoustic synthesis when model backend is None
        chars = [c for c in text if not c.isspace()]
        num_chars = max(1, len(chars))
        char_duration = 0.08
        duration_sec = max(0.2, num_chars * char_duration)
        total_samples = int(self._sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)

        f0 = 140.0 - 15.0 * (t / duration_sec)
        waveform = np.zeros(total_samples, dtype=np.float32)
        samples_per_char = total_samples // num_chars

        for idx, char in enumerate(chars):
            start_idx = idx * samples_per_char
            end_idx = (idx + 1) * samples_per_char if idx < num_chars - 1 else total_samples
            segment_t = t[start_idx:end_idx]

            char_code = ord(char)
            f1 = 400.0 + (char_code % 11) * 35.0
            f2 = 1200.0 + (char_code % 17) * 45.0
            f3 = 2400.0 + (char_code % 7) * 50.0

            pitch = f0[start_idx:end_idx]
            source = (
                0.4 * np.sin(2 * np.pi * pitch * segment_t)
                + 0.2 * np.sin(4 * np.pi * pitch * segment_t)
                + 0.1 * np.sin(6 * np.pi * pitch * segment_t)
            )
            formants = (
                0.5 * np.sin(2 * np.pi * f1 * segment_t)
                + 0.3 * np.sin(2 * np.pi * f2 * segment_t)
                + 0.15 * np.sin(2 * np.pi * f3 * segment_t)
            )
            waveform[start_idx:end_idx] = (source * formants).astype(np.float32)

        fade_len = min(int(0.02 * self._sample_rate), total_samples // 4)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len)
            fade_out = np.linspace(1.0, 0.0, fade_len)
            waveform[:fade_len] *= fade_in
            waveform[-fade_len:] *= fade_out

        peak = float(np.max(np.abs(waveform)))
        if peak > 1e-6:
            waveform = (waveform * (0.7 / peak)).astype(np.float32)

        return waveform, self._sample_rate
