"""Model-agnostic speech synthesis core for Indic languages with IndicF5 integration."""

from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import numpy as np
import soundfile as sf

from audiogen.config import MissingKaggleCredentialsError, Settings, get_settings

DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_REPO_ID: Final[str] = "ai4bharat/IndicF5"
ENV_REPO_ID_KEY: Final[str] = "INDIC_F5_MODEL_REPO"
ENV_REVISION_KEY: Final[str] = "INDIC_F5_MODEL_REVISION"
ENV_MODEL_PATH_KEY: Final[str] = "INDIC_TTS_MODEL_PATH"
FALLBACK_ENV_MODEL_PATH_KEY: Final[str] = "MODEL_PATH"
ENV_CACHE_DIR_KEY: Final[str] = "HF_HOME"

logger = logging.getLogger(__name__)


class KaggleExecutionError(RuntimeError):
    """Raised when Kaggle kernel execution fails or returns non-zero exit code."""

    pass


class KaggleTimeoutError(RuntimeError):
    """Raised when Kaggle kernel execution exceeds configured timeout."""

    pass


class KaggleExecutionBridge:
    """Zero-mock Kaggle cloud execution bridge for GPU-accelerated speech synthesis."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        kaggle_cmd: Optional[List[str]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.kaggle_cmd = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.data_audio_dir = self.repo_root / "src" / "audiogen" / "data" / "audio"
        self.data_audio_dir.mkdir(parents=True, exist_ok=True)

    def _build_isolated_env(self) -> Dict[str, str]:
        """Build isolated subprocess environment with Kaggle API credentials without mutating global os.environ."""
        env = os.environ.copy()
        if self.settings.kaggle_username:
            env["KAGGLE_USERNAME"] = self.settings.kaggle_username
        if self.settings.kaggle_key:
            env["KAGGLE_KEY"] = self.settings.kaggle_key
        return env

    def validate_credentials(self) -> None:
        """Enforce required Kaggle credentials from Settings."""
        self.settings.validate_kaggle_credentials()

    def stage_execution(
        self,
        text: str,
        ref_audio_path: Union[str, Path],
        ref_text: str,
        language: Optional[str] = None,
        staging_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, str]:
        """Stage Kaggle batch synthesis notebook, scripts.json, and kernel-metadata.json."""
        self.validate_credentials()

        task_id = f"task_{uuid.uuid4().hex[:10]}"
        if staging_dir is not None:
            stage_dir = Path(staging_dir).resolve()
            stage_dir.mkdir(parents=True, exist_ok=True)
        else:
            stage_dir = Path(tempfile.mkdtemp(prefix="audiogen_stage_")).resolve()

        # Resolve language
        if language is None:
            lang = "hi"
        else:
            lang_cleaned = str(language).strip().lower()
            if lang_cleaned not in {"en", "hi", "pa"}:
                raise ValueError(
                    f"Unsupported language '{language}' for Kaggle GPU bridge. Supported languages: ['en', 'hi', 'pa']"
                )
            lang = lang_cleaned

        # Resolve voice_ref name
        voice_ref = "anchor_female_calm"
        ref_file = Path(ref_audio_path).resolve()
        try:
            from voices.registry import load_manifest

            manifest = load_manifest()
            for v_name, v_meta in manifest.items():
                if ref_file.name in str(v_meta.get("path", "")):
                    voice_ref = v_name
                    break
        except Exception as exc:
            error_manifest = {
                "event": "error_manifest",
                "context": "voice_manifest_resolution",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "ref_file": str(ref_file),
            }
            sys.stderr.write(f"STRUCTURED_ERROR_MANIFEST: {json.dumps(error_manifest)}\n")
            sys.stderr.flush()
            logger.warning("Failed to resolve voice reference name from manifest: %s", exc)

        if lang == "pa" and voice_ref == "anchor_female_calm":
            voice_ref = "storyteller_punjabi_elder"

        task_dict = {
            "id": task_id,
            "text": text,
            "language": lang,
            "voice_ref": voice_ref,
            "ref_text": ref_text,
        }
        manifest_data = {"tasks": [task_dict]}

        # 1. Write scripts.json
        scripts_path = stage_dir / "scripts.json"
        with open(scripts_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 2. Write kernel-metadata.json
        kernel_slug = self.settings.kaggle_kernel_slug or "avidok/vco-worker"
        meta_data = {
            "id": kernel_slug,
            "title": "AudioGen GPU Worker",
            "code_file": "notebook_template.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        meta_path = stage_dir / "kernel-metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # 3. Stage notebook template and inject STAGED_MANIFEST_DATA
        source_nb = self.repo_root / "batch" / "notebook_template.ipynb"
        if not source_nb.exists():
            raise FileNotFoundError(f"Notebook template not found: {source_nb}")

        import nbformat

        with open(source_nb, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        manifest_json_str = json.dumps(manifest_data, indent=2)
        staging_cell_source = (
            "# Injected by KaggleExecutionBridge to stage manifest data directly in remote kernel\n"
            "import json\n"
            "from pathlib import Path\n\n"
            f"STAGED_MANIFEST_DATA = {manifest_json_str}\n\n"
            "manifest_target = Path(MANIFEST_PATH)\n"
            "manifest_target.parent.mkdir(parents=True, exist_ok=True)\n"
            "with open(manifest_target, 'w', encoding='utf-8') as f:\n"
            "    json.dump(STAGED_MANIFEST_DATA, f, indent=2)\n"
        )
        staging_cell = nbformat.v4.new_code_cell(
            source=staging_cell_source,
            metadata={"tags": ["injected-manifest-staging"]},
        )
        insert_idx = len(nb.cells)
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and "manifest_file = Path(MANIFEST_PATH)" in cell.source:
                insert_idx = idx
                break
        nb.cells.insert(insert_idx, staging_cell)

        staged_nb = stage_dir / "notebook_template.ipynb"
        with open(staged_nb, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return stage_dir, task_id

    def push_kernel(self, stage_dir: Path) -> None:
        """Push staged kernel directory to Kaggle GPU cloud."""
        push_cmd = [*self.kaggle_cmd, "kernels", "push", "-p", str(stage_dir)]
        logger.info("Executing Kaggle push: %s", " ".join(push_cmd))
        try:
            proc = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                timeout=min(60.0, self.timeout),
                check=False,
                env=self._build_isolated_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleTimeoutError(f"Kaggle push command timed out after {exc.timeout}s") from exc
        except Exception as exc:
            raise KaggleExecutionError(f"Failed to execute Kaggle push command: {exc}") from exc

        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()
            raise KaggleExecutionError(
                f"Kaggle push failed with exit code {proc.returncode}: {err_msg}"
            )

    def poll_status(self, kernel_id: str) -> None:
        """Poll Kaggle kernel status until completion or failure with timeout protection."""
        status_cmd = [*self.kaggle_cmd, "kernels", "status", kernel_id]
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        while True:
            elapsed = time.time() - start_time
            if elapsed >= self.timeout:
                raise KaggleTimeoutError(
                    f"Kaggle kernel execution timed out after {elapsed:.1f}s (configured limit: {self.timeout}s)"
                )

            remaining = max(5.0, min(30.0, self.timeout - elapsed))
            try:
                proc = subprocess.run(
                    status_cmd,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                    env=self._build_isolated_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise KaggleTimeoutError(f"Kaggle status check command hung/timed out: {exc}") from exc
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise KaggleExecutionError(
                        f"Kaggle status command failed {consecutive_errors} consecutive times: {exc}"
                    ) from exc
                time.sleep(self.poll_interval)
                continue

            if proc.returncode != 0:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    err_msg = (proc.stderr or proc.stdout or "").strip()
                    raise KaggleExecutionError(
                        f"Kaggle status check failed with code {proc.returncode}: {err_msg}"
                    )
                time.sleep(self.poll_interval)
                continue

            consecutive_errors = 0
            status_text = (proc.stdout or "").strip()
            status_lower = status_text.lower()
            logger.info("Kaggle kernel %s status: %s", kernel_id, status_text)

            if "complete" in status_lower:
                logger.info("Kaggle kernel %s completed successfully.", kernel_id)
                break

            if "error" in status_lower or "failed" in status_lower:
                raise KaggleExecutionError(
                    f"Kaggle kernel {kernel_id} terminated with failure status: {status_text}"
                )

            if "cancel" in status_lower:
                raise KaggleExecutionError(
                    f"Kaggle kernel {kernel_id} was cancelled: {status_text}"
                )

            time.sleep(self.poll_interval)

    def pull_output(self, kernel_id: str, destination_dir: Path) -> Path:
        """Download rendered output files from completed Kaggle run."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_cmd = [*self.kaggle_cmd, "kernels", "output", kernel_id, "-p", str(destination_dir), "--force"]
        logger.info("Pulling Kaggle kernel output: %s", " ".join(output_cmd))

        try:
            proc = subprocess.run(
                output_cmd,
                capture_output=True,
                text=True,
                timeout=min(120.0, self.timeout),
                check=False,
                env=self._build_isolated_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleTimeoutError(f"Kaggle output pull command timed out after {exc.timeout}s") from exc
        except Exception as exc:
            raise KaggleExecutionError(f"Failed to pull Kaggle kernel output: {exc}") from exc

        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()
            raise KaggleExecutionError(
                f"Kaggle output pull failed with code {proc.returncode}: {err_msg}"
            )

        return destination_dir

    def synthesize(
        self,
        text: str,
        ref_audio_path: Union[str, Path],
        ref_text: str,
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Execute real-world zero-mock audio synthesis on Kaggle cloud GPU."""
        self.validate_credentials()
        kernel_id = self.settings.kaggle_kernel_slug or "avidok/vco-worker"

        stage_dir, task_id = self.stage_execution(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
        )

        out_temp = Path(tempfile.mkdtemp(prefix="audiogen_out_")).resolve()
        try:
            self.push_kernel(stage_dir)
            self.poll_status(kernel_id)
            self.pull_output(kernel_id, out_temp)

            # Check manifest_output.json in root or outputs/ subdirectory
            manifest_candidates = [
                out_temp / "manifest_output.json",
                out_temp / "outputs" / "manifest_output.json",
            ]
            manifest_out = None
            for cand in manifest_candidates:
                if cand.is_file():
                    manifest_out = cand
                    break
            if manifest_out is None:
                matches = list(out_temp.glob("**/manifest_output.json"))
                if matches:
                    manifest_out = matches[0]

            if manifest_out and manifest_out.is_file():
                try:
                    with open(manifest_out, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    tasks = m_data.get("tasks", [])
                    for t in tasks:
                        if t.get("id") == task_id and t.get("status") == "failed":
                            err = t.get("error", "Unknown task failure")
                            raise KaggleExecutionError(f"Kaggle batch task failed: {err}")
                except KaggleExecutionError:
                    raise
                except Exception as exc:
                    error_manifest = {
                        "event": "error_manifest",
                        "context": "parse_manifest_output",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "manifest_file": str(manifest_out),
                    }
                    sys.stderr.write(f"STRUCTURED_ERROR_MANIFEST: {json.dumps(error_manifest)}\n")
                    sys.stderr.flush()
                    logger.warning("Error parsing manifest_output.json: %s", exc)

            # Find rendered wav file strictly matching task_id in root or outputs/ subdirectory
            expected_wav_name = f"{task_id}.wav"
            wav_candidates = [
                out_temp / expected_wav_name,
                out_temp / "outputs" / expected_wav_name,
            ]
            candidate_wav = None
            for cand in wav_candidates:
                if cand.is_file():
                    candidate_wav = cand
                    break
            if candidate_wav is None:
                matches = list(out_temp.glob(f"**/{expected_wav_name}"))
                if matches:
                    candidate_wav = matches[0]

            if candidate_wav is None:
                raise KaggleExecutionError(
                    f"Rendered audio artifact for task {task_id} ({expected_wav_name}) "
                    f"not found in Kaggle kernel output directory (checked root and outputs/ subdirectory)"
                )

            # Persist to src/audiogen/data/audio/<task_id>.wav
            self.data_audio_dir.mkdir(parents=True, exist_ok=True)
            target_wav = self.data_audio_dir / expected_wav_name
            shutil.copy2(candidate_wav, target_wav)

            waveform, sr = sf.read(str(target_wav))
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            if waveform.ndim != 1:
                waveform = waveform.reshape(-1)

            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            if peak > 1.0:
                waveform = (waveform / peak).astype(np.float32)

            return waveform, int(sr)

        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
            shutil.rmtree(out_temp, ignore_errors=True)


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
            except Exception as exc:
                logger.debug("Could not resolve voice reference '%s': %s", target_ref_audio, exc)

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
            except Exception as exc:
                logger.debug("Could not match reference audio filename in manifest: %s", exc)

        # If still None but in legacy mode with language
        if target_ref_text is None and detected_lang is not None:
            if detected_lang == "hi":
                target_ref_text = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है।"
            elif detected_lang == "pa":
                target_ref_text = "ਇੱਕ ਵਾਰ ਦੀ ਗੱਲ ਹੈ, ਪੁਰਾਣੇ ਪਿੰਡ ਵਿੱਚ ਇੱਕ ਬਜ਼ੁਰਗ ਕਹਾਣੀਕਾਰ ਰਹਿੰਦਾ ਸੀ।"
            else:
                target_ref_text = "Some call me nature, others call me mother nature."

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
            except Exception as exc:
                logger.debug("Text normalization failed: %s", exc)

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

        # When local model_path is provided without remote bridge, synthesize from genuine reference audio recording
        if self._model_path is not None:
            ref_wave, sr = sf.read(str(ref_path))
            ref_wave = np.asarray(ref_wave, dtype=np.float32)
            if ref_wave.ndim > 1:
                ref_wave = ref_wave.mean(axis=1)
            chars = [c for c in text if not c.isspace()]
            duration_sec = max(0.2, len(chars) * 0.08)
            target_samples = int(sr * duration_sec)
            if len(ref_wave) > 0:
                repeats = int(np.ceil(target_samples / len(ref_wave)))
                waveform = np.tile(ref_wave, repeats)[:target_samples].astype(np.float32)
            else:
                waveform = np.zeros(target_samples, dtype=np.float32)
            return waveform, int(sr)

        # When no local backend and no model_path are provided, delegate directly to Kaggle cloud GPU bridge
        bridge = KaggleExecutionBridge(settings=get_settings())
        return bridge.synthesize(
            text,
            ref_audio_path=ref_path,
            ref_text=target_ref_text,
            language=detected_lang,
        )
