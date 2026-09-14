"""Voice registry module for AudioGen.

Provides a cached, single shared source of truth mapping voice identifiers
to their reference audio files on disk with language filtering and fail-fast validation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Final, List, Optional, Union


class VoiceNotFoundError(KeyError):
    """Raised when a requested voice name is not present in the voice registry manifest."""

    pass


@dataclass(frozen=True)
class VoiceRecord:
    """Immutable voice record containing reference audio path, transcript, and languages."""

    path: str
    ref_text: str
    language: List[str]
    description: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        """Provide dictionary key access for backward compatibility."""
        if item in {"path", "ref_text", "language", "description"}:
            return getattr(self, item)
        raise KeyError(item)


    def to_dict(self) -> Dict[str, Any]:
        """Convert voice record to a standard dictionary."""
        d: Dict[str, Any] = {
            "path": self.path,
            "ref_text": self.ref_text,
            "language": list(self.language),
        }
        if self.description is not None:
            d["description"] = self.description
        return d

    def __fspath__(self) -> str:
        """Allow VoiceRecord to be used where a path-like object is expected."""
        return self.path


# Default path resolution constants
VOICE_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = VOICE_DIR.parent
DEFAULT_MANIFEST_PATH: Final[Path] = VOICE_DIR / "registry_schema.json"
ENV_VOICE_REGISTRY_PATH: Final[str] = "VOICE_REGISTRY_PATH"

# Thread-safe in-memory cache
_REGISTRY_LOCK: Final[threading.Lock] = threading.Lock()
_REGISTRY_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CACHED_MANIFEST_PATH: Optional[Path] = None


def resolve_manifest_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the absolute Path to the voice registry manifest file.

    Precedence:
    1. Explicit custom_path argument (if provided)
    2. VOICE_REGISTRY_PATH environment variable (if set and non-empty)
    3. Default manifest path at `voices/registry_schema.json`

    Args:
        custom_path: Optional explicit file path to registry manifest.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If the resolved manifest file does not exist on disk.
    """
    if custom_path is not None:
        path_str = str(custom_path).strip()
        if not path_str:
            target = DEFAULT_MANIFEST_PATH
        else:
            target = Path(path_str)
    else:
        env_val = os.environ.get(ENV_VOICE_REGISTRY_PATH)
        if env_val is not None and env_val.strip():
            target = Path(env_val.strip())
        else:
            target = DEFAULT_MANIFEST_PATH

    # If relative, resolve relative to current working directory or REPO_ROOT
    if not target.is_absolute():
        if target.exists():
            resolved = target.resolve()
        elif (REPO_ROOT / target).exists():
            resolved = (REPO_ROOT / target).resolve()
        else:
            resolved = target.resolve()
    else:
        resolved = target.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Voice registry manifest file not found: {resolved}")

    return resolved


def load_manifest(
    manifest_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load and cache the voice registry JSON manifest.

    Loads the JSON manifest from disk once and caches it in memory. Thread-safe.
    Subsequent calls return the cached data unless force_reload=True or a different
    path is explicitly supplied.

    Args:
        manifest_path: Optional path override for manifest.
        force_reload: If True, bypasses cache and re-reads file from disk.

    Returns:
        Dictionary mapping voice_name (str) to voice metadata dictionary.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If JSON is malformed, not a dictionary, or contains invalid voice entries.
        json.JSONDecodeError: If JSON syntax is invalid.
    """
    global _REGISTRY_CACHE, _CACHED_MANIFEST_PATH

    with _REGISTRY_LOCK:
        resolved_path = resolve_manifest_path(manifest_path)

        if not force_reload and _REGISTRY_CACHE is not None and _CACHED_MANIFEST_PATH == resolved_path:
            return copy.deepcopy(_REGISTRY_CACHE)

        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Voice registry manifest must be a JSON object mapping voice names to metadata.")

        processed: Dict[str, Dict[str, Any]] = {}
        for voice_name, metadata in data.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"Voice entry '{voice_name}' must be a dictionary.")
            if "path" not in metadata or "language" not in metadata:
                raise ValueError(f"Voice entry '{voice_name}' missing required field: 'path' or 'language'")

            if (
                "ref_text" not in metadata
                or not isinstance(metadata["ref_text"], str)
                or not metadata["ref_text"].strip()
            ):
                raise ValueError(f"Voice entry '{voice_name}' missing or empty required field: 'ref_text'")

            entry = metadata.copy()
            lang = entry["language"]
            if isinstance(lang, str):
                entry["language"] = [lang]
            elif isinstance(lang, list):
                entry["language"] = [str(item) for item in lang]
            else:
                raise ValueError(f"Voice entry '{voice_name}' has invalid 'language' type: {type(lang).__name__}")

            for l in entry["language"]:
                if l not in {"hi", "pa"}:
                    raise ValueError(f"Unsupported language '{l}' in voice entry '{voice_name}'")

            entry["ref_text"] = entry["ref_text"].strip()
            processed[voice_name] = entry

        _REGISTRY_CACHE = processed
        _CACHED_MANIFEST_PATH = resolved_path
        return copy.deepcopy(_REGISTRY_CACHE)


def clear_registry_cache() -> None:
    """Reset the in-memory manifest cache.

    Primarily used by test suites to ensure clean isolation between test cases.
    """
    global _REGISTRY_CACHE, _CACHED_MANIFEST_PATH
    with _REGISTRY_LOCK:
        _REGISTRY_CACHE = None
        _CACHED_MANIFEST_PATH = None


def get_voice_ref(voice_name: str) -> VoiceRecord:
    """Retrieve the VoiceRecord for a given voice.

    Validates that the voice name exists in the manifest, the reference text is
    valid, and the referenced audio file exists on disk. Fails fast if any
    condition is not met.

    Args:
        voice_name: Unique voice identifier key (e.g. 'anchor_male_energetic').

    Returns:
        VoiceRecord containing path, ref_text, and language list.

    Raises:
        TypeError: If voice_name is not a string.
        VoiceNotFoundError: If voice_name is not in the registry manifest (inherits from KeyError).
        ValueError: If ref_text is missing or empty.
        FileNotFoundError: If the voice exists in manifest but its audio file is missing on disk.
    """
    if not isinstance(voice_name, str):
        raise TypeError(f"Expected voice_name to be str, got {type(voice_name).__name__}")

    manifest = load_manifest()

    if voice_name not in manifest:
        raise VoiceNotFoundError(f"Voice '{voice_name}' not found in voice registry manifest.")

    entry = manifest[voice_name]
    ref_text = entry.get("ref_text")
    if not ref_text or not isinstance(ref_text, str) or not ref_text.strip():
        raise ValueError(
            f"Reference text (ref_text) is missing or empty for voice '{voice_name}'. Corrupt manifest entry."
        )

    raw_path_str = entry.get("path")
    if not raw_path_str:
        raise FileNotFoundError(f"Reference audio file path for voice '{voice_name}' is empty.")

    raw_path = Path(raw_path_str)
    if raw_path.is_absolute():
        resolved_audio_path = raw_path.resolve()
    else:
        manifest_dir = _CACHED_MANIFEST_PATH.parent if _CACHED_MANIFEST_PATH else VOICE_DIR
        candidate_manifest = (manifest_dir / raw_path).resolve()
        candidate_repo = (REPO_ROOT / raw_path).resolve()

        if candidate_manifest.exists():
            resolved_audio_path = candidate_manifest
        elif candidate_repo.exists():
            resolved_audio_path = candidate_repo
        else:
            try:
                manifest_dir.relative_to(REPO_ROOT)
                resolved_audio_path = candidate_repo
            except ValueError:
                resolved_audio_path = candidate_manifest

    if not resolved_audio_path.is_file():
        raise FileNotFoundError(
            f"Reference audio file for voice '{voice_name}' does not exist on disk: {resolved_audio_path}"
        )

    return VoiceRecord(
        path=str(resolved_audio_path),
        ref_text=ref_text.strip(),
        language=list(entry.get("language", [])),
        description=entry.get("description"),
    )


def list_voices(language: Optional[str] = None) -> List[str]:
    """List all registered voice names, optionally filtered by language code.

    Args:
        language: Optional ISO language code (e.g. 'hi', 'pa'). Case-insensitive.
                  If None, returns all registered voices.

    Returns:
        Sorted list of voice names matching the language criteria (or all voices if language is None).
        Returns empty list [] if no voices match the specified language.

    Raises:
        TypeError: If language is provided and is not a string.
    """
    if language is not None and not isinstance(language, str):
        raise TypeError(f"Expected language to be str or None, got {type(language).__name__}")

    manifest = load_manifest()

    if language is None:
        return sorted(manifest.keys())

    target_lang = language.strip().lower()
    matching: List[str] = []
    for voice_name, metadata in manifest.items():
        supported_langs = [str(item).strip().lower() for item in metadata.get("language", [])]
        if target_lang in supported_langs:
            matching.append(voice_name)

    return sorted(matching)


def get_voice_metadata(voice_name: str) -> Dict[str, Any]:
    """Retrieve the metadata dictionary for a specific voice.

    Args:
        voice_name: Unique voice identifier key.

    Returns:
        A shallow copy of the voice metadata dictionary containing keys like
        'path', 'language', and optional descriptive fields.

    Raises:
        TypeError: If voice_name is not a string.
        VoiceNotFoundError: If voice_name is not in the registry manifest.
    """
    if not isinstance(voice_name, str):
        raise TypeError(f"Expected voice_name to be str, got {type(voice_name).__name__}")

    manifest = load_manifest()

    if voice_name not in manifest:
        raise VoiceNotFoundError(f"Voice '{voice_name}' not found in voice registry manifest.")

    return copy.deepcopy(manifest[voice_name])


__all__ = [
    "VoiceNotFoundError",
    "VoiceRecord",
    "resolve_manifest_path",
    "load_manifest",
    "clear_registry_cache",
    "get_voice_ref",
    "list_voices",
    "get_voice_metadata",
]
