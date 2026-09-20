"""Unit test suite for voice registry with IndicF5 VoiceRecord support."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import struct
from typing import Any, Dict, Generator
from unittest.mock import patch
import wave

import pytest

import voices.registry
from voices.registry import (
    DEFAULT_MANIFEST_PATH,
    ENV_VOICE_REGISTRY_PATH,
    REPO_ROOT,
    VOICE_DIR,
    VoiceNotFoundError,
    VoiceRecord,
    clear_registry_cache,
    get_voice_metadata,
    get_voice_ref,
    list_voices,
    load_manifest,
    register_voice,
    resolve_manifest_path,
)


def _create_dummy_wav(path: Path, duration_sec: float = 0.1) -> None:
    """Helper to create a minimal valid 16-bit PCM mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24000
    num_frames = int(sample_rate * duration_sec)
    frames = struct.pack(f"<{num_frames}h", *(0 for _ in range(num_frames)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


@pytest.fixture(autouse=True)
def clean_registry_cache_fixture() -> Generator[None, None, None]:
    """Ensure in-memory manifest cache is cleared before and after each test."""
    clear_registry_cache()
    yield
    clear_registry_cache()


@pytest.fixture
def temp_voice_workspace(tmp_path: Path) -> Path:
    """Create an isolated voice workspace with dummy wav files."""
    voices_dir = tmp_path / "voices"
    refs_dir = voices_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    _create_dummy_wav(refs_dir / "v1.wav")
    _create_dummy_wav(refs_dir / "v2.wav")
    _create_dummy_wav(refs_dir / "v3.wav")
    _create_dummy_wav(refs_dir / "v4.wav")

    return voices_dir


@pytest.fixture
def mock_manifest_file(temp_voice_workspace: Path) -> Path:
    """Create a test manifest with known test voices in temp workspace."""
    manifest_data: Dict[str, Any] = {
        "voice_hi_only": {
            "path": "refs/v1.wav",
            "ref_text": "नमस्ते भारत, यह एक परीक्षण है।",
            "language": ["hi"],
            "description": "Hindi only voice",
        },
        "voice_pa_only": {
            "path": "refs/v2.wav",
            "ref_text": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ, ਇਹ ਇਕ ਟੈਸਟ ਹੈ।",
            "language": ["pa"],
            "description": "Punjabi only voice",
        },
        "voice_bilingual": {
            "path": "refs/v3.wav",
            "ref_text": "नमस्ते और ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਦੋਵਾਂ ਭਾਸ਼ਾਵਾਂ ਲਈ।",
            "language": ["hi", "pa"],
            "description": "Bilingual voice",
        },
        "voice_string_lang": {
            "path": "refs/v4.wav",
            "ref_text": "नमस्ते यह स्ट्रिंग भाषा का परीक्षण है।",
            "language": "hi",
            "description": "Voice with string language representation",
        },
    }
    manifest_path = temp_voice_workspace / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_path


@pytest.fixture
def missing_ref_manifest(temp_voice_workspace: Path) -> Path:
    """Create a manifest pointing to a non-existent audio file."""
    manifest_data = {
        "ghost_voice": {
            "path": "refs/ghost_voice.wav",
            "ref_text": "यह एक भूतिया आवाज़ का परीक्षण है।",
            "language": ["hi"],
            "description": "Voice pointing to non-existent audio file",
        }
    }
    manifest_path = temp_voice_workspace / "ghost_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_path


@pytest.fixture
def malformed_json_file(tmp_path: Path) -> Path:
    """Create a file with unparseable JSON syntax."""
    malformed_path = tmp_path / "malformed.json"
    with open(malformed_path, "w", encoding="utf-8") as f:
        f.write("{ unclosed: json ")
    return malformed_path


# ==============================================================================
# 1. Error Hierarchy Tests
# ==============================================================================


def test_voice_not_found_error_hierarchy() -> None:
    """Verify VoiceNotFoundError subclasses KeyError for legacy compatibility."""
    assert issubclass(VoiceNotFoundError, KeyError)

    with pytest.raises(KeyError):
        raise VoiceNotFoundError("Voice 'test' not found in voice registry manifest.")

    with pytest.raises(VoiceNotFoundError) as exc_info:
        raise VoiceNotFoundError("Voice 'test' not found in voice registry manifest.")
    assert "test" in str(exc_info.value)


# ==============================================================================
# 2. Manifest Loading & Schema Validation Tests
# ==============================================================================


def test_default_manifest_loads_successfully() -> None:
    """Verify default production manifest loads and contains required default voices with transcripts."""
    manifest = load_manifest()
    assert isinstance(manifest, dict)
    assert "anchor_male_energetic" in manifest
    assert "anchor_female_calm" in manifest
    assert "storyteller_punjabi_elder" in manifest

    # Check structure
    male = manifest["anchor_male_energetic"]
    assert "path" in male
    assert "ref_text" in male
    assert isinstance(male["ref_text"], str) and len(male["ref_text"]) > 0
    assert "language" in male
    assert isinstance(male["language"], list)
    assert "hi" in male["language"]
    assert "pa" in male["language"]


def test_load_manifest_string_language_coercion(mock_manifest_file: Path) -> None:
    """Verify scalar string language is coerced to a list of strings."""
    manifest = load_manifest(mock_manifest_file)
    assert manifest["voice_string_lang"]["language"] == ["hi"]


def test_load_manifest_malformed_json(malformed_json_file: Path) -> None:
    """Verify loading invalid JSON raises JSONDecodeError or ValueError."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_manifest(malformed_json_file)


def test_load_manifest_invalid_root_type(tmp_path: Path) -> None:
    """Verify manifest with non-dictionary root raises ValueError."""
    invalid_path = tmp_path / "list_root.json"
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump(["item1", "item2"], f)

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_manifest(invalid_path)


def test_load_manifest_missing_required_fields(tmp_path: Path) -> None:
    """Verify missing required fields ('path' or 'language') raise ValueError."""
    # Missing language
    p1 = tmp_path / "no_lang.json"
    with open(p1, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": {"path": "refs/v1.wav", "ref_text": "परीक्षण"}}, f)
    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(p1)

    # Missing path
    p2 = tmp_path / "no_path.json"
    with open(p2, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": {"language": ["hi"], "ref_text": "परीक्षण"}}, f)
    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(p2)

    # Value is not a dictionary
    p3 = tmp_path / "not_dict.json"
    with open(p3, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": "invalid_string"}, f)
    with pytest.raises(ValueError, match="must be a dictionary"):
        load_manifest(p3)


def test_load_manifest_missing_ref_text_raises_value_error(tmp_path: Path) -> None:
    """Verify loading manifest with missing ref_text raises ValueError."""
    bad_manifest = tmp_path / "no_ref_text.json"
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": {"path": "refs/v1.wav", "language": ["hi"]}}, f)
    with pytest.raises(ValueError, match="missing or empty required field: 'ref_text'"):
        load_manifest(bad_manifest)


def test_load_manifest_empty_ref_text_raises_value_error(tmp_path: Path) -> None:
    """Verify loading manifest with empty or whitespace-only ref_text raises ValueError."""
    bad_manifest = tmp_path / "empty_ref_text.json"
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": {"path": "refs/v1.wav", "ref_text": "   ", "language": ["hi"]}}, f)
    with pytest.raises(ValueError, match="missing or empty required field: 'ref_text'"):
        load_manifest(bad_manifest)


def test_load_manifest_unsupported_language_raises_value_error(tmp_path: Path) -> None:
    """Verify loading manifest with language other than en/hi/pa raises ValueError."""
    bad_manifest = tmp_path / "unsupported_lang.json"
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump({"bad_voice": {"path": "refs/v1.wav", "ref_text": "Hola mundo", "language": ["es"]}}, f)
    with pytest.raises(ValueError, match="Unsupported language 'es' in voice entry 'bad_voice'"):
        load_manifest(bad_manifest)


def test_load_manifest_accepts_english_language(tmp_path: Path) -> None:
    """Verify loading manifest with English language 'en' succeeds."""
    manifest_path = tmp_path / "en_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"en_voice": {"path": "refs/v1.wav", "ref_text": "Hello world", "language": ["en"]}}, f)
    loaded = load_manifest(manifest_path)
    assert "en_voice" in loaded
    assert loaded["en_voice"]["language"] == ["en"]


def test_register_voice_atomic_disk_update(temp_voice_workspace: Path) -> None:
    """Verify register_voice persists new voice atomically to disk and cache."""
    manifest_path = temp_voice_workspace / "custom_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    test_audio = temp_voice_workspace / "refs" / "v1.wav"
    rec = voices.registry.register_voice(
        voice_name="test_voice_1",
        ref_audio_path=test_audio,
        ref_text="This is a test transcript for voice 1.",
        languages=["en"],
        description="Custom test voice",
        manifest_path=manifest_path,
    )
    assert isinstance(rec, voices.registry.VoiceRecord)
    assert rec.ref_text == "This is a test transcript for voice 1."
    assert rec.language == ["en"]

    # Verify persisted on disk
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "test_voice_1" in data
    assert data["test_voice_1"]["ref_text"] == "This is a test transcript for voice 1."


def test_register_voice_validation_errors(temp_voice_workspace: Path) -> None:
    """Verify register_voice validation checks for invalid name, empty text, or bad language."""
    test_audio = temp_voice_workspace / "refs" / "v1.wav"

    with pytest.raises(ValueError, match="Invalid voice identifier"):
        voices.registry.register_voice("bad voice with spaces", test_audio, "text", ["en"])

    with pytest.raises(ValueError, match="ref_text"):
        voices.registry.register_voice("good_voice", test_audio, "   ", ["en"])

    with pytest.raises(ValueError, match="Unsupported language 'es'"):
        voices.registry.register_voice("good_voice", test_audio, "text", ["es"])


def test_register_voice_thread_safety(temp_voice_workspace: Path) -> None:
    """Verify concurrent register_voice calls safely update manifest without corruption."""
    manifest_path = temp_voice_workspace / "concurrent_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    test_audio = temp_voice_workspace / "refs" / "v1.wav"
    errors = []

    def worker(i: int):
        try:
            voices.registry.register_voice(
                voice_name=f"thread_voice_{i}",
                ref_audio_path=test_audio,
                ref_text=f"Thread voice transcript number {i}",
                languages=["en"],
                manifest_path=manifest_path,
            )
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 8


# ==============================================================================
# 3. Reference Path Resolution & Lookup Tests
# ==============================================================================


def test_get_voice_ref_success() -> None:
    """Verify get_voice_ref returns VoiceRecord with valid attributes and dict access."""
    for voice_name in ["anchor_male_energetic", "anchor_female_calm", "storyteller_punjabi_elder"]:
        voice_rec = get_voice_ref(voice_name)
        assert isinstance(voice_rec, VoiceRecord)
        assert isinstance(voice_rec.path, str)
        assert os.path.isabs(voice_rec.path)
        assert Path(voice_rec.path).exists()
        assert Path(voice_rec.path).is_file()
        assert voice_rec.path.endswith(f"{voice_name}.wav")

        assert isinstance(voice_rec.ref_text, str)
        assert len(voice_rec.ref_text.strip()) > 0
        assert isinstance(voice_rec.language, list)
        assert len(voice_rec.language) > 0

        # Dict-like indexing
        assert voice_rec["path"] == voice_rec.path
        assert voice_rec["ref_text"] == voice_rec.ref_text
        assert voice_rec["language"] == voice_rec.language
        assert voice_rec["description"] == voice_rec.description

        # VoiceRecord with description=None returns None for ["description"]
        rec_no_desc = VoiceRecord(path=voice_rec.path, ref_text=voice_rec.ref_text, language=voice_rec.language)
        assert rec_no_desc["description"] is None

        # Invalid key access restricted to valid fields
        with pytest.raises(KeyError):
            _ = voice_rec["non_existent_key"]
        with pytest.raises(KeyError):
            _ = voice_rec["to_dict"]

        # to_dict conversion
        d = voice_rec.to_dict()
        assert isinstance(d, dict)
        assert d["path"] == voice_rec.path
        assert d["ref_text"] == voice_rec.ref_text
        assert d["language"] == voice_rec.language



def test_get_voice_ref_not_found() -> None:
    """Verify get_voice_ref raises VoiceNotFoundError for unknown voice name."""
    with pytest.raises(VoiceNotFoundError) as exc_info:
        get_voice_ref("non_existent_voice")
    assert issubclass(exc_info.type, KeyError)
    assert "non_existent_voice" in str(exc_info.value)


def test_get_voice_ref_type_error() -> None:
    """Verify get_voice_ref raises TypeError for non-string input."""
    with pytest.raises(TypeError, match="Expected voice_name to be str"):
        get_voice_ref(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected voice_name to be str"):
        get_voice_ref(None)  # type: ignore[arg-type]


def test_get_voice_ref_missing_file_on_disk(missing_ref_manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_voice_ref fails fast with FileNotFoundError if audio file is missing."""
    monkeypatch.setenv(ENV_VOICE_REGISTRY_PATH, str(missing_ref_manifest))
    with pytest.raises(FileNotFoundError) as exc_info:
        get_voice_ref("ghost_voice")
    assert "ghost_voice" in str(exc_info.value)
    assert "does not exist on disk" in str(exc_info.value)


def test_get_voice_ref_corrupt_manifest_empty_ref_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_voice_ref rejects corrupt cache entries with missing/empty ref_text at lookup time."""
    from voices import registry

    corrupt_cache = {
        "corrupt_voice": {
            "path": "voices/refs/anchor_male_energetic.wav",
            "ref_text": "  ",
            "language": ["hi"],
        }
    }
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", corrupt_cache)
    monkeypatch.setattr(registry, "_CACHED_MANIFEST_PATH", registry.resolve_manifest_path())
    with pytest.raises(ValueError, match="Reference text \\(ref_text\\) is missing or empty for voice 'corrupt_voice'"):
        get_voice_ref("corrupt_voice")



# ==============================================================================
# 4. Voice Listing & Language Filtering Tests
# ==============================================================================


def test_list_voices_unfiltered() -> None:
    """Verify list_voices() returns all registered voice names sorted alphabetically."""
    voices = list_voices()
    assert isinstance(voices, list)
    assert voices == sorted(voices)
    assert "anchor_female_calm" in voices
    assert "anchor_male_energetic" in voices
    assert "storyteller_punjabi_elder" in voices


def test_list_voices_filtered_by_language() -> None:
    """Verify list_voices() correctly filters by ISO language code."""
    hi_voices = list_voices("hi")
    assert "anchor_male_energetic" in hi_voices
    assert "anchor_female_calm" in hi_voices
    assert "storyteller_punjabi_elder" not in hi_voices
    assert hi_voices == sorted(hi_voices)

    pa_voices = list_voices("pa")
    assert "anchor_male_energetic" in pa_voices
    assert "storyteller_punjabi_elder" in pa_voices
    assert "anchor_female_calm" not in pa_voices
    assert pa_voices == sorted(pa_voices)


def test_list_voices_case_insensitivity_and_whitespace() -> None:
    """Verify list_voices normalizes case and whitespace in language filter."""
    assert list_voices(" HI ") == list_voices("hi")
    assert list_voices("Pa") == list_voices("pa")


def test_list_voices_no_matches_returns_empty_list() -> None:
    """Verify list_voices returns empty list if no voices match language."""
    assert list_voices("unknown_language") == []


def test_list_voices_type_error() -> None:
    """Verify list_voices raises TypeError for non-string language."""
    with pytest.raises(TypeError, match="Expected language to be str or None"):
        list_voices(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected language to be str or None"):
        list_voices(["hi"])  # type: ignore[arg-type]


# ==============================================================================
# 5. Metadata Retrieval Tests
# ==============================================================================


def test_get_voice_metadata_success() -> None:
    """Verify get_voice_metadata returns shallow copy of voice metadata."""
    meta = get_voice_metadata("anchor_male_energetic")
    assert isinstance(meta, dict)
    assert meta["path"] == "voices/refs/anchor_male_energetic.wav"
    assert meta["language"] == ["hi", "pa"]
    assert "ref_text" in meta
    assert len(meta["ref_text"]) > 0
    assert "description" in meta

    # Ensure immutability / shallow copy isolation
    meta["path"] = "tampered_path"
    fresh_meta = get_voice_metadata("anchor_male_energetic")
    assert fresh_meta["path"] == "voices/refs/anchor_male_energetic.wav"


def test_get_voice_metadata_not_found() -> None:
    """Verify get_voice_metadata raises VoiceNotFoundError for missing voice."""
    with pytest.raises(VoiceNotFoundError):
        get_voice_metadata("unknown_voice")


def test_get_voice_metadata_type_error() -> None:
    """Verify get_voice_metadata raises TypeError for non-string voice name."""
    with pytest.raises(TypeError, match="Expected voice_name to be str"):
        get_voice_metadata(999)  # type: ignore[arg-type]


# ==============================================================================
# 6. Caching, Manifest Path Precedence, & Concurrency Tests
# ==============================================================================


def test_single_load_caching_invariant() -> None:
    """Verify the manifest file is loaded and parsed exactly once across multiple calls."""
    with patch("json.load", wraps=json.load) as spy_json_load:
        # 10 consecutive lookups
        for _ in range(10):
            get_voice_ref("anchor_male_energetic")
            list_voices()
            get_voice_metadata("anchor_male_energetic")

        # Must have read/parsed the JSON file exactly once
        assert spy_json_load.call_count == 1

        # Force reload triggers another load
        load_manifest(force_reload=True)
        assert spy_json_load.call_count == 2


def test_resolve_manifest_path_precedence(
    mock_manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify manifest path resolution precedence (explicit > env var > default)."""
    # 1. Default path
    assert resolve_manifest_path() == DEFAULT_MANIFEST_PATH.resolve()

    # 2. Env variable precedence
    monkeypatch.setenv(ENV_VOICE_REGISTRY_PATH, str(mock_manifest_file))
    assert resolve_manifest_path() == mock_manifest_file.resolve()

    # 3. Explicit argument overrides env variable
    explicit_file = tmp_path / "explicit.json"
    with open(explicit_file, "w", encoding="utf-8") as f:
        json.dump({}, f)
    assert resolve_manifest_path(explicit_file) == explicit_file.resolve()

    # 4. Non-existent path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Voice registry manifest file not found"):
        resolve_manifest_path(tmp_path / "does_not_exist.json")


def test_concurrent_multithreaded_lookups() -> None:
    """Verify registry is safe for concurrent access across worker threads."""
    errors = []

    def worker(idx: int) -> None:
        try:
            for _ in range(20):
                ref = get_voice_ref("anchor_male_energetic")
                assert os.path.exists(ref.path)
                voices = list_voices("hi")
                assert "anchor_male_energetic" in voices
                meta = get_voice_metadata("anchor_female_calm")
                assert meta["language"] == ["hi"]
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(16)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0


def test_cache_immutability_and_deep_isolation() -> None:
    """Verify mutating manifest or metadata does not pollute the cached registry."""
    manifest1 = load_manifest()
    manifest1["anchor_male_energetic"]["language"].append("fr")
    manifest1["new_polluting_key"] = {}

    manifest2 = load_manifest()
    assert "new_polluting_key" not in manifest2
    assert "fr" not in manifest2["anchor_male_energetic"]["language"]

    meta = get_voice_metadata("anchor_male_energetic")
    meta["language"].append("de")
    fresh_meta = get_voice_metadata("anchor_male_energetic")
    assert "de" not in fresh_meta["language"]


def test_registered_reference_audio_health() -> None:
    """Verify all registered voice references have valid audio, duration >= 1.0s, and non-silent waveform."""
    import numpy as np

    manifest = load_manifest()
    assert len(manifest) >= 3
    assert "anchor_male_energetic" in manifest
    assert "storyteller_punjabi_elder" in manifest
    assert "anchor_female_calm" in manifest

    for voice_name in manifest:
        voice_rec = get_voice_ref(voice_name)
        file_path = Path(voice_rec.path)
        assert file_path.is_file(), f"Audio file for '{voice_name}' not found: {file_path}"

        with wave.open(str(file_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / framerate

            assert channels == 1, f"Voice '{voice_name}' must be mono, got {channels} channels"
            assert sampwidth == 2, f"Voice '{voice_name}' must be 16-bit PCM, got sampwidth={sampwidth}"
            assert framerate == 24000, f"Voice '{voice_name}' must be 24 kHz, got {framerate} Hz"
            assert duration >= 1.0, f"Voice '{voice_name}' duration {duration:.2f}s < 1.0s"
            assert duration >= 3.0, f"Voice '{voice_name}' duration {duration:.2f}s < 3.0s recommended minimum"

            raw_bytes = wf.readframes(frames)
            audio_samples = np.frombuffer(raw_bytes, dtype=np.int16)
            max_amplitude = int(np.max(np.abs(audio_samples)))
            assert max_amplitude > 100, f"Voice '{voice_name}' audio is silent (max_amp={max_amplitude})"


def test_registered_reference_transcripts_authentic() -> None:
    """Verify reference transcripts match genuine speech text in manifest for all voices."""
    manifest = load_manifest()
    anchor = manifest["anchor_male_energetic"]
    assert "ਭਹੰਪੀ" in anchor["ref_text"]
    assert len(anchor["ref_text"]) > 20

    elder = manifest["storyteller_punjabi_elder"]
    assert "ਬਜ਼ੁਰਗ" in elder["ref_text"]
    assert len(elder["ref_text"]) > 20

    female = manifest["anchor_female_calm"]
    assert "नमस्ते" in female["ref_text"]
    assert len(female["ref_text"]) > 20


