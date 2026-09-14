"""Unit tests for batch manifest schema and validation contracts (Ticket 003)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from batch.manifest_schema import (
    BatchJob,
    BatchOutput,
    ReelAudioTask,
    TaskResult,
    load_manifest_file,
)
from voices.registry import VoiceNotFoundError, clear_registry_cache


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure in-memory registry cache is cleared before and after each test."""
    clear_registry_cache()
    yield
    clear_registry_cache()


def test_valid_reel_audio_task():
    """Validate valid Hindi and Punjabi tasks construct cleanly."""
    task_hi = ReelAudioTask(
        id="task_hi_01",
        text="नमस्ते, आज के मुख्य समाचार।",
        language="hi",
        voice_ref="anchor_female_calm",
    )
    assert task_hi.id == "task_hi_01"
    assert task_hi.language == "hi"
    assert task_hi.voice_ref == "anchor_female_calm"

    task_pa = ReelAudioTask(
        id="task_pa_01",
        text="ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਅੱਜ ਦਾ ਵਿਚਾਰ।",
        language="pa",
        voice_ref="storyteller_punjabi_elder",
    )
    assert task_pa.id == "task_pa_01"
    assert task_pa.language == "pa"
    assert task_pa.voice_ref == "storyteller_punjabi_elder"


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n"])
def test_empty_text_rejection(empty_text: str):
    """Manifest schema must reject empty or whitespace-only text."""
    with pytest.raises(ValidationError) as exc_info:
        ReelAudioTask(
            id="task_empty_text",
            text=empty_text,
            language="hi",
            voice_ref="anchor_male_energetic",
        )
    assert "Task text cannot be empty or whitespace-only" in str(exc_info.value)


@pytest.mark.parametrize("empty_id", ["", "   ", "\t"])
def test_empty_id_rejection(empty_id: str):
    """Manifest schema must reject empty or whitespace-only task IDs."""
    with pytest.raises(ValidationError) as exc_info:
        ReelAudioTask(
            id=empty_id,
            text="वैध पाठ",
            language="hi",
            voice_ref="anchor_male_energetic",
        )
    assert "Task ID cannot be empty or whitespace-only" in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_id",
    [
        "../traversal",
        "../../etc/passwd",
        "foo/bar",
        "task with spaces",
        "task.wav",
        "task$name",
        "task#1",
        "task;rm",
        "task\\backslash",
    ],
)
def test_task_id_sanitization_regex_invalid(invalid_id: str):
    """Manifest schema must reject task IDs that do not match ^[a-zA-Z0-9_-]+$."""
    with pytest.raises(ValidationError) as exc_info:
        ReelAudioTask(
            id=invalid_id,
            text="वैध पाठ",
            language="hi",
            voice_ref="anchor_male_energetic",
        )
    assert "must match '^[a-zA-Z0-9_-]+$'" in str(exc_info.value)


@pytest.mark.parametrize(
    "valid_id",
    ["reel_01", "task-123", "REEL_task_99-PA", "012345", "_task-ID_"],
)
def test_task_id_sanitization_regex_valid(valid_id: str):
    """Manifest schema must accept clean alphanumeric, underscore, and hyphen IDs."""
    task = ReelAudioTask(
        id=valid_id,
        text="वैध पाठ",
        language="hi",
        voice_ref="anchor_male_energetic",
    )
    assert task.id == valid_id


def test_duplicate_task_id_rejection():
    """BatchJob must reject manifests with duplicate task IDs."""
    t1 = ReelAudioTask(
        id="duplicate_id",
        text="पहला वाक्य",
        language="hi",
        voice_ref="anchor_male_energetic",
    )
    t2 = ReelAudioTask(
        id="duplicate_id",
        text="ਦੂਜਾ ਵਾਕ",
        language="pa",
        voice_ref="storyteller_punjabi_elder",
    )
    with pytest.raises(ValidationError) as exc_info:
        BatchJob(tasks=[t1, t2])
    assert "Duplicate task IDs found in manifest" in str(exc_info.value)



@pytest.mark.parametrize("invalid_lang", ["en", "fr", "es", ""])
def test_invalid_language_rejection(invalid_lang: str):
    """Manifest schema must reject unsupported language codes."""
    with pytest.raises(ValidationError):
        ReelAudioTask(
            id="task_invalid_lang",
            text="वैध पाठ",
            language=invalid_lang,  # type: ignore
            voice_ref="anchor_male_energetic",
        )


@pytest.mark.parametrize("bad_voice_ref", ["non_existent_voice", "", "   "])
def test_unresolvable_voice_ref_rejection(bad_voice_ref: str):
    """Manifest validation must reject any voice_ref not resolvable via voices.registry.get_voice_ref."""
    with pytest.raises(ValidationError) as exc_info:
        ReelAudioTask(
            id="task_bad_voice",
            text="वैध पाठ",
            language="hi",
            voice_ref=bad_voice_ref,
        )
    err_str = str(exc_info.value)
    assert "voice_ref" in err_str


def test_batch_job_model():
    """Verify BatchJob container requirements, including non-empty task list."""
    task1 = ReelAudioTask(
        id="t1",
        text="पहला वाक्य",
        language="hi",
        voice_ref="anchor_male_energetic",
    )
    job = BatchJob(tasks=[task1])
    assert len(job.tasks) == 1
    assert job.tasks[0].id == "t1"

    # Reject empty task list
    with pytest.raises(ValidationError) as exc_info:
        BatchJob(tasks=[])
    assert "BatchJob must contain at least one task" in str(exc_info.value)


def test_load_manifest_file(tmp_path: Path):
    """Verify load_manifest_file loads both list and dict formats, and fails on invalid inputs."""
    # 1. Non-existent file
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_manifest_file(missing_file)

    # 2. List format
    list_manifest = tmp_path / "list_manifest.json"
    tasks_data = [
        {
            "id": "t1",
            "text": "पहला वाक्य",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
        {
            "id": "t2",
            "text": "ਦੂਜਾ ਵਾਕ",
            "language": "pa",
            "voice_ref": "storyteller_punjabi_elder",
        },
    ]
    list_manifest.write_text(json.dumps(tasks_data), encoding="utf-8")
    job1 = load_manifest_file(list_manifest)
    assert len(job1.tasks) == 2
    assert job1.tasks[0].id == "t1"
    assert job1.tasks[1].id == "t2"

    # 3. Dict format with 'tasks' key
    dict_manifest = tmp_path / "dict_manifest.json"
    dict_manifest.write_text(json.dumps({"tasks": tasks_data}), encoding="utf-8")
    job2 = load_manifest_file(dict_manifest)
    assert len(job2.tasks) == 2

    # 4. Invalid structure format
    invalid_manifest = tmp_path / "invalid_manifest.json"
    invalid_manifest.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_manifest_file(invalid_manifest)
    assert "Invalid manifest structure" in str(exc_info.value)


def test_task_result_and_batch_output():
    """Verify TaskResult and BatchOutput models serialize and validate accurately."""
    r1 = TaskResult(id="t1", status="success", output_file="/path/to/t1.wav")
    r2 = TaskResult(id="t2", status="failed", error="Inference failure")

    summary = BatchOutput(
        total=2,
        successful=1,
        failed=1,
        tasks=[r1, r2],
    )
    assert summary.total == 2
    assert summary.successful == 1
    assert summary.failed == 1
    assert summary.tasks[0].output_file == "/path/to/t1.wav"
    assert summary.tasks[1].error == "Inference failure"
