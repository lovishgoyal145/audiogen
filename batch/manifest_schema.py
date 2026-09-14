"""Pydantic schemas and validation for batch reel audio synthesis jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from voices.registry import get_voice_ref, VoiceNotFoundError

TASK_ID_REGEX: Final[re.Pattern] = re.compile(r"^[a-zA-Z0-9_-]+$")


class ReelAudioTask(BaseModel):
    """Specification for a single reel audio synthesis task."""

    id: str = Field(..., description="Unique task identifier.")
    text: str = Field(..., description="Raw text script in Hindi or Punjabi.")
    language: Literal["hi", "pa"] = Field(..., description="Language code: 'hi' or 'pa'.")
    voice_ref: str = Field(..., description="Voice identifier key in the voice registry.")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Task ID cannot be empty or whitespace-only.")
        cleaned = v.strip()
        if not TASK_ID_REGEX.match(cleaned):
            raise ValueError(
                f"Task ID '{cleaned}' is invalid: must match '^[a-zA-Z0-9_-]+$' "
                "to prevent directory traversal and file path exploits."
            )
        return cleaned

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Task text cannot be empty or whitespace-only.")
        return v

    @field_validator("voice_ref")
    @classmethod
    def validate_voice_ref(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("voice_ref cannot be empty or whitespace-only.")
        try:
            get_voice_ref(v)
        except Exception as exc:
            raise ValueError(
                f"voice_ref '{v}' is not resolvable via voices.registry.get_voice_ref: {exc}"
            ) from exc
        return v


class BatchJob(BaseModel):
    """Container for a batch audio synthesis job containing multiple tasks."""

    tasks: List[ReelAudioTask] = Field(..., description="List of audio tasks to synthesize.")

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: List[ReelAudioTask]) -> List[ReelAudioTask]:
        if not v:
            raise ValueError("BatchJob must contain at least one task.")
        seen_ids: set[str] = set()
        duplicate_ids: List[str] = []
        for task in v:
            if task.id in seen_ids:
                duplicate_ids.append(task.id)
            seen_ids.add(task.id)
        if duplicate_ids:
            unique_dups = sorted(set(duplicate_ids))
            raise ValueError(f"Duplicate task IDs found in manifest: {unique_dups}")
        return v


class TaskResult(BaseModel):
    """Execution result for a single task."""

    id: str
    status: Literal["success", "failed"]
    output_file: Optional[str] = None
    error: Optional[str] = None


class BatchOutput(BaseModel):
    """Aggregate output summary written to manifest_output.json."""

    total: int
    successful: int
    failed: int
    tasks: List[TaskResult]


def load_manifest_file(manifest_path: Union[str, Path]) -> BatchJob:
    """Load and validate a BatchJob from a JSON file path.

    Supports both top-level list of tasks or dict with 'tasks' key.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return BatchJob(tasks=data)
    elif isinstance(data, dict) and "tasks" in data:
        return BatchJob(**data)
    else:
        raise ValueError(f"Invalid manifest structure in {p}: expected list or dict with 'tasks'")
