"""Comprehensive unit and integration tests for KaggleExecutionBridge and E2E generation pipeline.

Tests credential validation gate, payload packing, notebook cell generation,
status polling lifecycle, timeout protection, output pulling, WAV persistence,
and the zero-mock invariant.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Generator
from unittest.mock import MagicMock, patch
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from audiogen.config import (
    MissingKaggleCredentialsError,
    Settings,
    get_settings,
    validate_kaggle_credentials,
)
from audiogen.engine import (
    KaggleExecutionBridge,
    KaggleExecutionError,
    KaggleTimeoutError,
    Synthesizer,
)
from audiogen.main import REPO_ROOT, app
import voices.registry


def _create_test_wav(path: Path, duration_sec: float = 0.2, sample_rate: int = 24000) -> None:
    """Helper to generate a valid 16-bit PCM mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_frames, endpoint=False)
    # 220Hz test tone
    audio_data = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


@pytest.fixture(autouse=True)
def clean_kaggle_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Isolate environment variables and application state between tests."""
    # Ensure Kaggle credentials are unset by default
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_SLUG", raising=False)

    app.dependency_overrides.clear()
    app.state.custom_voices = {}
    app.state.synthesizer = None
    get_settings.cache_clear()

    yield

    app.dependency_overrides.clear()
    app.state.custom_voices = {}
    app.state.synthesizer = None
    get_settings.cache_clear()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_reference_wav(tmp_path: Path) -> Path:
    """Fixture providing a temporary reference WAV file."""
    wav_path = tmp_path / "reference.wav"
    _create_test_wav(wav_path)
    return wav_path


# ==============================================================================
# 1. Credential Gate Tests
# ==============================================================================


def test_credential_gate_rejects_missing_both_credentials(client: TestClient) -> None:
    """Test POST /api/generate returns HTTP 422 when both KAGGLE_USERNAME and KAGGLE_KEY are missing."""
    payload = {
        "text": "Hello world from AudioGen speech synthesis engine.",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    expected_msg = "Missing required Kaggle API credentials in .env: KAGGLE_USERNAME, KAGGLE_KEY"
    assert data["detail"] == expected_msg


def test_credential_gate_rejects_missing_username_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test POST /api/generate returns HTTP 422 specifying KAGGLE_USERNAME when key is provided."""
    monkeypatch.setenv("KAGGLE_KEY", "test_kaggle_api_secret_key")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    get_settings.cache_clear()

    payload = {
        "text": "Hello world from AudioGen speech synthesis engine.",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"] == "Missing required Kaggle API credentials in .env: KAGGLE_USERNAME"


def test_credential_gate_rejects_missing_key_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test POST /api/generate returns HTTP 422 specifying KAGGLE_KEY when username is provided."""
    monkeypatch.setenv("KAGGLE_USERNAME", "test_kaggle_user")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    get_settings.cache_clear()

    payload = {
        "text": "Hello world from AudioGen speech synthesis engine.",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"] == "Missing required Kaggle API credentials in .env: KAGGLE_KEY"


def test_direct_credential_validator_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_kaggle_credentials and bridge raise MissingKaggleCredentialsError."""
    # Test neither provided
    with pytest.raises(MissingKaggleCredentialsError) as exc_info:
        validate_kaggle_credentials()
    assert "KAGGLE_USERNAME, KAGGLE_KEY" in str(exc_info.value)

    # Test username only provided
    monkeypatch.setenv("KAGGLE_USERNAME", "myuser")
    get_settings.cache_clear()
    with pytest.raises(MissingKaggleCredentialsError) as exc_info:
        validate_kaggle_credentials()
    assert "KAGGLE_KEY" in str(exc_info.value)
    assert "KAGGLE_USERNAME" not in str(exc_info.value)

    # Test key only provided
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_KEY", "mykey")
    get_settings.cache_clear()
    with pytest.raises(MissingKaggleCredentialsError) as exc_info:
        validate_kaggle_credentials()
    assert "KAGGLE_USERNAME" in str(exc_info.value)

    # Test both provided passes
    monkeypatch.setenv("KAGGLE_USERNAME", "myuser")
    monkeypatch.setenv("KAGGLE_KEY", "mykey")
    get_settings.cache_clear()
    settings = validate_kaggle_credentials()
    assert settings.kaggle_username == "myuser"
    assert settings.kaggle_key == "mykey"

    bridge = KaggleExecutionBridge(settings=settings)
    bridge.validate_credentials()  # Should not raise


# ==============================================================================
# 2. Payload Packing and Manifest Construction
# ==============================================================================


def test_stage_execution_creates_valid_manifests_and_notebook(
    dummy_reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify scripts.json, kernel-metadata.json, and run.ipynb structure."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    monkeypatch.setenv("KAGGLE_KERNEL_SLUG", "avidok/vco-worker")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge()
    stage_dir, task_id = bridge.stage_execution(
        text="Testing audio staging manifest creation.",
        ref_audio_path=dummy_reference_wav,
        ref_text="Reference voice transcript.",
        language="hi",
    )

    try:
        assert stage_dir.is_dir()

        # 1. Verify scripts.json
        scripts_file = stage_dir / "scripts.json"
        assert scripts_file.is_file()
        with open(scripts_file, "r", encoding="utf-8") as f:
            scripts_data = json.load(f)

        assert "tasks" in scripts_data
        tasks = scripts_data["tasks"]
        assert len(tasks) == 1
        task = tasks[0]
        assert task["id"] == task_id
        assert task["text"] == "Testing audio staging manifest creation."
        assert task["ref_text"] == "Reference voice transcript."
        assert task["language"] == "hi"

        # 2. Verify kernel-metadata.json
        metadata_file = stage_dir / "kernel-metadata.json"
        assert metadata_file.is_file()
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["id"] == "avidok/vco-worker"
        assert meta["code_file"] in ("run.ipynb", "notebook_template.ipynb")
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_gpu"] is True
        assert meta["enable_internet"] is True

        # 3. Verify notebook file exists and contains cells
        notebook_file = stage_dir / meta["code_file"]
        assert notebook_file.is_file()
        with open(notebook_file, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb
        assert len(nb["cells"]) >= 2
        # Check that scripts.json and IndicF5 references are embedded
        full_code = "".join(
            "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
        )
        assert "scripts.json" in full_code
        assert "task_id" in full_code or "tasks" in full_code

        # Verify unsupported language raises ValueError without silent coercion
        with pytest.raises(
            ValueError,
            match=r"Unsupported language 'es' for Kaggle GPU bridge\. Supported languages: \['en', 'hi', 'pa'\]",
        ):
            bridge.stage_execution(
                text="Test spanish",
                ref_audio_path=dummy_reference_wav,
                ref_text="Reference voice transcript.",
                language="es",
            )

    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def test_push_kernel_invocation_and_error_trapping(
    dummy_reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify push_kernel handles successful exit, non-zero codes, and timeouts."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge()
    stage_dir, _ = bridge.stage_execution(
        text="Sample text",
        ref_audio_path=dummy_reference_wav,
        ref_text="Sample ref",
    )

    try:
        # Success case
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Kernel pushed successfully")
            bridge.push_kernel(stage_dir)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "kernels" in cmd
            assert "push" in cmd
            assert str(stage_dir) in cmd

        # Non-zero exit case
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="403 Forbidden: Invalid credentials", stdout=""
            )
            with pytest.raises(KaggleExecutionError) as exc_info:
                bridge.push_kernel(stage_dir)
            assert "403 Forbidden" in str(exc_info.value)

        # Timeout case
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["kaggle"], timeout=10.0)):
            with pytest.raises(KaggleTimeoutError) as exc_info:
                bridge.push_kernel(stage_dir)
            assert "timed out after 10.0s" in str(exc_info.value)

    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


# ==============================================================================
# 3. Polling Lifecycle & State Transitions
# ==============================================================================


def test_poll_status_transitions_to_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify poll_status handles transition from queued -> running -> complete."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)

    # Mock sequence: queued -> running -> complete
    statuses = [
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'queued'"),
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'running'"),
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'complete'"),
    ]

    with patch("subprocess.run", side_effect=statuses) as mock_run:
        bridge.poll_status("avidok/vco-worker")
        assert mock_run.call_count == 3


def test_poll_status_traps_kernel_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify poll_status raises KaggleExecutionError when status is 'error'."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)

    statuses = [
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'running'"),
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'error' (An error occurred while executing)"),
    ]

    with patch("subprocess.run", side_effect=statuses):
        with pytest.raises(KaggleExecutionError) as exc_info:
            bridge.poll_status("avidok/vco-worker")
        assert "failure status" in str(exc_info.value).lower()
        assert "error" in str(exc_info.value).lower()


def test_poll_status_traps_kernel_cancelled_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify poll_status raises KaggleExecutionError when status is 'cancelled'."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)

    statuses = [
        MagicMock(returncode=0, stdout="avidok/vco-worker has status 'cancel'"),
    ]

    with patch("subprocess.run", side_effect=statuses):
        with pytest.raises(KaggleExecutionError) as exc_info:
            bridge.poll_status("avidok/vco-worker")
        assert "cancelled" in str(exc_info.value).lower()


def test_poll_status_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify poll_status enforces timeout when kernel runs longer than configured limit."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    # Short timeout
    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=0.05)

    def slow_status(*args, **kwargs):
        time.sleep(0.06)
        return MagicMock(returncode=0, stdout="has status 'running'")

    with patch("subprocess.run", side_effect=slow_status):
        with pytest.raises(KaggleTimeoutError) as exc_info:
            bridge.poll_status("avidok/vco-worker")
        assert "timed out" in str(exc_info.value).lower()


# ==============================================================================
# 4. Output Download and WAV Persistence
# ==============================================================================


def test_pull_output_command_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pull_output invokes kaggle kernels output with destination path."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge()
    dest = tmp_path / "download"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Files written to /tmp")
        result = bridge.pull_output("avidok/vco-worker", dest)
        assert result == dest
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "kernels" in cmd
        assert "output" in cmd
        assert "avidok/vco-worker" in cmd
        assert "-p" in cmd
        assert str(dest) in cmd


def test_synthesize_retrieves_and_persists_wav(
    dummy_reference_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify synthesize downloads rendered audio from outputs/ subdir, saves to data/audio, and returns valid waveform."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)
    bridge.data_audio_dir = tmp_path / "audio"

    mock_uuid = MagicMock()
    mock_uuid.hex = "testwav0123456"
    expected_task_id = f"task_{mock_uuid.hex[:10]}"

    # Mock output to test outputs/ subdirectory handling
    def fake_pull_output(kernel_id: str, dest: Path) -> Path:
        outputs_sub = dest / "outputs"
        outputs_sub.mkdir(parents=True, exist_ok=True)
        _create_test_wav(outputs_sub / f"{expected_task_id}.wav", duration_sec=0.5, sample_rate=24000)
        return dest

    with (
        patch.object(bridge, "push_kernel") as mock_push,
        patch.object(bridge, "poll_status") as mock_poll,
        patch.object(bridge, "pull_output", side_effect=fake_pull_output) as mock_pull,
        patch("uuid.uuid4", return_value=mock_uuid),
    ):
        waveform, sr = bridge.synthesize(
            text="Testing audio retrieval.",
            ref_audio_path=dummy_reference_wav,
            ref_text="Reference transcript.",
            language="hi",
        )

        assert mock_push.called
        assert mock_poll.called
        assert mock_pull.called
        assert sr == 24000
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) > 0

        # Verify file persisted strictly as {task_id}.wav in bridge.data_audio_dir
        target_file = bridge.data_audio_dir / f"{expected_task_id}.wav"
        assert target_file.is_file()
        assert target_file.stat().st_size > 44


def test_synthesize_rejects_missing_task_id_wav(
    dummy_reference_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify synthesize rejects kernel output containing random wav files when {task_id}.wav is absent."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)
    bridge.data_audio_dir = tmp_path / "audio"

    expected_task_id = "task_missing01"
    mock_uuid = MagicMock()
    mock_uuid.hex = "missing0123456"

    # Write an unrelated wav file
    def fake_pull_unrelated_output(kernel_id: str, dest: Path) -> Path:
        outputs_sub = dest / "outputs"
        outputs_sub.mkdir(parents=True, exist_ok=True)
        _create_test_wav(outputs_sub / "unrelated_other_task.wav", duration_sec=0.2, sample_rate=24000)
        return dest

    with (
        patch.object(bridge, "push_kernel"),
        patch.object(bridge, "poll_status"),
        patch.object(bridge, "pull_output", side_effect=fake_pull_unrelated_output),
        patch("uuid.uuid4", return_value=mock_uuid),
    ):
        with pytest.raises(KaggleExecutionError) as exc_info:
            bridge.synthesize(
                text="Sample text",
                ref_audio_path=dummy_reference_wav,
                ref_text="Sample ref",
            )
        assert "not found in Kaggle kernel output directory" in str(exc_info.value)
        assert expected_task_id in str(exc_info.value)


def test_process_isolation_does_not_mutate_global_environ(
    dummy_reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify subprocess.run receives isolated credentials env without mutating global os.environ."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    settings = Settings(kaggle_username="isolated_user", kaggle_key="isolated_key")
    bridge = KaggleExecutionBridge(settings=settings)

    # Validate global os.environ does not have credentials
    assert "KAGGLE_USERNAME" not in os.environ
    assert "KAGGLE_KEY" not in os.environ

    bridge.validate_credentials()
    assert "KAGGLE_USERNAME" not in os.environ
    assert "KAGGLE_KEY" not in os.environ

    # Verify push_kernel passes isolated env to subprocess.run
    stage_dir, _ = bridge.stage_execution(
        text="Testing isolation",
        ref_audio_path=dummy_reference_wav,
        ref_text="Ref text",
    )
    try:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Success")
            bridge.push_kernel(stage_dir)
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert "env" in call_kwargs
            passed_env = call_kwargs["env"]
            assert passed_env["KAGGLE_USERNAME"] == "isolated_user"
            assert passed_env["KAGGLE_KEY"] == "isolated_key"
            # Global env still unmutated
            assert "KAGGLE_USERNAME" not in os.environ
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def test_synthesize_traps_manifest_task_error(
    dummy_reference_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify synthesize raises KaggleExecutionError if manifest_output.json reports task failure."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    bridge = KaggleExecutionBridge(poll_interval=0.01, timeout=5.0)

    expected_task_id = "task_mocktask01"
    mock_uuid = MagicMock()
    mock_uuid.hex = "mocktask0123456"

    def fake_pull_failed_output(kernel_id: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        # Write failure manifest
        with open(dest / "manifest_output.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tasks": [
                        {
                            "id": expected_task_id,
                            "status": "failed",
                            "error": "CUDA out of memory in IndicF5 inference",
                        }
                    ]
                },
                f,
            )
        return dest

    with (
        patch.object(bridge, "push_kernel"),
        patch.object(bridge, "poll_status"),
        patch.object(bridge, "pull_output", side_effect=fake_pull_failed_output),
        patch("uuid.uuid4", return_value=mock_uuid),
    ):
        with pytest.raises(KaggleExecutionError) as exc_info:
            bridge.synthesize(
                text="Sample text",
                ref_audio_path=dummy_reference_wav,
                ref_text="Sample ref",
            )
        assert "CUDA out of memory" in str(exc_info.value)


# ==============================================================================
# 5. End-to-End API Synthesis & Error Responses
# ==============================================================================


def test_api_generate_e2e_successful_synthesis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /api/generate flows through Kaggle bridge and returns 200 WAV binary."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    # Create dummy waveform for synthesize mock
    t = np.linspace(0, 0.3, 7200, endpoint=False)
    synthetic_wave = (0.25 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

    with patch.object(
        KaggleExecutionBridge, "synthesize", return_value=(synthetic_wave, 24000)
    ) as mock_bridge_synth:
        payload = {
            "text": "Full end to end test synthesizing real speech through Kaggle pipeline.",
            "language": "en",
            "speaker_ref_name": "narrator_english_neutral",
        }
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        content = response.content
        assert content.startswith(b"RIFF")
        assert b"WAVE" in content[:12]
        assert len(content) > 44
        mock_bridge_synth.assert_called_once()


def test_api_generate_e2e_kaggle_execution_error_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /api/generate handles KaggleExecutionError with HTTP 500 and structured manifest."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    with patch.object(
        KaggleExecutionBridge,
        "synthesize",
        side_effect=KaggleExecutionError("Remote kernel crashed with GPU OOM"),
    ):
        payload = {
            "text": "Failing speech generation request.",
            "language": "en",
            "speaker_ref_name": "narrator_english_neutral",
        }
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Kaggle execution failure" in data["detail"]
        assert "GPU OOM" in data["detail"]


def test_api_generate_e2e_kaggle_timeout_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /api/generate handles KaggleTimeoutError with HTTP 500 and structured manifest."""
    monkeypatch.setenv("KAGGLE_USERNAME", "avidok")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
    get_settings.cache_clear()

    with patch.object(
        KaggleExecutionBridge,
        "synthesize",
        side_effect=KaggleTimeoutError("Kaggle kernel execution timed out after 600.0s"),
    ):
        payload = {
            "text": "Timeout speech generation request.",
            "language": "en",
            "speaker_ref_name": "narrator_english_neutral",
        }
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Kaggle execution failure" in data["detail"]
        assert "timed out after 600.0s" in data["detail"]


# ==============================================================================
# 6. Zero-Mock Invariant
# ==============================================================================


def test_zero_mock_invariant_never_falls_back_to_sine_wave(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that under failure conditions, no silent fallback to synthetic 440Hz sine wave occurs."""
    # Ensure credentials missing
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    get_settings.cache_clear()

    payload = {
        "text": "This request must fail explicitly and not produce a sine wave.",
        "language": "en",
        "speaker_ref_name": "narrator_english_neutral",
    }
    response = client.post("/api/generate", json=payload)

    # Must return 422, NEVER 200
    assert response.status_code == 422
    assert response.headers.get("content-type") != "audio/wav"

    # Now simulate failed engine with credentials present
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    get_settings.cache_clear()

    with patch.object(
        KaggleExecutionBridge,
        "synthesize",
        side_effect=RuntimeError("Hard engine failure"),
    ):
        response = client.post("/api/generate", json=payload)
        # Must return 500, NEVER 200
        assert response.status_code == 500
        assert response.headers.get("content-type") != "audio/wav"
