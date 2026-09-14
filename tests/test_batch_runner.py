"""Unit test suite for batch runner, local notebook execution, and headless Kaggle automation (Ticket 003)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
import nbformat

from batch.runner import (
    BatchRunnerError,
    KaggleExecutionError,
    KaggleTimeoutError,
    execute_notebook_locally,
    main,
    run_kaggle,
    run_local,
)
from audiogen.engine import Synthesizer
from voices.registry import clear_registry_cache


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean registry cache for all tests."""
    clear_registry_cache()
    yield
    clear_registry_cache()


@pytest.fixture
def notebook_path() -> Path:
    """Return path to batch/notebook_template.ipynb."""
    path = Path(__file__).resolve().parents[1] / "batch" / "notebook_template.ipynb"
    assert path.is_file(), f"Notebook template missing at {path}"
    return path


@pytest.fixture
def sample_manifest(tmp_path: Path) -> Path:
    """Create a valid sample 2-task manifest."""
    manifest_file = tmp_path / "sample_manifest.json"
    tasks = [
        {
            "id": "reel_01",
            "text": "नमस्ते, यह पहला परीक्षण संदेश है।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
        {
            "id": "reel_02",
            "text": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਇਹ ਦੂਜਾ ਟੈਸਟ ਸੁਨੇਹਾ ਹੈ।",
            "language": "pa",
            "voice_ref": "storyteller_punjabi_elder",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")
    return manifest_file


def test_notebook_parameters_cell_exact_variables(notebook_path: Path):
    """Verify parameters cell in notebook_template.ipynb exposes EXACTLY MANIFEST_PATH, MODEL_WEIGHTS_DIR, OUTPUT_DIR."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    parameter_cells = [
        cell
        for cell in nb.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1, "Expected exactly one cell tagged 'parameters'"

    param_cell = parameter_cells[0]
    tree = ast.parse(param_cell.source)

    assigned_vars = []
    for node in tree.body:
        assert isinstance(node, ast.Assign), "All statements in parameters cell should be assignments"
        for target in node.targets:
            assert isinstance(target, ast.Name), "Assignment targets must be variable names"
            assigned_vars.append(target.id)

    assert sorted(assigned_vars) == ["MANIFEST_PATH", "MODEL_WEIGHTS_DIR", "OUTPUT_DIR"], (
        f"Parameters cell must expose exactly MANIFEST_PATH, MODEL_WEIGHTS_DIR, OUTPUT_DIR, got: {assigned_vars}"
    )


def test_notebook_final_cell_sys_exit(notebook_path: Path):
    """Verify the final cell of the notebook cleanly terminates with sys.exit(0)."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(code_cells) > 0, "Notebook has no code cells"
    last_cell = code_cells[-1]

    assert "sys.exit(0)" in last_cell.source.strip(), (
        f"Final code cell must execute sys.exit(0), got: {last_cell.source}"
    )


def test_run_local_success(notebook_path: Path, sample_manifest: Path, tmp_path: Path):
    """Verify end-to-end local batch run produces valid WAV files and manifest_output.json."""
    output_dir = tmp_path / "local_outputs"

    summary = run_local(
        manifest_path=sample_manifest,
        output_dir=output_dir,
        model_weights_dir=Path.cwd(),
        notebook_path=notebook_path,
    )

    assert summary["total"] == 2
    assert summary["successful"] == 2
    assert summary["failed"] == 0
    assert len(summary["tasks"]) == 2

    # Check that audio files were generated on disk
    wav1 = output_dir / "reel_01.wav"
    wav2 = output_dir / "reel_02.wav"
    assert wav1.is_file()
    assert wav2.is_file()
    assert wav1.stat().st_size > 0
    assert wav2.stat().st_size > 0

    # Check manifest_output.json exists and matches
    manifest_out = output_dir / "manifest_output.json"
    assert manifest_out.is_file()
    with open(manifest_out, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == summary


def test_per_item_failure_continuation(notebook_path: Path, tmp_path: Path, monkeypatch):
    """Explicitly test fault-tolerant processing: 3 tasks where task 2 fails, tasks 1 and 3 succeed."""
    manifest_file = tmp_path / "three_tasks.json"
    tasks = [
        {
            "id": "task_1",
            "text": "पहला कार्य जो सफल होगा।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
        {
            "id": "task_2",
            "text": "दूसरा कार्य जो कृत्रिम विफलता उत्पन्न करेगा।",
            "language": "hi",
            "voice_ref": "anchor_female_calm",
        },
        {
            "id": "task_3",
            "text": "ਤੀਜਾ ਕੰਮ ਜੋ ਸਫਲਤਾਪੂਰਵਕ ਮੁਕੰਮਲ ਹੋਵੇਗਾ।",
            "language": "pa",
            "voice_ref": "storyteller_punjabi_elder",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    real_synthesize = Synthesizer.synthesize

    def mock_synthesize(self, text: str, language: str, speaker_ref: Any = None):
        if "विफलता" in text or "task_2" in text:
            raise RuntimeError("Simulated acoustic synthesis failure on task 2")
        return real_synthesize(self, text, language, speaker_ref)

    monkeypatch.setattr(Synthesizer, "synthesize", mock_synthesize)

    output_dir = tmp_path / "fault_tolerant_outputs"
    summary = run_local(
        manifest_path=manifest_file,
        output_dir=output_dir,
        model_weights_dir=Path.cwd(),
        notebook_path=notebook_path,
    )

    assert summary["total"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1

    # Task 1: succeeded
    t1 = summary["tasks"][0]
    assert t1["id"] == "task_1"
    assert t1["status"] == "success"
    assert (output_dir / "task_1.wav").is_file()

    # Task 2: failed
    t2 = summary["tasks"][1]
    assert t2["id"] == "task_2"
    assert t2["status"] == "failed"
    assert "Simulated acoustic synthesis failure" in t2["error"]
    assert not (output_dir / "task_2.wav").exists()

    # Task 3: succeeded
    t3 = summary["tasks"][2]
    assert t3["id"] == "task_3"
    assert t3["status"] == "success"
    assert (output_dir / "task_3.wav").is_file()

    # Check manifest_output.json recorded all three
    manifest_out = output_dir / "manifest_output.json"
    assert manifest_out.is_file()
    with open(manifest_out, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert len(on_disk["tasks"]) == 3
    assert on_disk["tasks"][1]["status"] == "failed"


def test_run_kaggle_success_flow(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle automation workflow: push -> poll queued/running/complete -> pull output."""
    output_dir = tmp_path / "kaggle_outputs"
    staging_dir = tmp_path / "kaggle_staging"

    status_calls = 0

    def fake_subprocess_run(cmd, *args, **kwargs):
        nonlocal status_calls
        cmd_str = " ".join(cmd)

        if "kernels push" in cmd_str:
            assert (staging_dir / "kernel-metadata.json").is_file()
            assert (staging_dir / "scripts.json").is_file()
            staged_nb_path = staging_dir / "notebook_template.ipynb"
            assert staged_nb_path.is_file()
            with open(staged_nb_path, "r", encoding="utf-8") as f:
                staged_nb = nbformat.read(f, as_version=4)
            # Verify manifest delivery is injected directly into notebook cells
            staging_cells = [
                c for c in staged_nb.cells
                if "injected-manifest-staging" in c.metadata.get("tags", [])
                or "STAGED_MANIFEST_DATA" in c.source
            ]
            assert len(staging_cells) == 1, "Expected injected manifest staging cell in staged notebook"
            assert "reel_01" in staging_cells[0].source
            return subprocess.CompletedProcess(cmd, 0, stdout="Kernel version 1 pushed.", stderr="")

        elif "kernels status" in cmd_str:
            status_calls += 1
            if status_calls == 1:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="username/audiogen-batch-synthesis has status 'queued'", stderr=""
                )
            elif status_calls == 2:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="username/audiogen-batch-synthesis has status 'running'", stderr=""
                )
            else:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="username/audiogen-batch-synthesis has status 'complete'", stderr=""
                )

        elif "kernels output" in cmd_str:
            # Simulate downloaded output files
            output_dir.mkdir(parents=True, exist_ok=True)
            mock_out = {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "tasks": [{"id": "reel_01", "status": "success"}, {"id": "reel_02", "status": "success"}],
            }
            (output_dir / "manifest_output.json").write_text(json.dumps(mock_out), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="Output pulled successfully.", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        result = run_kaggle(
            manifest_path=sample_manifest,
            output_dir=output_dir,
            staging_dir=staging_dir,
            poll_interval=0.01,
            timeout=30.0,
        )

    assert status_calls == 3
    assert result["total"] == 2
    assert result["successful"] == 2
    assert result["failed"] == 0


def test_run_kaggle_timeout(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle timeout triggers KaggleTimeoutError without infinite loop."""
    output_dir = tmp_path / "kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        if "kernels push" in cmd_str:
            return subprocess.CompletedProcess(cmd, 0, stdout="Pushed.", stderr="")
        elif "kernels status" in cmd_str:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="username/audiogen-batch-synthesis has status 'running'", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        with pytest.raises(KaggleTimeoutError) as exc_info:
            run_kaggle(
                manifest_path=sample_manifest,
                output_dir=output_dir,
                poll_interval=0.01,
                timeout=0.05,
            )
        assert "timed out" in str(exc_info.value).lower()


def test_run_kaggle_cli_hang(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle CLI hanging raises KaggleTimeoutError."""
    output_dir = tmp_path / "kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        if "kernels push" in cmd_str:
            return subprocess.CompletedProcess(cmd, 0, stdout="Pushed.", stderr="")
        elif "kernels status" in cmd_str:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        with pytest.raises(KaggleTimeoutError) as exc_info:
            run_kaggle(
                manifest_path=sample_manifest,
                output_dir=output_dir,
                poll_interval=0.01,
                timeout=30.0,
            )
        assert "timed out" in str(exc_info.value).lower() or "hung" in str(exc_info.value).lower()


def test_run_kaggle_failure_status(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle kernel returning 'error' status raises KaggleExecutionError."""
    output_dir = tmp_path / "kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        if "kernels push" in cmd_str:
            return subprocess.CompletedProcess(cmd, 0, stdout="Pushed.", stderr="")
        elif "kernels status" in cmd_str:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="username/audiogen-batch-synthesis has status 'error'", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        with pytest.raises(KaggleExecutionError) as exc_info:
            run_kaggle(
                manifest_path=sample_manifest,
                output_dir=output_dir,
                poll_interval=0.01,
                timeout=30.0,
            )
        assert "terminated with failure status" in str(exc_info.value)


def test_run_kaggle_push_failure(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle push returning non-zero code raises KaggleExecutionError."""
    output_dir = tmp_path / "kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="403 Forbidden")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(KaggleExecutionError) as exc_info:
            run_kaggle(
                manifest_path=sample_manifest,
                output_dir=output_dir,
                timeout=30.0,
            )
        assert "Kaggle push failed" in str(exc_info.value)


def test_run_kaggle_output_pull_failure(sample_manifest: Path, tmp_path: Path):
    """Test Kaggle output pull returning non-zero code raises KaggleExecutionError."""
    output_dir = tmp_path / "kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        if "kernels push" in cmd_str:
            return subprocess.CompletedProcess(cmd, 0, stdout="Pushed.", stderr="")
        elif "kernels status" in cmd_str:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="username/audiogen-batch-synthesis has status 'complete'", stderr=""
            )
        elif "kernels output" in cmd_str:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Failed to download outputs")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        with pytest.raises(KaggleExecutionError) as exc_info:
            run_kaggle(
                manifest_path=sample_manifest,
                output_dir=output_dir,
                poll_interval=0.01,
                timeout=30.0,
            )
        assert "Kaggle output pull failed" in str(exc_info.value)


def test_cli_invocation(sample_manifest: Path, tmp_path: Path, capsys):
    """Test CLI main() invocation for local mode."""
    output_dir = tmp_path / "cli_outputs"
    cli_args = [
        "--manifest",
        str(sample_manifest),
        "--mode",
        "local",
        "--output-dir",
        str(output_dir),
        "--model-weights-dir",
        str(Path.cwd()),
    ]
    main(cli_args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["total"] == 2
    assert result["successful"] == 2


def test_cli_invocation_kaggle(sample_manifest: Path, tmp_path: Path, capsys):
    """Test CLI main() invocation for kaggle mode with mocked execution."""
    output_dir = tmp_path / "cli_kaggle_outputs"

    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        if "kernels push" in cmd_str:
            return subprocess.CompletedProcess(cmd, 0, stdout="Pushed.", stderr="")
        elif "kernels status" in cmd_str:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="username/audiogen-batch-synthesis has status 'complete'", stderr=""
            )
        elif "kernels output" in cmd_str:
            output_dir.mkdir(parents=True, exist_ok=True)
            mock_out = {"total": 2, "successful": 2, "failed": 0, "tasks": []}
            (output_dir / "manifest_output.json").write_text(json.dumps(mock_out), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="Pulled.", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    cli_args = [
        "--manifest",
        str(sample_manifest),
        "--mode",
        "kaggle",
        "--output-dir",
        str(output_dir),
        "--timeout",
        "30.0",
        "--poll-interval",
        "0.01",
    ]
    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("time.sleep", return_value=None):
        main(cli_args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["total"] == 2
    assert result["successful"] == 2

