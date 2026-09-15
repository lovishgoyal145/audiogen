"""Unit test suite for batch runner, local notebook execution, and headless Kaggle automation (Ticket 003)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional
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
from voices.registry import VoiceNotFoundError, clear_registry_cache, get_voice_ref


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

    def mock_synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: str = "",
        *args: Any,
        **kwargs: Any,
    ):
        if "विफलता" in text or "task_2" in text:
            raise RuntimeError("Simulated acoustic synthesis failure on task 2")
        return real_synthesize(
            self,
            text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            *args,
            **kwargs,
        )

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


def test_batch_synthesizer_call_args_indicf5(
    notebook_path: Path, sample_manifest: Path, tmp_path: Path, monkeypatch
):
    """Verify Synthesizer.synthesize is called with actual IndicF5 signature arguments (AC1)."""
    real_synthesize = Synthesizer.synthesize
    mock_synth = MagicMock()

    def spy_synthesize(self, text: str, *args: Any, **kwargs: Any):
        mock_synth(text, *args, **kwargs)
        return real_synthesize(self, text, *args, **kwargs)

    monkeypatch.setattr(Synthesizer, "synthesize", spy_synthesize)

    output_dir = tmp_path / "call_args_output"
    summary = run_local(
        manifest_path=sample_manifest,
        output_dir=output_dir,
        model_weights_dir=Path.cwd(),
        notebook_path=notebook_path,
    )

    assert summary["successful"] == 2
    assert mock_synth.call_count == 2

    # Verify call args for task 1 (anchor_male_energetic)
    task1_voice = get_voice_ref("anchor_male_energetic")
    call_args_1 = mock_synth.call_args_list[0]
    pos_args_1, kw_args_1 = call_args_1
    assert pos_args_1[0] == "नमस्ते, यह पहला परीक्षण संदेश है।"
    assert kw_args_1.get("ref_audio_path") == task1_voice.path
    assert kw_args_1.get("ref_text") == task1_voice.ref_text

    # Verify call args for task 2 (storyteller_punjabi_elder)
    task2_voice = get_voice_ref("storyteller_punjabi_elder")
    call_args_2 = mock_synth.call_args_list[1]
    pos_args_2, kw_args_2 = call_args_2
    assert pos_args_2[0] == "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਇਹ ਦੂਜਾ ਟੈਸਟ ਸੁਨੇਹਾ ਹੈ।"
    assert kw_args_2.get("ref_audio_path") == task2_voice.path
    assert kw_args_2.get("ref_text") == task2_voice.ref_text


def test_batch_language_mismatch_fails_and_continues(notebook_path: Path, tmp_path: Path):
    """Verify task with language incompatible with resolved voice fails without aborting batch (AC3)."""
    manifest_file = tmp_path / "lang_mismatch_manifest.json"
    tasks = [
        {
            "id": "task_mismatch",
            "text": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਇਹ ਇਕ ਟੈਸਟ ਹੈ।",
            "language": "pa",
            "voice_ref": "anchor_female_calm",  # Only supports 'hi'
        },
        {
            "id": "task_valid",
            "text": "नमस्ते, यह एक वैध परीक्षण कार्य है।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    output_dir = tmp_path / "lang_mismatch_output"
    summary = run_local(
        manifest_path=manifest_file,
        output_dir=output_dir,
        model_weights_dir=Path.cwd(),
        notebook_path=notebook_path,
    )

    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1

    # Task 1 failed due to language incompatibility
    t1 = summary["tasks"][0]
    assert t1["id"] == "task_mismatch"
    assert t1["status"] == "failed"
    assert "does not support language 'pa'" in t1["error"]
    assert not (output_dir / "task_mismatch.wav").exists()

    # Task 2 succeeded
    t2 = summary["tasks"][1]
    assert t2["id"] == "task_valid"
    assert t2["status"] == "success"
    assert (output_dir / "task_valid.wav").is_file()


def test_batch_unknown_voice_ref_fails_and_continues(notebook_path: Path, tmp_path: Path):
    """Verify task with unknown voice_ref records failed in manifest_output.json and batch continues (AC2)."""
    manifest_file = tmp_path / "unknown_voice_manifest.json"
    tasks = [
        {
            "id": "task_bad_voice",
            "text": "नमस्ते, यह अज्ञात आवाज वाला कार्य है।",
            "language": "hi",
            "voice_ref": "non_existent_ghost_voice",
        },
        {
            "id": "task_good_voice",
            "text": "नमस्ते, यह वैध कार्य है।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    output_dir = tmp_path / "unknown_voice_output"
    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": str(Path.cwd()),
        "OUTPUT_DIR": str(output_dir),
    }
    summary = execute_notebook_locally(
        notebook_path=notebook_path,
        parameters=parameters,
    )

    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1

    t1 = summary["tasks"][0]
    assert t1["id"] == "task_bad_voice"
    assert t1["status"] == "failed"
    assert "Voice resolution error" in t1["error"]
    assert "non_existent_ghost_voice" in t1["error"]
    assert not (output_dir / "task_bad_voice.wav").exists()

    t2 = summary["tasks"][1]
    assert t2["id"] == "task_good_voice"
    assert t2["status"] == "success"
    assert (output_dir / "task_good_voice.wav").is_file()


def test_batch_missing_reference_audio_file_fails_with_distinct_message(
    notebook_path: Path, tmp_path: Path, monkeypatch
):
    """Verify task with missing on-disk audio file records distinct FileNotFoundError error message (AC4)."""
    manifest_file = tmp_path / "missing_audio_manifest.json"
    tasks = [
        {
            "id": "task_missing_audio",
            "text": "पहला कार्य जिसकी संदर्भ ऑडियो फाइल मौजूद नहीं है।",
            "language": "hi",
            "voice_ref": "anchor_female_calm",
        },
        {
            "id": "task_intact_audio",
            "text": "दूसरा कार्य जो पूर्ण रूप से सफल होगा।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    import voices.registry as reg

    real_get_voice = reg.get_voice_ref

    def fake_get_voice(name: str):
        if name == "anchor_female_calm":
            raise FileNotFoundError(
                "Reference audio file missing for voice 'anchor_female_calm': /nonexistent/path/to/anchor_female_calm.wav"
            )
        return real_get_voice(name)

    monkeypatch.setattr(reg, "get_voice_ref", fake_get_voice)

    output_dir = tmp_path / "missing_audio_output"
    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": str(Path.cwd()),
        "OUTPUT_DIR": str(output_dir),
    }
    summary = execute_notebook_locally(
        notebook_path=notebook_path,
        parameters=parameters,
    )

    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1

    t1 = summary["tasks"][0]
    assert t1["id"] == "task_missing_audio"
    assert t1["status"] == "failed"
    # Error message must be distinct from VoiceNotFoundError: contains reference audio file not found, NOT voice resolution error
    assert "Reference audio file" in t1["error"]
    assert "not found on disk" in t1["error"]
    assert "Voice resolution error" not in t1["error"]

    t2 = summary["tasks"][1]
    assert t2["id"] == "task_intact_audio"
    assert t2["status"] == "success"
    assert (output_dir / "task_intact_audio.wav").is_file()


def test_batch_model_loaded_once_for_entire_manifest(
    notebook_path: Path, sample_manifest: Path, tmp_path: Path, monkeypatch
):
    """Verify Synthesizer is initialized exactly once for the entire batch job (AC5)."""
    real_init = Synthesizer.__init__
    init_spy = MagicMock()

    def spy_init(self, *args: Any, **kwargs: Any):
        init_spy(*args, **kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(Synthesizer, "__init__", spy_init)

    output_dir = tmp_path / "single_load_output"
    summary = run_local(
        manifest_path=sample_manifest,
        output_dir=output_dir,
        model_weights_dir=Path.cwd(),
        notebook_path=notebook_path,
    )

    assert summary["successful"] == 2
    assert init_spy.call_count == 1, (
        f"Synthesizer must be instantiated exactly once, got {init_spy.call_count} times."
    )


def test_batch_unsupported_language_code_fails_and_continues(notebook_path: Path, tmp_path: Path):
    """Verify task with unsupported language code fails without aborting batch."""
    manifest_file = tmp_path / "unsupported_lang_manifest.json"
    tasks = [
        {
            "id": "task_invalid_lang",
            "text": "Hello this is English text.",
            "language": "en",
            "voice_ref": "anchor_male_energetic",
        },
        {
            "id": "task_valid_lang",
            "text": "नमस्ते, यह हिंदी टेक्स्ट है।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    output_dir = tmp_path / "unsupported_lang_output"
    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": str(Path.cwd()),
        "OUTPUT_DIR": str(output_dir),
    }
    summary = execute_notebook_locally(
        notebook_path=notebook_path,
        parameters=parameters,
    )

    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1

    t1 = summary["tasks"][0]
    assert t1["id"] == "task_invalid_lang"
    assert t1["status"] == "failed"
    assert "Unsupported language 'en'" in t1["error"]

    t2 = summary["tasks"][1]
    assert t2["id"] == "task_valid_lang"
    assert t2["status"] == "success"
    assert (output_dir / "task_valid_lang.wav").is_file()


def test_batch_unified_mixed_outcome_manifest_isolation(
    notebook_path: Path, tmp_path: Path, monkeypatch
):
    """Verify genuine batch isolation on a single 5-task manifest exercising all 3 failure modes interleaved with success.

    Sequence:
    Task 1: Success (Hindi, anchor_male_energetic)
    Task 2: Failure 1 - Unknown Voice (non-existent registry voice)
    Task 3: Success (Punjabi, storyteller_punjabi_elder)
    Task 4: Failure 2 - Language Mismatch (Punjabi requested for Hindi-only anchor_female_calm)
    Task 5: Failure 3 - Missing Audio on Disk (FileNotFoundError)
    """
    import voices.registry as reg

    real_get_voice = reg.get_voice_ref

    def fake_get_voice(name: str):
        if name == "missing_disk_audio_voice":
            raise FileNotFoundError(
                "Reference audio file missing for voice 'missing_disk_audio_voice': /nonexistent/audio.wav"
            )
        return real_get_voice(name)

    monkeypatch.setattr(reg, "get_voice_ref", fake_get_voice)

    manifest_file = tmp_path / "unified_mixed_manifest.json"
    tasks = [
        {
            "id": "t1_success",
            "text": "नमस्ते, पहला कार्य पूर्ण रूप से सफल होगा।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
        {
            "id": "t2_unknown_voice",
            "text": "दूसरा कार्य जो अज्ञात आवाज के कारण विफल होगा।",
            "language": "hi",
            "voice_ref": "ghost_unregistered_voice",
        },
        {
            "id": "t3_success",
            "text": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਤੀਜਾ ਕੰਮ ਪੂਰੀ ਤਰ੍ਹਾਂ ਸਫਲ ਹੋਵੇਗਾ।",
            "language": "pa",
            "voice_ref": "storyteller_punjabi_elder",
        },
        {
            "id": "t4_lang_mismatch",
            "text": "ਚੌਥਾ ਕੰਮ ਜੋ ਭਾਸ਼ਾ ਬੇਮੇਲ ਹੋਣ ਕਾਰਨ ਅਸਫਲ ਹੋਵੇਗਾ।",
            "language": "pa",
            "voice_ref": "anchor_female_calm",
        },
        {
            "id": "t5_missing_audio",
            "text": "पांचवां कार्य जिसकी ऑडियो फाइल डिस्क पर नहीं मिलेगी।",
            "language": "hi",
            "voice_ref": "missing_disk_audio_voice",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    output_dir = tmp_path / "unified_mixed_output"
    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": str(Path.cwd()),
        "OUTPUT_DIR": str(output_dir),
    }
    summary = execute_notebook_locally(
        notebook_path=notebook_path,
        parameters=parameters,
    )

    # 1. Verify global counts
    assert summary["total"] == 5
    assert summary["successful"] == 2
    assert summary["failed"] == 3

    # 2. Task 1: Success
    task1 = summary["tasks"][0]
    assert task1["id"] == "t1_success"
    assert task1["status"] == "success"
    assert task1["error"] is None
    assert (output_dir / "t1_success.wav").is_file()
    assert (output_dir / "t1_success.wav").stat().st_size > 0

    # 3. Task 2: Unknown Voice failure
    task2 = summary["tasks"][1]
    assert task2["id"] == "t2_unknown_voice"
    assert task2["status"] == "failed"
    assert "Voice resolution error" in task2["error"]
    assert "ghost_unregistered_voice" in task2["error"]
    assert not (output_dir / "t2_unknown_voice.wav").exists()

    # 4. Task 3: Success
    task3 = summary["tasks"][2]
    assert task3["id"] == "t3_success"
    assert task3["status"] == "success"
    assert task3["error"] is None
    assert (output_dir / "t3_success.wav").is_file()
    assert (output_dir / "t3_success.wav").stat().st_size > 0

    # 5. Task 4: Language Mismatch failure
    task4 = summary["tasks"][3]
    assert task4["id"] == "t4_lang_mismatch"
    assert task4["status"] == "failed"
    assert "does not support language 'pa'" in task4["error"]
    assert not (output_dir / "t4_lang_mismatch.wav").exists()

    # 6. Task 5: Missing Audio on Disk failure (distinguishable from unknown voice)
    task5 = summary["tasks"][4]
    assert task5["id"] == "t5_missing_audio"
    assert task5["status"] == "failed"
    assert "Reference audio file" in task5["error"]
    assert "not found on disk" in task5["error"]
    assert "Voice resolution error" not in task5["error"]
    assert not (output_dir / "t5_missing_audio.wav").exists()


def test_batch_raw_dict_task_id_directory_traversal_sanitized(
    notebook_path: Path, tmp_path: Path
):
    """Verify task_id in raw dictionaries is sanitized against directory traversal attacks."""
    manifest_file = tmp_path / "traversal_manifest.json"
    tasks = [
        {
            "id": "../../escape_attack_task",
            "text": "नमस्ते, पथ ट्रैवर्सल सुरक्षा परीक्षण।",
            "language": "hi",
            "voice_ref": "anchor_male_energetic",
        },
    ]
    manifest_file.write_text(json.dumps(tasks), encoding="utf-8")

    output_dir = tmp_path / "traversal_sandbox"
    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": str(Path.cwd()),
        "OUTPUT_DIR": str(output_dir),
    }
    summary = execute_notebook_locally(
        notebook_path=notebook_path,
        parameters=parameters,
    )

    assert summary["total"] == 1
    assert summary["successful"] == 1

    # Sanitized task_id must be "escape_attack_task" placed safely inside output_dir
    sanitized_wav = output_dir / "escape_attack_task.wav"
    assert sanitized_wav.is_file()
    # Ensure no file was created outside output_dir
    assert not (tmp_path / "escape_attack_task.wav").exists()



