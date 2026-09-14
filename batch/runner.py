"""Papermill batch runner and headless Kaggle automation for AudioGen."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nbformat

from batch.manifest_schema import BatchJob, load_manifest_file

logger = logging.getLogger(__name__)


class BatchRunnerError(Exception):
    """Base exception for batch runner errors."""

    pass


class KaggleExecutionError(BatchRunnerError):
    """Raised when Kaggle CLI commands fail or kernel returns error status."""

    pass


class KaggleTimeoutError(BatchRunnerError):
    """Raised when Kaggle kernel execution or CLI commands exceed configured timeout."""

    pass


def execute_notebook_locally(
    notebook_path: Union[str, Path],
    parameters: Dict[str, Any],
    output_notebook_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute notebook locally with parameter injection, respecting the parameters cell.

    Args:
        notebook_path: Path to the .ipynb notebook template.
        parameters: Dictionary of variable names and values to inject.
        output_notebook_path: Optional path to save the executed notebook.

    Returns:
        Dictionary containing execution summary.
    """
    nb_file = Path(notebook_path).resolve()
    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook file does not exist: {nb_file}")

    with open(nb_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Prepare injected execution namespace
    exec_namespace: Dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(nb_file),
    }
    for key, value in parameters.items():
        exec_namespace[key] = value

    executed_count = 0
    clean_exit = False

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        tags = cell.metadata.get("tags", [])
        if "parameters" in tags:
            # Execute default parameters cell then override with injected parameters
            try:
                exec(cell.source, exec_namespace)
            except Exception as exc:
                raise BatchRunnerError(
                    f"Failed executing parameters cell (index {cell_idx}): {exc}"
                ) from exc
            for key, value in parameters.items():
                exec_namespace[key] = value
            executed_count += 1
            continue

        try:
            exec(cell.source, exec_namespace)
            executed_count += 1
        except SystemExit as exc:
            if exc.code in (0, None):
                clean_exit = True
                executed_count += 1
                break
            raise BatchRunnerError(
                f"Notebook cell (index {cell_idx}) exited with non-zero status: {exc.code}"
            ) from exc
        except Exception as exc:
            raise BatchRunnerError(
                f"Notebook cell execution failed at index {cell_idx}: {exc}"
            ) from exc

    if output_notebook_path is not None:
        out_path = Path(output_notebook_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    output_dir = parameters.get("OUTPUT_DIR")
    if output_dir:
        manifest_out = Path(output_dir) / "manifest_output.json"
        if manifest_out.exists():
            with open(manifest_out, "r", encoding="utf-8") as f:
                return json.load(f)

    return {
        "status": "success",
        "clean_exit": clean_exit,
        "executed_cells": executed_count,
    }


def run_local(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    model_weights_dir: Optional[Union[str, Path]] = None,
    notebook_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute batch synthesis locally using the parameter-injected notebook.

    Args:
        manifest_path: Path to JSON manifest file.
        output_dir: Destination directory for audio files and manifest_output.json.
        model_weights_dir: Optional path to model weights checkpoint directory.
        notebook_path: Optional path override for notebook template.

    Returns:
        Summary dictionary loaded from manifest_output.json.
    """
    manifest_file = Path(manifest_path).resolve()
    # Validate manifest format and entries before proceeding
    load_manifest_file(manifest_file)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if notebook_path is not None:
        nb_path = Path(notebook_path).resolve()
    else:
        nb_path = Path(__file__).resolve().parent / "notebook_template.ipynb"

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook template not found: {nb_path}")

    weights_dir = str(Path(model_weights_dir).resolve()) if model_weights_dir else ""

    parameters = {
        "MANIFEST_PATH": str(manifest_file),
        "MODEL_WEIGHTS_DIR": weights_dir,
        "OUTPUT_DIR": str(out_dir),
    }

    execute_notebook_locally(
        notebook_path=nb_path,
        parameters=parameters,
    )

    manifest_output_file = out_dir / "manifest_output.json"
    if not manifest_output_file.exists():
        raise BatchRunnerError(
            f"Local batch execution finished but {manifest_output_file} was not generated."
        )

    with open(manifest_output_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_kaggle(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    model_weights_dir: Optional[Union[str, Path]] = None,
    notebook_path: Optional[Union[str, Path]] = None,
    kernel_metadata_path: Optional[Union[str, Path]] = None,
    staging_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
    poll_interval: float = 10.0,
    kaggle_cmd: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Stage, push, poll, and pull output for headless Kaggle batch synthesis.

    Args:
        manifest_path: Path to JSON manifest file.
        output_dir: Destination directory to store pulled kernel outputs.
        model_weights_dir: Optional model weights location.
        notebook_path: Optional path to notebook template.
        kernel_metadata_path: Optional path to kernel-metadata.json template.
        staging_dir: Optional directory for staging files. If None, uses a temp directory.
        timeout: Maximum execution timeout in seconds.
        poll_interval: Seconds between status polling queries.
        kaggle_cmd: Base command list for Kaggle CLI (defaults to ['kaggle']).

    Returns:
        Summary dictionary parsed from manifest_output.json, or execution status.
    """
    manifest_file = Path(manifest_path).resolve()
    # Fail-fast manifest schema validation
    load_manifest_file(manifest_file)

    cleanup_staging = False
    if staging_dir is not None:
        stage_dir = Path(staging_dir).resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
    else:
        stage_temp = tempfile.mkdtemp(prefix="audiogen_kaggle_stage_")
        stage_dir = Path(stage_temp).resolve()
        cleanup_staging = True

    try:
        # Resolve notebook and metadata templates
        if notebook_path is not None:
            source_nb = Path(notebook_path).resolve()
        else:
            source_nb = Path(__file__).resolve().parent / "notebook_template.ipynb"

        if not source_nb.exists():
            raise FileNotFoundError(f"Notebook template not found: {source_nb}")

        if kernel_metadata_path is not None:
            source_meta = Path(kernel_metadata_path).resolve()
        else:
            source_meta = (
                Path(__file__).resolve().parents[1] / "config" / "kaggle_kernel_metadata.json"
            )

        if not source_meta.exists():
            raise FileNotFoundError(f"Kernel metadata template not found: {source_meta}")

        with open(source_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        kernel_id = meta_data.get("id")
        if not kernel_id or "/" not in kernel_id:
            raise ValueError(f"Invalid or missing kernel id in metadata: {kernel_id}")

        # Stage files and inject manifest data directly into the notebook via nbformat
        with open(source_nb, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        manifest_json_str = json.dumps(manifest_data, indent=2)
        staging_cell_source = (
            "# Injected by run_kaggle to stage manifest data directly in remote kernel\n"
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

        staged_nb = stage_dir / source_nb.name
        with open(staged_nb, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        staged_manifest = stage_dir / "scripts.json"
        shutil.copy2(manifest_file, staged_manifest)

        meta_data["code_file"] = source_nb.name
        staged_meta = stage_dir / "kernel-metadata.json"
        with open(staged_meta, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        k_cmd = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]

        # Step 1: Push kernel to Kaggle
        push_cmd = [*k_cmd, "kernels", "push", "-p", str(stage_dir)]
        logger.info("Executing Kaggle push: %s", " ".join(push_cmd))

        try:
            push_proc = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                timeout=min(60.0, timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleTimeoutError(
                f"Kaggle push command timed out after {exc.timeout} seconds."
            ) from exc
        except Exception as exc:
            raise KaggleExecutionError(f"Failed to execute Kaggle push command: {exc}") from exc

        if push_proc.returncode != 0:
            err_msg = (push_proc.stderr or push_proc.stdout or "").strip()
            raise KaggleExecutionError(
                f"Kaggle push failed with exit code {push_proc.returncode}: {err_msg}"
            )

        # Step 2: Poll status with fail-fast timeouts and error trapping
        status_cmd = [*k_cmd, "kernels", "status", kernel_id]
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise KaggleTimeoutError(
                    f"Kaggle kernel execution timed out after {elapsed:.1f}s (configured limit: {timeout}s)"
                )

            remaining_timeout = max(5.0, min(30.0, timeout - elapsed))
            try:
                status_proc = subprocess.run(
                    status_cmd,
                    capture_output=True,
                    text=True,
                    timeout=remaining_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise KaggleTimeoutError(
                    f"Kaggle status check command hung/timed out: {exc}"
                ) from exc
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise KaggleExecutionError(
                        f"Kaggle status command failed {consecutive_errors} consecutive times: {exc}"
                    ) from exc
                time.sleep(poll_interval)
                continue

            if status_proc.returncode != 0:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    err_msg = (status_proc.stderr or status_proc.stdout or "").strip()
                    raise KaggleExecutionError(
                        f"Kaggle status check failed with code {status_proc.returncode}: {err_msg}"
                    )
                time.sleep(poll_interval)
                continue

            consecutive_errors = 0
            status_text = (status_proc.stdout or "").strip()
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

            # Kernel is queued or running; wait before next poll
            time.sleep(poll_interval)

        # Step 3: Pull outputs
        destination_dir = Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_cmd = [*k_cmd, "kernels", "output", kernel_id, "-p", str(destination_dir), "--force"]

        logger.info("Pulling Kaggle kernel output: %s", " ".join(output_cmd))
        try:
            output_proc = subprocess.run(
                output_cmd,
                capture_output=True,
                text=True,
                timeout=min(120.0, timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleTimeoutError(
                f"Kaggle output pull command timed out after {exc.timeout} seconds."
            ) from exc
        except Exception as exc:
            raise KaggleExecutionError(f"Failed to pull Kaggle kernel output: {exc}") from exc

        if output_proc.returncode != 0:
            err_msg = (output_proc.stderr or output_proc.stdout or "").strip()
            raise KaggleExecutionError(
                f"Kaggle output pull failed with code {output_proc.returncode}: {err_msg}"
            )

        pulled_manifest = destination_dir / "manifest_output.json"
        if pulled_manifest.exists():
            with open(pulled_manifest, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "status": "complete",
            "kernel_id": kernel_id,
            "output_dir": str(destination_dir),
        }

    finally:
        if cleanup_staging and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def main(args: Optional[List[str]] = None) -> None:
    """CLI entrypoint for batch runner."""
    parser = argparse.ArgumentParser(
        description="AudioGen Batch Runner: local or Kaggle execution."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest containing reel scripts.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "kaggle"],
        default="local",
        help="Execution mode: local (parameter-injected notebook) or kaggle (headless remote).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where audio files and manifest_output.json will be saved.",
    )
    parser.add_argument(
        "--model-weights-dir",
        default=None,
        help="Path to model weights directory.",
    )
    parser.add_argument(
        "--notebook-path",
        default=None,
        help="Path to custom notebook template.",
    )
    parser.add_argument(
        "--kernel-metadata",
        default=None,
        help="Path to custom Kaggle kernel metadata JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for remote execution / commands.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Polling interval in seconds for Kaggle status checks.",
    )

    parsed = parser.parse_args(args)

    if parsed.mode == "local":
        result = run_local(
            manifest_path=parsed.manifest,
            output_dir=parsed.output_dir,
            model_weights_dir=parsed.model_weights_dir,
            notebook_path=parsed.notebook_path,
        )
    elif parsed.mode == "kaggle":
        result = run_kaggle(
            manifest_path=parsed.manifest,
            output_dir=parsed.output_dir,
            model_weights_dir=parsed.model_weights_dir,
            notebook_path=parsed.notebook_path,
            kernel_metadata_path=parsed.kernel_metadata,
            timeout=parsed.timeout,
            poll_interval=parsed.poll_interval,
        )
    else:
        parser.error(f"Unknown mode: {parsed.mode}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
