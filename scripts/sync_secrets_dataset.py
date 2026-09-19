#!/usr/bin/env python3
"""Synchronize runtime secrets from .env to a private Kaggle Dataset.

Maintains the private dataset `<KAGGLE_USERNAME>/audiogen-secrets` with:
- dataset-metadata.json
- secrets.json
- Individual plaintext secret files (TUNNEL_REGISTRY_WEBHOOK_URL, TUNNEL_REGISTRY_AUTH_TOKEN, SERVER_BEARER_TOKEN)

Uses SHA-256 caching via `.secrets_dataset_hash` to avoid redundant Kaggle API version churn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

try:
    import dotenv
except ImportError:
    dotenv = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_secrets_dataset")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_FILE_NAME: str = ".secrets_dataset_hash"

REQUIRED_SECRET_KEYS: List[str] = [
    "TUNNEL_REGISTRY_WEBHOOK_URL",
    "TUNNEL_REGISTRY_AUTH_TOKEN",
    "SERVER_BEARER_TOKEN",
]

OPTIONAL_SECRET_KEYS: List[str] = [
    "HF_TOKEN",
]

REQUIRED_AUTH_KEYS: List[str] = [
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
]


class SecretSyncError(RuntimeError):
    """Raised when Kaggle secret dataset synchronization fails."""

    pass


def load_secrets_config(env_file_path: Optional[Path] = None) -> Dict[str, str]:
    """Load and validate required secrets and credentials from .env and environment.

    Args:
        env_file_path: Optional path to .env file.

    Returns:
        Dictionary containing all required keys with non-empty string values.

    Raises:
        ValueError: If any required key is missing or empty.
    """
    target_env = env_file_path or (REPO_ROOT / ".env")
    env_values: Dict[str, str] = {}

    # System environment provides fallback
    for k in REQUIRED_SECRET_KEYS + REQUIRED_AUTH_KEYS:
        if k in os.environ and os.environ[k].strip():
            env_values[k] = os.environ[k].strip()

    # Optional keys fallback from system environment
    for k in OPTIONAL_SECRET_KEYS:
        if k in os.environ and os.environ[k].strip():
            env_values[k] = os.environ[k].strip()

    # Target .env file is authoritative
    if target_env.is_file() and dotenv is not None:
        file_vals = dotenv.dotenv_values(target_env)
        for k, v in file_vals.items():
            if v is not None and str(v).strip():
                env_values[k] = str(v).strip()

    missing = []
    for k in REQUIRED_SECRET_KEYS + REQUIRED_AUTH_KEYS:
        val = env_values.get(k)
        if not val:
            missing.append(k)

    if missing:
        raise ValueError(
            f"Missing required secrets/credentials for Kaggle dataset sync: {', '.join(missing)}"
        )

    return env_values


def compute_secrets_hash(config: Dict[str, str]) -> str:
    """Compute SHA-256 digest of secret values."""
    keys_to_hash = [k for k in REQUIRED_SECRET_KEYS if k in config]
    for k in OPTIONAL_SECRET_KEYS:
        if k in config:
            keys_to_hash.append(k)
    payload = "|".join(f"{k}:{config[k]}" for k in keys_to_hash)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_secure_file(path: Path, content: str) -> None:
    """Write file with restricted permissions (0o600)."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def stage_dataset_files(stage_dir: Path, dataset_slug: str, config: Dict[str, str]) -> None:
    """Stage metadata and secret files into the temporary upload directory.

    Args:
        stage_dir: Directory where files should be written.
        dataset_slug: Kaggle dataset identifier (owner/audiogen-secrets).
        config: Validated secrets configuration.
    """
    meta = {
        "title": "AudioGen Runtime Secrets",
        "id": dataset_slug,
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    meta_path = stage_dir / "dataset-metadata.json"
    _write_secure_file(meta_path, json.dumps(meta, indent=2))

    secrets_dict = {k: config[k] for k in REQUIRED_SECRET_KEYS if k in config}
    for opt_k in OPTIONAL_SECRET_KEYS:
        if opt_k in config:
            secrets_dict[opt_k] = config[opt_k]

    secrets_json_path = stage_dir / "secrets.json"
    _write_secure_file(secrets_json_path, json.dumps(secrets_dict, indent=2))

    for k, v in secrets_dict.items():
        _write_secure_file(stage_dir / k, v)


def sync_secrets_dataset(
    env_file_path: Optional[Path] = None,
    kaggle_cmd: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> str:
    """Synchronize runtime secrets to the private Kaggle dataset.

    Args:
        env_file_path: Optional path to .env file.
        kaggle_cmd: Optional binary / command override for Kaggle CLI.
        cache_dir: Optional directory where .secrets_dataset_hash is saved.
        force: If True, upload a new version even if cached hash matches.

    Returns:
        Dataset slug (<KAGGLE_USERNAME>/audiogen-secrets).

    Raises:
        ValueError: If required configuration is missing.
        SecretSyncError: If Kaggle CLI operations fail.
    """
    config = load_secrets_config(env_file_path)
    username = config["KAGGLE_USERNAME"]
    dataset_slug = f"{username}/audiogen-secrets"

    cmd_base = list(kaggle_cmd) if kaggle_cmd is not None else ["kaggle"]
    executable = cmd_base[0]

    if not shutil.which(executable) and not Path(executable).exists():
        raise SecretSyncError(f"Kaggle CLI executable '{executable}' was not found in PATH.")

    secrets_hash = compute_secrets_hash(config)
    cache_path = (cache_dir or REPO_ROOT) / CACHE_FILE_NAME
    cached_hash = None
    if cache_path.is_file():
        try:
            cached_hash = cache_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read secrets hash cache: %s", exc)

    sub_env = dict(os.environ)
    sub_env["KAGGLE_USERNAME"] = username
    sub_env["KAGGLE_KEY"] = config["KAGGLE_KEY"]

    # Check remote dataset existence
    status_cmd = [*cmd_base, "datasets", "status", dataset_slug]
    logger.info("Checking Kaggle dataset status for %s...", dataset_slug)
    try:
        status_proc = subprocess.run(
            status_cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
            env=sub_env,
        )
    except FileNotFoundError as exc:
        raise SecretSyncError(f"Kaggle CLI not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SecretSyncError(f"Kaggle dataset status check timed out after {exc.timeout}s") from exc
    except Exception as exc:
        raise SecretSyncError(f"Failed to check Kaggle dataset status: {exc}") from exc

    status_out = (status_proc.stdout or "").strip()
    status_err = (status_proc.stderr or "").strip()
    combined_status = f"{status_out} {status_err}".lower()

    dataset_exists = False
    if status_proc.returncode == 0:
        dataset_exists = True
    elif "404" in combined_status or "not found" in combined_status:
        dataset_exists = False
    elif "unauthorized" in combined_status or "401" in combined_status or "403" in combined_status:
        raise SecretSyncError(
            f"Kaggle credentials unauthorized or rejected checking dataset '{dataset_slug}': {status_err or status_out}"
        )
    else:
        raise SecretSyncError(
            f"Unexpected failure checking Kaggle dataset status for '{dataset_slug}' "
            f"(exit code {status_proc.returncode}): {status_err or status_out}"
        )

    if dataset_exists and not force and cached_hash == secrets_hash:
        logger.info("Kaggle secrets dataset is up to date (hash matches). Skipping upload.")
        return dataset_slug

    with tempfile.TemporaryDirectory() as tmp_dir:
        stage_path = Path(tmp_dir)
        stage_dataset_files(stage_path, dataset_slug, config)

        if not dataset_exists:
            logger.info("Creating new private Kaggle dataset '%s'...", dataset_slug)
            create_cmd = [*cmd_base, "datasets", "create", "-p", str(stage_path), "-r", "skip"]
            try:
                proc = subprocess.run(
                    create_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                    check=False,
                    env=sub_env,
                )
            except Exception as exc:
                raise SecretSyncError(f"Failed to execute dataset creation: {exc}") from exc

            if proc.returncode != 0:
                err_msg = (proc.stderr or proc.stdout or "").strip()
                raise SecretSyncError(f"Failed to create Kaggle dataset '{dataset_slug}': {err_msg}")

            # Poll briefly for readiness
            logger.info("Polling dataset status until ready...")
            ready = False
            for _ in range(15):
                time.sleep(2.0)
                poll_proc = subprocess.run(
                    status_cmd,
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                    check=False,
                    env=sub_env,
                )
                if poll_proc.returncode == 0 and "ready" in (poll_proc.stdout or "").lower():
                    ready = True
                    break
            if not ready:
                logger.warning("Dataset creation initiated, but status did not become 'ready' within 30s.")
        else:
            logger.info("Updating existing Kaggle dataset '%s'...", dataset_slug)
            version_cmd = [
                *cmd_base,
                "datasets",
                "version",
                "-p",
                str(stage_path),
                "-m",
                "Update AudioGen runtime secrets",
                "-r",
                "skip",
            ]
            try:
                proc = subprocess.run(
                    version_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                    check=False,
                    env=sub_env,
                )
            except Exception as exc:
                raise SecretSyncError(f"Failed to execute dataset version update: {exc}") from exc

            if proc.returncode != 0:
                err_msg = (proc.stderr or proc.stdout or "").strip()
                raise SecretSyncError(
                    f"Failed to create new dataset version for '{dataset_slug}': {err_msg}"
                )

        try:
            cache_path.write_text(secrets_hash, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to save secrets dataset hash cache: %s", exc)

    logger.info("Kaggle secrets dataset '%s' successfully synchronized.", dataset_slug)
    return dataset_slug


def main() -> int:
    """CLI entry point for scripts/sync_secrets_dataset.py."""
    try:
        slug = sync_secrets_dataset()
        print(f"Successfully synced secrets dataset: {slug}")
        return 0
    except Exception as exc:
        logger.error("Kaggle secrets dataset sync failed: %s", exc)
        print(f"\n[SECRETS SYNC FAILED] {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
