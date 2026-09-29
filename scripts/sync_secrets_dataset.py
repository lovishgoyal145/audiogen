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
import re
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


def load_secrets_config(
    env_file_path: Optional[Path] = None,
    secrets_override: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Load and validate required secrets and credentials from .env, environment, and overrides.

    Args:
        env_file_path: Optional path to .env file.
        secrets_override: Optional explicit secret key-value overrides.

    Returns:
        Dictionary containing all required keys with non-empty string values.

    Raises:
        ValueError: If any required key is missing or empty.
    """
    target_env = env_file_path or (REPO_ROOT / ".env")
    env_values: Dict[str, str] = {}

    # System environment provides fallback only when explicit env file is not specified
    if env_file_path is None:
        for k in REQUIRED_SECRET_KEYS + REQUIRED_AUTH_KEYS + OPTIONAL_SECRET_KEYS:
            if k in os.environ and os.environ[k].strip():
                env_values[k] = os.environ[k].strip()

    # Target .env file is authoritative
    if target_env.is_file() and dotenv is not None:
        file_vals = dotenv.dotenv_values(target_env)
        for k, v in file_vals.items():
            if v is not None and str(v).strip():
                env_values[k] = str(v).strip()

    # Explicit overrides take highest precedence
    if secrets_override is not None:
        for k, v in secrets_override.items():
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
    """Compute SHA-256 digest of secret values and codebase bundle."""
    keys_to_hash = [k for k in REQUIRED_SECRET_KEYS if k in config]
    for k in OPTIONAL_SECRET_KEYS:
        if k in config:
            keys_to_hash.append(k)
    payload = "|".join(config[k] for k in keys_to_hash)
    # Include modification timestamps of packaged files so code updates trigger sync
    code_marker = ""
    for check_file in (
        REPO_ROOT / "server" / "interactive_notebook.ipynb",
        REPO_ROOT / "server" / "app.py",
        REPO_ROOT / "requirements.txt",
    ):
        if check_file.is_file():
            code_marker += f":{check_file.stat().st_mtime_ns}"
    payload += code_marker
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
    parts = dataset_slug.split("/")
    slug_name = parts[-1] if parts else "audiogen-secrets"
    if slug_name == "audiogen-secrets":
        clean_title = "AudioGen Runtime Secrets"
    else:
        clean_title = f"AudioGen Secrets {slug_name}"[:50]
    meta = {
        "title": clean_title,
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

    # Bundle codebase archive into the dataset for zero-network execution on Kaggle
    code_tar_path = stage_dir / "code.tar.gz"
    try:
        import tarfile
        with tarfile.open(code_tar_path, "w:gz") as tar:
            for folder in ("src", "server", "voices"):
                src_path = REPO_ROOT / folder
                if src_path.is_dir():
                    tar.add(src_path, arcname=folder)
            req = REPO_ROOT / "requirements.txt"
            if req.is_file():
                tar.add(req, arcname="requirements.txt")
        logger.info("Bundled audiogen code into %s (size: %d bytes)", code_tar_path.name, code_tar_path.stat().st_size)
    except Exception as exc:
        logger.warning("Could not bundle code archive into dataset: %s", exc)


def sync_secrets_dataset(
    env_file_path: Optional[Path] = None,
    kaggle_cmd: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    dataset_slug_override: Optional[str] = None,
    secrets_override: Optional[Dict[str, str]] = None,
) -> str:
    """Synchronize runtime secrets to the private Kaggle dataset.

    Args:
        env_file_path: Optional path to .env file.
        kaggle_cmd: Optional binary / command override for Kaggle CLI.
        cache_dir: Optional directory where .secrets_dataset_hash is saved.
        force: If True, upload a new version even if cached hash matches.
        dataset_slug_override: Optional explicit dataset slug (e.g. for profile isolation).
        secrets_override: Optional explicit secret key-value overrides.

    Returns:
        Dataset slug (<KAGGLE_USERNAME>/audiogen-secrets or dataset_slug_override).

    Raises:
        ValueError: If required configuration is missing.
        SecretSyncError: If Kaggle CLI operations fail.
    """
    config = load_secrets_config(env_file_path, secrets_override=secrets_override)
    username = config["KAGGLE_USERNAME"]
    raw_slug = dataset_slug_override or f"{username}/audiogen-secrets"

    # Normalize slug to ensure Kaggle compliance (only alphanumeric and hyphens, no underscores)
    parts = raw_slug.split("/", 1)
    if len(parts) == 2:
        owner, slug_name = parts
    else:
        owner, slug_name = username, raw_slug
    clean_slug_name = re.sub(r"[^a-zA-Z0-9]+", "-", slug_name).strip("-").lower()
    dataset_slug = f"{owner}/{clean_slug_name}"

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
    sub_env["KAGGLE_API_TOKEN"] = config["KAGGLE_KEY"]

    # Isolate Kaggle configuration directory to prevent reading global ~/.kaggle/access_token
    kaggle_cfg_dir = (cache_dir or REPO_ROOT) / ".kaggle_sub_cfg"
    kaggle_cfg_dir.mkdir(parents=True, exist_ok=True)
    kjson = kaggle_cfg_dir / "kaggle.json"
    kjson.write_text(json.dumps({"username": username, "key": config["KAGGLE_KEY"]}), encoding="utf-8")
    try:
        os.chmod(kjson, 0o600)
    except Exception:
        pass
    sub_env["KAGGLE_CONFIG_DIR"] = str(kaggle_cfg_dir)

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
    elif "unauthorized" in combined_status or "401" in combined_status:
        raise SecretSyncError(
            f"Kaggle credentials unauthorized checking dataset '{dataset_slug}': {status_err or status_out}"
        )
    elif "403" in combined_status:
        # On Kaggle's API, GetDatasetStatus returns 403 when a dataset does NOT exist yet.
        # We verify whether credentials are valid and if the dataset exists via `kaggle datasets list --mine`.
        list_cmd = [*cmd_base, "datasets", "list", "--mine", "-v"]
        try:
            list_proc = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
                env=sub_env,
            )
        except Exception as exc:
            raise SecretSyncError(f"Failed to verify Kaggle dataset list: {exc}") from exc

        list_out = (list_proc.stdout or "").strip()
        list_err = (list_proc.stderr or "").strip()
        combined_list = f"{list_out} {list_err}".lower()

        if list_proc.returncode == 0:
            dataset_exists = any(dataset_slug.lower() in line.lower() for line in list_out.splitlines())
        elif "unauthorized" in combined_list or "401" in combined_list:
            raise SecretSyncError(
                f"Kaggle credentials unauthorized checking dataset '{dataset_slug}': {list_err or list_out}"
            )
        else:
            raise SecretSyncError(
                f"Failed to check Kaggle dataset status for '{dataset_slug}' (exit code {list_proc.returncode}): {list_err or list_out}"
            )
    else:
        raise SecretSyncError(
            f"Unexpected failure checking Kaggle dataset status for '{dataset_slug}' "
            f"(exit code {status_proc.returncode}): {status_err or status_out}"
        )

    if dataset_exists and not force and cached_hash == secrets_hash:
        logger.info("Kaggle secrets dataset is up to date (hash matches). Skipping upload.")
        return dataset_slug

    if dataset_exists and not force:
        # Verify remote dataset files to ensure code.tar.gz is present
        files_cmd = [*cmd_base, "datasets", "files", dataset_slug]
        try:
            files_proc = subprocess.run(
                files_cmd,
                capture_output=True,
                text=True,
                timeout=20.0,
                check=False,
                env=sub_env,
            )
            if files_proc.returncode == 0 and "code.tar.gz" not in (files_proc.stdout or ""):
                logger.info("Remote dataset '%s' missing code.tar.gz; forcing version update.", dataset_slug)
                force = True
        except Exception as exc:
            logger.debug("Failed to inspect dataset files: %s", exc)

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

            create_out = (proc.stdout or "").strip()
            create_err = (proc.stderr or "").strip()
            if proc.returncode != 0 or "dataset creation error" in create_out.lower() or "dataset creation error" in create_err.lower():
                err_msg = create_err or create_out
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
