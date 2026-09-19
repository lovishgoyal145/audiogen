"""Unit and integration tests for scripts/sync_secrets_dataset.py.

Verifies:
1. Environment and .env secret sourcing and validation.
2. Fast-fail behavior when required secrets or Kaggle credentials are missing.
3. Private Kaggle dataset creation (strictly private, no -u/--public flag).
4. SHA-256 hash caching to prevent redundant Kaggle dataset version uploads.
5. Version creation when secrets change.
6. Ephemeral staging directory cleanup and restricted 0o600 file permissions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Dict
from unittest.mock import MagicMock, patch
import pytest

from scripts.sync_secrets_dataset import (
    CACHE_FILE_NAME,
    REQUIRED_AUTH_KEYS,
    REQUIRED_SECRET_KEYS,
    SecretSyncError,
    compute_secrets_hash,
    load_secrets_config,
    stage_dataset_files,
    sync_secrets_dataset,
)


@pytest.fixture
def valid_config() -> Dict[str, str]:
    """Provide a valid dictionary of required secrets and Kaggle credentials."""
    return {
        "TUNNEL_REGISTRY_WEBHOOK_URL": "https://upstash.example.com",
        "TUNNEL_REGISTRY_AUTH_TOKEN": "mock-reg-auth-token-12345",
        "SERVER_BEARER_TOKEN": "mock-server-bearer-secret-67890",
        "KAGGLE_USERNAME": "testuser",
        "KAGGLE_KEY": "test-kaggle-api-key",
    }


def test_load_secrets_config_success(tmp_path: Path, valid_config: Dict[str, str]) -> None:
    """Verify load_secrets_config reads all required variables from .env file."""
    env_file = tmp_path / ".env"
    lines = [f"{k}={v}" for k, v in valid_config.items()]
    env_file.write_text("\n".join(lines), encoding="utf-8")

    loaded = load_secrets_config(env_file_path=env_file)
    for k, v in valid_config.items():
        assert loaded[k] == v


@pytest.mark.parametrize("missing_key", REQUIRED_SECRET_KEYS + REQUIRED_AUTH_KEYS)
def test_load_secrets_config_missing_key_raises(
    tmp_path: Path, valid_config: Dict[str, str], missing_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify load_secrets_config raises ValueError if any required key is missing or empty."""
    del valid_config[missing_key]
    monkeypatch.delenv(missing_key, raising=False)
    env_file = tmp_path / ".env"
    lines = [f"{k}={v}" for k, v in valid_config.items()]
    env_file.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_secrets_config(env_file_path=env_file)
    assert missing_key in str(exc_info.value)


def test_compute_secrets_hash_consistency(valid_config: Dict[str, str]) -> None:
    """Verify SHA-256 hash is deterministic and sensitive to changes."""
    hash1 = compute_secrets_hash(valid_config)
    hash2 = compute_secrets_hash(valid_config)
    assert hash1 == hash2

    altered_config = dict(valid_config)
    altered_config["SERVER_BEARER_TOKEN"] = "altered-token"
    hash_altered = compute_secrets_hash(altered_config)
    assert hash1 != hash_altered


def test_stage_dataset_files_permissions_and_contents(tmp_path: Path, valid_config: Dict[str, str]) -> None:
    """Verify stage_dataset_files creates files with 0o600 permissions and exact contents."""
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    dataset_slug = f"{valid_config['KAGGLE_USERNAME']}/audiogen-secrets"

    stage_dataset_files(stage_dir, dataset_slug, valid_config)

    # Check metadata
    meta_path = stage_dir / "dataset-metadata.json"
    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == dataset_slug
    assert meta["title"] == "AudioGen Runtime Secrets"
    assert meta["licenses"] == [{"name": "CC0-1.0"}]

    # Check secrets.json
    secrets_json = stage_dir / "secrets.json"
    assert secrets_json.is_file()
    file_mode = stat.S_IMODE(os.stat(secrets_json).st_mode)
    assert file_mode == 0o600, f"Expected 0o600, got {oct(file_mode)}"
    with open(secrets_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in REQUIRED_SECRET_KEYS:
        assert data[k] == valid_config[k]

    # Check individual secret files
    for k in REQUIRED_SECRET_KEYS:
        sf = stage_dir / k
        assert sf.is_file()
        assert stat.S_IMODE(os.stat(sf).st_mode) == 0o600
        assert sf.read_text(encoding="utf-8") == valid_config[k]


def test_sync_secrets_dataset_creates_private_dataset_when_nonexistent(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify sync_secrets_dataset invokes `kaggle datasets create` without --public / -u."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    commands_run = []

    def mock_run(cmd, **kwargs):
        commands_run.append(list(cmd))
        if "status" in cmd:
            # First check returns 404 (nonexistent)
            if len(commands_run) == 1:
                return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="404 - Not Found")
            # Subsequent check after create returns ready
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ready", stderr="")
        if "create" in cmd:
            # Assert private creation: neither -u nor --public is in the command arguments
            assert "-u" not in cmd
            assert "--public" not in cmd
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Dataset created", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("time.sleep", return_value=None):
        slug = sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)

    assert slug == f"{valid_config['KAGGLE_USERNAME']}/audiogen-secrets"
    # Ensure cache file was written
    cache_file = tmp_path / CACHE_FILE_NAME
    assert cache_file.is_file()
    assert cache_file.read_text(encoding="utf-8") == compute_secrets_hash(valid_config)

    # Ensure create command was issued
    assert any("create" in c for c in commands_run)
    assert not any("version" in c for c in commands_run)


def test_sync_secrets_dataset_skips_when_hash_matches(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify sync_secrets_dataset skips upload when cache hash matches remote dataset."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    # Seed cache
    cache_file = tmp_path / CACHE_FILE_NAME
    expected_hash = compute_secrets_hash(valid_config)
    cache_file.write_text(expected_hash, encoding="utf-8")

    commands_run = []

    def mock_run(cmd, **kwargs):
        commands_run.append(list(cmd))
        if "status" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ready", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=mock_run):
        slug = sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)

    assert slug == f"{valid_config['KAGGLE_USERNAME']}/audiogen-secrets"
    # Only status should have been called; neither create nor version should be called
    assert len(commands_run) == 1
    assert "status" in commands_run[0]
    assert not any("create" in c for c in commands_run)
    assert not any("version" in c for c in commands_run)


def test_sync_secrets_dataset_creates_version_when_hash_differs(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify sync_secrets_dataset creates new version when existing dataset secrets changed."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    # Seed stale cache
    cache_file = tmp_path / CACHE_FILE_NAME
    cache_file.write_text("stale-sha256-hash-value", encoding="utf-8")

    commands_run = []

    def mock_run(cmd, **kwargs):
        commands_run.append(list(cmd))
        if "status" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ready", stderr="")
        if "version" in cmd:
            assert "-m" in cmd
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Dataset version created", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=mock_run):
        slug = sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)

    assert slug == f"{valid_config['KAGGLE_USERNAME']}/audiogen-secrets"
    assert any("version" in c for c in commands_run)
    assert not any("create" in c for c in commands_run)
    # Cache file updated with new hash
    assert cache_file.read_text(encoding="utf-8") == compute_secrets_hash(valid_config)


def test_sync_secrets_dataset_cli_missing_raises_error(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify sync_secrets_dataset raises SecretSyncError if Kaggle CLI executable is missing."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(SecretSyncError) as exc_info:
            sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)
    assert "was not found in PATH" in str(exc_info.value)


def test_sync_secrets_dataset_cleans_up_staging_on_error(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify temporary staging directory is cleaned up even if Kaggle command fails."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    staged_dirs_seen = []

    def mock_run(cmd, **kwargs):
        if "status" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="404 - Not Found")
        if "create" in cmd:
            p_idx = cmd.index("-p")
            staged_path = Path(cmd[p_idx + 1])
            staged_dirs_seen.append(staged_path)
            # Stage directory exists during execution
            assert staged_path.is_dir()
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="Failed to create dataset")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=mock_run):
        with pytest.raises(SecretSyncError):
            sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)

    assert len(staged_dirs_seen) == 1
    # After context exits, temp staging dir must be unlinked
    assert not staged_dirs_seen[0].exists()


def test_sync_secrets_dataset_unexpected_status_error_raises_secretsyncerror(
    tmp_path: Path, valid_config: Dict[str, str]
) -> None:
    """Verify sync_secrets_dataset raises SecretSyncError on non-404 status failures."""
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in valid_config.items()), encoding="utf-8")

    def mock_run(cmd, **kwargs):
        if "status" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=2,
                stdout="",
                stderr="Internal Server Error: 500",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=mock_run):
        with pytest.raises(SecretSyncError) as exc_info:
            sync_secrets_dataset(env_file_path=env_file, cache_dir=tmp_path)
    assert "Unexpected failure checking Kaggle dataset status" in str(exc_info.value)


def test_validate_dataset_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_dataset_status returns True when status check succeeds."""
    from scripts.verify_env_and_auth import validate_dataset_status

    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "datasets", "status", "testuser/audiogen-secrets"],
            returncode=0,
            stdout="ready",
            stderr="",
        )
        assert validate_dataset_status() is True
        mock_run.assert_called_once()


def test_validate_dataset_status_failure_raises_preflight_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_dataset_status raises PreflightValidationError when dataset is missing/error."""
    from scripts.verify_env_and_auth import PreflightValidationError, validate_dataset_status

    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "datasets", "status", "testuser/audiogen-secrets"],
            returncode=1,
            stdout="",
            stderr="404 - Not Found",
        )
        with pytest.raises(PreflightValidationError) as exc_info:
            validate_dataset_status()
        assert "is missing or inaccessible" in str(exc_info.value)


def test_validate_dataset_status_missing_username_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_dataset_status raises error when username is not set."""
    from scripts.verify_env_and_auth import PreflightValidationError, validate_dataset_status

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_dataset_status()
    assert "KAGGLE_USERNAME must be defined" in str(exc_info.value)


