"""Unit tests for orchestrator/session_manager.py."""

from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import httpx
import pytest

from orchestrator.session_manager import (
    DEFAULT_STAGE_DIR,
    REPO_ROOT,
    TERMINAL_FAILURE_STATUSES,
    KagglePushError,
    KaggleStatusError,
    _kaggle_push,
    get_kaggle_status,
    is_tunnel_healthy,
    parse_kaggle_status_output,
)


def test_terminal_failure_statuses_exact_members() -> None:
    """Verify all terminal failure statuses required by specification are present."""
    expected = {
        "ERROR",
        "FAILED",
        "CANCELLED",
        "CANCEL",
        "CANCEL_REQUESTED",
        "CANCEL_ACKNOWLEDGED",
        "COMPLETE",
    }
    assert expected.issubset(TERMINAL_FAILURE_STATUSES)


def test_kaggle_push_success(tmp_path: Path) -> None:
    """Test _kaggle_push invokes `kaggle kernels push -p <dir>` successfully."""
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "kernels", "push", "-p", str(stage_dir)],
            returncode=0,
            stdout="Kernel version 1 successfully pushed.",
            stderr="",
        )

        _kaggle_push(stage_dir=stage_dir)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["kaggle", "kernels", "push", "-p"]
        assert cmd[4] == str(stage_dir)


def test_default_stage_dir_points_to_server() -> None:
    """Verify DEFAULT_STAGE_DIR points to server directory and contains notebook."""
    expected_stage = REPO_ROOT / "server"
    assert DEFAULT_STAGE_DIR == expected_stage
    assert DEFAULT_STAGE_DIR.is_dir()
    assert (DEFAULT_STAGE_DIR / "interactive_notebook.ipynb").is_file()
    assert (REPO_ROOT / "config" / "kernel-metadata.json").is_file()


def test_kaggle_push_uses_default_server_stage_dir() -> None:
    """Verify default invocation of _kaggle_push pushes from server directory and cleans up temp links."""
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "kernels", "push", "-p", str(DEFAULT_STAGE_DIR)],
            returncode=0,
            stdout="Kernel version 1 pushed.",
            stderr="",
        )

        _kaggle_push()

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["kaggle", "kernels", "push", "-p"]
        assert cmd[4] == str(DEFAULT_STAGE_DIR)
        # Ensure temporary metadata link was cleaned up
        assert not (DEFAULT_STAGE_DIR / "kernel-metadata.json").exists()


def test_kernel_metadata_code_file_relative_and_resolvable() -> None:
    """Verify kernel metadata references interactive_notebook.ipynb relatively."""
    import json

    meta_file = REPO_ROOT / "config" / "kernel-metadata.json"
    assert meta_file.is_file()

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    code_file = data.get("code_file")
    assert code_file == "interactive_notebook.ipynb"
    assert not Path(code_file).is_absolute()
    assert "notebook_template.ipynb" not in code_file
    assert (DEFAULT_STAGE_DIR / code_file).is_file()


def test_kaggle_push_non_zero_exit_raises_error(tmp_path: Path) -> None:
    """Test _kaggle_push raises KagglePushError on non-zero exit code."""
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "kernels", "push"],
            returncode=1,
            stdout="",
            stderr="403 - Forbidden: Invalid credentials",
        )

        with pytest.raises(KagglePushError) as exc_info:
            _kaggle_push(stage_dir=tmp_path)

        assert "Kaggle push failed with exit code 1" in str(exc_info.value)
        assert "Forbidden" in str(exc_info.value)


def test_kaggle_push_missing_cli_raises_error(tmp_path: Path) -> None:
    """Test _kaggle_push raises KagglePushError when kaggle executable is missing."""
    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(KagglePushError) as exc_info:
            _kaggle_push(stage_dir=tmp_path)

        assert "executable" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_kaggle_push_timeout_raises_error(tmp_path: Path) -> None:
    """Test _kaggle_push raises KagglePushError when command times out."""
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="kaggle", timeout=60.0)):
        with pytest.raises(KagglePushError) as exc_info:
            _kaggle_push(stage_dir=tmp_path)

        assert "timed out after 60.0s" in str(exc_info.value)


@pytest.mark.parametrize(
    "cli_output, expected_status",
    [
        ('avidok/audiogen has status "running"', "RUNNING"),
        ("avidok/audiogen has status 'complete'", "COMPLETE"),
        ('avidok/audiogen has status "error"', "ERROR"),
        ("avidok/audiogen has status 'queued'", "QUEUED"),
        ('avidok/audiogen has status "cancel_requested"', "CANCEL_REQUESTED"),
        ('avidok/audiogen has status "cancelled"', "CANCELLED"),
        ("Kernel status is FAILED", "FAILED"),
        ("RUNNING", "RUNNING"),
    ],
)
def test_parse_kaggle_status_output(cli_output: str, expected_status: str) -> None:
    """Test status string extraction and normalization from CLI stdout."""
    assert parse_kaggle_status_output(cli_output) == expected_status


def test_get_kaggle_status_success() -> None:
    """Test get_kaggle_status executes CLI and returns normalized uppercase status."""
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "kernels", "status", "avidok/audiogen"],
            returncode=0,
            stdout='avidok/audiogen has status "running"',
            stderr="",
        )

        status = get_kaggle_status("avidok/audiogen")
        assert status == "RUNNING"
        mock_run.assert_called_once()


def test_get_kaggle_status_failure_raises_error() -> None:
    """Test get_kaggle_status raises KaggleStatusError on non-zero exit code."""
    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["kaggle", "kernels", "status", "avidok/audiogen"],
            returncode=1,
            stdout="",
            stderr="Kernel not found",
        )

        with pytest.raises(KaggleStatusError) as exc_info:
            get_kaggle_status("avidok/audiogen")

        assert "Kaggle status command failed" in str(exc_info.value)


def test_is_tunnel_healthy_missing_registry_url() -> None:
    """Test is_tunnel_healthy returns None when no registry URL is configured."""
    with patch.dict("os.environ", {}, clear=True):
        assert is_tunnel_healthy(registry_url=None) is None


def test_is_tunnel_healthy_registry_unreachable() -> None:
    """Test is_tunnel_healthy returns None when registry endpoint is unreachable."""
    client = MagicMock(spec=httpx.Client)
    client.get.side_effect = httpx.ConnectError("Connection refused")

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result is None


def test_is_tunnel_healthy_registry_missing_tunnel_url() -> None:
    """Test is_tunnel_healthy returns None when registry response lacks tunnel_url."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"secret": "token123"}  # missing tunnel_url
    client.get.return_value = mock_resp

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result is None


def test_is_tunnel_healthy_health_check_fails() -> None:
    """Test is_tunnel_healthy returns None if registry has URL but GET /health fails.

    Guards the invariant: Never treat presence alone as readiness.
    """
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "registry" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 503  # Server booting / unhealthy
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result is None


def test_is_tunnel_healthy_health_check_times_out() -> None:
    """Test is_tunnel_healthy returns None when GET /health times out."""
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        if "registry" in url:
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            raise httpx.TimeoutException("Health check timed out")
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result is None


def test_is_tunnel_healthy_success() -> None:
    """Test is_tunnel_healthy returns validated URL when both registry and GET /health succeed."""
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "registry" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online", "gpu": True}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"


def test_is_tunnel_healthy_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy includes Authorization: Bearer <token> when TUNNEL_REGISTRY_AUTH_TOKEN is set."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "token-session-123")
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "registry" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    assert client.get.call_count == 2
    # Step 1: registry call must contain the Authorization header
    reg_call = client.get.call_args_list[0]
    assert reg_call.kwargs.get("headers") == {"Authorization": "Bearer token-session-123"}
    # Step 2: tunnel health check must NOT contain the registry auth header
    health_call = client.get.call_args_list[1]
    assert "headers" not in health_call.kwargs


def test_is_tunnel_healthy_without_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is unset."""
    monkeypatch.delenv("TUNNEL_REGISTRY_AUTH_TOKEN", raising=False)
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "registry" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    reg_call = client.get.call_args_list[0]
    assert "headers" not in reg_call.kwargs


def test_is_tunnel_healthy_with_empty_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is empty/whitespace."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "   ")
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "registry" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/api", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    reg_call = client.get.call_args_list[0]
    assert "headers" not in reg_call.kwargs
