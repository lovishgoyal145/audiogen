"""Unit tests for orchestrator/session_manager.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
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


def test_interactive_notebook_clones_public_repo_without_pat() -> None:
    """Verify interactive_notebook.ipynb clones public repository without GITHUB_PAT or secret client."""
    import json

    notebook_path = DEFAULT_STAGE_DIR / "interactive_notebook.ipynb"
    assert notebook_path.is_file()

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_code = "\n".join("".join(cell.get("source", [])) for cell in nb_data.get("cells", []))

    # GITHUB_PAT must not be referenced anywhere
    assert "GITHUB_PAT" not in all_code

    # Public repository git clone must be used
    expected_clone = "git clone https://github.com/lovishgoyal145/audiogen.git /kaggle/working/audiogen"
    assert expected_clone in all_code

    # Cell 1 (bootstrap) must not use UserSecretsClient for cloning
    cell_1_code = "".join(nb_data["cells"][1].get("source", []))
    assert "UserSecretsClient" not in cell_1_code


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
        ('avidok/audiogen has status "KernelWorkerStatus.ERROR"', "ERROR"),
        ("avidok/audiogen has status 'KernelWorkerStatus.RUNNING'", "RUNNING"),
        ('avidok/audiogen has status "KernelWorkerStatus.QUEUED"', "QUEUED"),
        ('avidok/audiogen has status "KernelWorkerStatus.CANCELLED"', "CANCELLED"),
        ('avidok/audiogen has status "KernelWorkerStatus.FAILED"', "FAILED"),
        ('avidok/audiogen has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
        ("avidok/audiogen has status 'KernelWorkerStatus.ERROR'", "ERROR"),
        ("avidok/audiogen has status 'KernelWorkerStatus.RUNNING'", "RUNNING"),
        ("avidok/audiogen has status KernelWorkerStatus.ERROR", "ERROR"),
        ("avidok/audiogen has status KernelWorkerStatus.RUNNING", "RUNNING"),
        ('avidok/audiogen has status "KernelWorkerStatus.ERROR."', "ERROR"),
        ("avidok/audiogen has status 'KernelWorkerStatus.RUNNING.'", "RUNNING"),
        ("avidok/audiogen has status KernelWorkerStatus.QUEUED.", "QUEUED"),
        ("KernelWorkerStatus.COMPLETE.", "COMPLETE"),
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

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result is None
    assert client.get.call_args[0][0] == "https://registry.example.com/get/tunnel_url"


def test_is_tunnel_healthy_registry_missing_tunnel_url() -> None:
    """Test is_tunnel_healthy returns None when registry response lacks tunnel_url."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"secret": "token123"}  # missing tunnel_url
    client.get.return_value = mock_resp

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result is None
    assert client.get.call_args[0][0] == "https://registry.example.com/get/tunnel_url"


def test_is_tunnel_healthy_health_check_fails() -> None:
    """Test is_tunnel_healthy returns None if registry has URL but GET /health fails.

    Guards the invariant: Never treat presence alone as readiness.
    """
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 503  # Server booting / unhealthy
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result is None


def test_is_tunnel_healthy_health_check_times_out() -> None:
    """Test is_tunnel_healthy returns None when GET /health times out."""
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        if "tunnel_url" in url:
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            raise httpx.TimeoutException("Health check timed out")
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result is None


def test_is_tunnel_healthy_success() -> None:
    """Test is_tunnel_healthy returns validated URL when both registry and GET /health succeed."""
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online", "gpu": True}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"
    assert client.get.call_args_list[0][0][0] == "https://registry.example.com/get/tunnel_url"
    assert client.get.call_args_list[1][0][0] == "https://active-tunnel.trycloudflare.com/health"


def test_is_tunnel_healthy_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy asserts exact path and includes Authorization: Bearer <token>."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "token-session-123")
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    assert client.get.call_count == 2
    # Step 1: registry call must target exact /get/tunnel_url path and contain Authorization header
    reg_call = client.get.call_args_list[0]
    assert reg_call[0][0] == "https://registry.example.com/get/tunnel_url"
    assert reg_call.kwargs.get("headers") == {"Authorization": "Bearer token-session-123"}
    # Step 2: tunnel health check must NOT contain the registry auth header
    health_call = client.get.call_args_list[1]
    assert health_call[0][0] == "https://active-tunnel.trycloudflare.com/health"
    assert "headers" not in health_call.kwargs


def test_is_tunnel_healthy_upstash_rest_json_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy parses Upstash Redis REST JSON result string: {"result": "{\"tunnel_url\": \"...\"}"}."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "upstash-token")
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {
                "result": json.dumps({
                    "tunnel_url": "https://active-upstash.trycloudflare.com",
                    "secret": "secret123",
                })
            }
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    # Provide bare Upstash domain -> should auto-resolve to /get/tunnel_url
    result = is_tunnel_healthy(registry_url="https://large-pup-282364.upstash.io", client=client)
    assert result == "https://active-upstash.trycloudflare.com"

    reg_call = client.get.call_args_list[0]
    assert reg_call[0][0] == "https://large-pup-282364.upstash.io/get/tunnel_url"
    assert reg_call.kwargs.get("headers") == {"Authorization": "Bearer upstash-token"}


def test_is_tunnel_healthy_upstash_rest_raw_url_string() -> None:
    """Test is_tunnel_healthy parses Upstash Redis REST raw string result: {"result": "https://..."}."""
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"result": "https://raw-url.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://large-pup-282364.upstash.io/get/tunnel_url", client=client)
    assert result == "https://raw-url.trycloudflare.com"


def test_is_tunnel_healthy_upstash_rest_key_not_found() -> None:
    """Test is_tunnel_healthy returns None when Upstash returns {"result": null}."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": None}
    client.get.return_value = mock_resp

    result = is_tunnel_healthy(registry_url="https://large-pup-282364.upstash.io", client=client)
    assert result is None
    assert client.get.call_args[0][0] == "https://large-pup-282364.upstash.io/get/tunnel_url"


def test_is_tunnel_healthy_without_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is unset."""
    monkeypatch.delenv("TUNNEL_REGISTRY_AUTH_TOKEN", raising=False)
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    reg_call = client.get.call_args_list[0]
    assert reg_call[0][0] == "https://registry.example.com/get/tunnel_url"
    assert "headers" not in reg_call.kwargs


def test_is_tunnel_healthy_with_empty_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_tunnel_healthy sends no Authorization header when TUNNEL_REGISTRY_AUTH_TOKEN is empty/whitespace."""
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "   ")
    client = MagicMock(spec=httpx.Client)

    def mock_get(url: str, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "tunnel_url" in url:
            resp.status_code = 200
            resp.json.return_value = {"tunnel_url": "https://active-tunnel.trycloudflare.com"}
            return resp
        elif "health" in url:
            resp.status_code = 200
            resp.json.return_value = {"status": "online"}
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    client.get.side_effect = mock_get

    result = is_tunnel_healthy(registry_url="https://registry.example.com/get/tunnel_url", client=client)
    assert result == "https://active-tunnel.trycloudflare.com"

    reg_call = client.get.call_args_list[0]
    assert reg_call[0][0] == "https://registry.example.com/get/tunnel_url"
    assert "headers" not in reg_call.kwargs


def test_kaggle_push_creates_and_cleans_up_ephemeral_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _kaggle_push writes ephemeral runtime_secrets.json with 0o600 perms during push and unlinks in finally."""
    monkeypatch.setenv("TUNNEL_REGISTRY_WEBHOOK_URL", "https://registry.example.com/set/tunnel_url")
    monkeypatch.setenv("SERVER_BEARER_TOKEN", "super-secret-token")
    monkeypatch.setenv("TUNNEL_REGISTRY_AUTH_TOKEN", "auth-token-xyz")

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    secrets_file = stage_dir / "runtime_secrets.json"
    assert not secrets_file.exists()

    def fake_subprocess_run(cmd, **kwargs):
        # Invariant check: While push is executing, secrets_file must exist on disk with exact payload & 0o600
        assert secrets_file.is_file()
        file_mode = stat.S_IMODE(os.stat(secrets_file).st_mode)
        assert file_mode == 0o600, f"Expected 0o600 permissions, got {oct(file_mode)}"
        with open(secrets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["TUNNEL_REGISTRY_WEBHOOK_URL"] == "https://registry.example.com/set/tunnel_url"
        assert data["SERVER_BEARER_TOKEN"] == "super-secret-token"
        assert data["TUNNEL_REGISTRY_AUTH_TOKEN"] == "auth-token-xyz"
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Pushed", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        _kaggle_push(stage_dir=stage_dir)

    # Invariant check: In finally, secrets_file must be completely cleaned up
    assert not secrets_file.exists()


def test_kaggle_push_overwrites_existing_runtime_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _kaggle_push unconditionally overwrites existing runtime_secrets.json with fresh credentials."""
    monkeypatch.setenv("TUNNEL_REGISTRY_WEBHOOK_URL", "https://fresh-registry.example.com")
    monkeypatch.setenv("SERVER_BEARER_TOKEN", "fresh-token")

    stage_dir = tmp_path / "stage_overwrite"
    stage_dir.mkdir()
    secrets_file = stage_dir / "runtime_secrets.json"
    # Pre-seed with stale credentials
    with open(secrets_file, "w", encoding="utf-8") as f:
        json.dump({"TUNNEL_REGISTRY_WEBHOOK_URL": "stale", "SERVER_BEARER_TOKEN": "stale"}, f)

    def fake_subprocess_run(cmd, **kwargs):
        assert secrets_file.is_file()
        with open(secrets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Invariant: must be overwritten with fresh values
        assert data["TUNNEL_REGISTRY_WEBHOOK_URL"] == "https://fresh-registry.example.com"
        assert data["SERVER_BEARER_TOKEN"] == "fresh-token"
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Pushed", stderr="")

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        _kaggle_push(stage_dir=stage_dir)

    assert not secrets_file.exists()


def test_kaggle_push_cleans_up_ephemeral_secrets_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _kaggle_push unlinks ephemeral runtime_secrets.json even when the push command fails."""
    monkeypatch.setenv("TUNNEL_REGISTRY_WEBHOOK_URL", "https://registry.example.com")
    monkeypatch.setenv("SERVER_BEARER_TOKEN", "token123")

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    secrets_file = stage_dir / "runtime_secrets.json"

    with patch("shutil.which", return_value="/usr/local/bin/kaggle"), \
         patch("subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Fail")):
        with pytest.raises(KagglePushError):
            _kaggle_push(stage_dir=stage_dir)

    assert not secrets_file.exists()


def test_interactive_notebook_headless_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify interactive notebook secrets cell executes safely when UserSecretsClient raises exception."""
    notebook_path = DEFAULT_STAGE_DIR / "interactive_notebook.ipynb"
    assert notebook_path.is_file()

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Locate secrets cell (id: c3519be0)
    secrets_cell = next(cell for cell in nb_data["cells"] if cell.get("id") == "c3519be0")
    code = "".join(secrets_cell["source"])

    # Simulate headless environment: UserSecretsClient raises ConnectionError / HTTPError
    class MockFailingSecretsClient:
        def __init__(self):
            pass

        def get_secret(self, key):
            raise ConnectionError("HTTP Error 400: Bad Request")

    # Staged runtime secrets file simulation
    staged_secrets = {
        "TUNNEL_REGISTRY_WEBHOOK_URL": "https://headless-test.example.com",
        "SERVER_BEARER_TOKEN": "headless-secret-token",
    }
    secrets_file = tmp_path / "runtime_secrets.json"
    with open(secrets_file, "w", encoding="utf-8") as f:
        json.dump(staged_secrets, f)

    orig_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # Clear env to test loading from staged secrets
        monkeypatch.delenv("TUNNEL_REGISTRY_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SERVER_BEARER_TOKEN", raising=False)
        fake_kaggle_secrets = MagicMock()
        fake_kaggle_secrets.UserSecretsClient = MockFailingSecretsClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle_secrets)

        exec_globals = {}
        # Must execute without throwing any exception
        exec(code, exec_globals)

        assert os.environ.get("TUNNEL_REGISTRY_WEBHOOK_URL") == "https://headless-test.example.com"
        assert os.environ.get("SERVER_BEARER_TOKEN") == "headless-secret-token"
        # Invariant: candidate file must be unlinked to prevent leaking in Kaggle output datasets
        assert not secrets_file.exists(), "Staged secrets file must be unlinked after loading"
    finally:
        os.chdir(orig_cwd)


def test_interactive_notebook_has_no_hardcoded_secrets() -> None:
    """Verify interactive notebook template contains ZERO hardcoded credentials or bearer tokens."""
    notebook_path = DEFAULT_STAGE_DIR / "interactive_notebook.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_code = "\n".join("".join(cell.get("source", [])) for cell in nb_data.get("cells", []))

    forbidden_patterns = [
        "https://large-pup",
        ".upstash.io",
        "trycloudflare.com",
        "ghp_",
        "token = \"",
        "token = '",
        "secret = \"",
        "secret = '",
        "api_key = \"",
        "api_key = '",
        "password = \"",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in all_code, f"Potential hardcoded secret pattern '{pattern}' found in notebook template."

    # Verify that tokens and URLs are retrieved dynamically from os.environ
    assert 'os.environ.get("SERVER_BEARER_TOKEN")' in all_code
    assert 'os.environ.get("TUNNEL_REGISTRY_WEBHOOK_URL")' in all_code

