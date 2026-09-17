"""Unit and integration tests for scripts/verify_env_and_auth.py.

Includes mandatory mock assertions on the headers argument for HTTP client invocations,
environment variable validations, live probe handling, and port availability checks.
"""

from __future__ import annotations

from pathlib import Path
import socket
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest

from scripts.verify_env_and_auth import (
    REQUIRED_ENV_VARS,
    PreflightValidationError,
    check_port_open,
    load_environment,
    main,
    probe_tunnel_registry,
    run_preflight_checks,
    validate_env_vars,
)


@pytest.fixture
def valid_env_dict() -> Dict[str, str]:
    """Provide a dictionary with all required and valid environment variables."""
    return {
        "TUNNEL_REGISTRY_WEBHOOK_URL": "https://large-pup-282364.upstash.io",
        "TUNNEL_REGISTRY_AUTH_TOKEN": "gQAAAAAABE78AAIgcDJjMGY1NDA1ZGZlMTM0M2NhOGZhODY0NzFhMjQ2OWVlMg",
        "SERVER_BEARER_TOKEN": "fb79377512ee73bee5a83a166db83e23ac4f8560603e1724fc03290cd41f0bc1",
        "KAGGLE_USERNAME": "avidok",
        "KAGGLE_KEY": "KGAT_81e3edec96ea3de2aa6af5dd839c735e",
        "KAGGLE_KERNEL_SLUG": "avidok/audiogen",
    }


# ==============================================================================
# Environment Variable Validation Tests
# ==============================================================================


def test_validate_env_vars_success(valid_env_dict: Dict[str, str]) -> None:
    """Verify validate_env_vars returns normalized dictionary when all variables are valid."""
    result = validate_env_vars(valid_env_dict)
    for var in REQUIRED_ENV_VARS:
        assert var in result
        assert result[var] == valid_env_dict[var]


@pytest.mark.parametrize("missing_var", REQUIRED_ENV_VARS)
def test_validate_env_vars_missing_required_variable(
    valid_env_dict: Dict[str, str], missing_var: str
) -> None:
    """Verify that omitting any required environment variable raises PreflightValidationError."""
    del valid_env_dict[missing_var]
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_env_vars(valid_env_dict)
    assert f"Missing required environment variable: '{missing_var}'" in str(exc_info.value)


@pytest.mark.parametrize("empty_var", REQUIRED_ENV_VARS)
def test_validate_env_vars_empty_or_whitespace_variable(
    valid_env_dict: Dict[str, str], empty_var: str
) -> None:
    """Verify that an empty or whitespace-only value raises PreflightValidationError."""
    valid_env_dict[empty_var] = "   "
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_env_vars(valid_env_dict)
    assert f"Environment variable '{empty_var}' is empty" in str(exc_info.value)


def test_validate_env_vars_invalid_webhook_url_scheme(valid_env_dict: Dict[str, str]) -> None:
    """Verify that non-http/https URL raises PreflightValidationError."""
    valid_env_dict["TUNNEL_REGISTRY_WEBHOOK_URL"] = "ftp://invalid-url.com"
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_env_vars(valid_env_dict)
    assert "Invalid URL structure" in str(exc_info.value)


def test_validate_env_vars_invalid_webhook_url_no_netloc(valid_env_dict: Dict[str, str]) -> None:
    """Verify that a malformed URL with no netloc raises PreflightValidationError."""
    valid_env_dict["TUNNEL_REGISTRY_WEBHOOK_URL"] = "http://"
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_env_vars(valid_env_dict)
    assert "Invalid URL structure" in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_slug",
    [
        "invalid_slug_without_slash",
        "/missing_owner",
        "missing_kernel/",
        "has spaces/slug",
        "special$char/slug",
    ],
)
def test_validate_env_vars_invalid_kaggle_kernel_slug(
    valid_env_dict: Dict[str, str], bad_slug: str
) -> None:
    """Verify that malformed Kaggle kernel slugs raise PreflightValidationError."""
    valid_env_dict["KAGGLE_KERNEL_SLUG"] = bad_slug
    with pytest.raises(PreflightValidationError) as exc_info:
        validate_env_vars(valid_env_dict)
    assert "Invalid format for KAGGLE_KERNEL_SLUG" in str(exc_info.value)


# ==============================================================================
# Live Health Probe Tests & Mandatory Header Mock Assertions
# ==============================================================================


def test_probe_tunnel_registry_with_auth_token_mock_assertion() -> None:
    """Verify probe forwards Authorization header and asserts on headers argument.

    Rule Guardrail: Mock assertions on the headers argument are mandatory for any
    new or modified HTTP client invocation.
    """
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"status": "ok"}'
    mock_client.get.return_value = mock_resp

    test_url = "https://registry.example.com/health"
    test_token = "mock-secret-auth-token"

    success = probe_tunnel_registry(url=test_url, auth_token=test_token, client=mock_client)
    assert success is True

    # Mandatory mock assertion on the headers argument
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert call_args.kwargs.get("headers") == {"Authorization": f"Bearer {test_token}"}
    assert call_args[0][0] == test_url


def test_probe_tunnel_registry_without_auth_token_mock_assertion() -> None:
    """Verify probe sends no Authorization header when auth_token is None or empty.

    Rule Guardrail: Mock assertions on the headers argument are mandatory.
    """
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"status": "ok"}'
    mock_client.get.return_value = mock_resp

    test_url = "https://registry.example.com/health"

    success = probe_tunnel_registry(url=test_url, auth_token=None, client=mock_client)
    assert success is True

    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert "headers" not in call_args.kwargs or call_args.kwargs.get("headers") is None


def test_probe_tunnel_registry_upstash_direct_ping_success() -> None:
    """Verify Upstash Redis REST bare domain probes /ping directly on first attempt without 400 EOF."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"result": "PONG"}'
    mock_client.get.return_value = mock_resp

    success = probe_tunnel_registry(
        url="https://large-pup-282364.upstash.io",
        auth_token="test-token",
        client=mock_client,
    )
    assert success is True
    assert mock_client.get.call_count == 1
    call_args = mock_client.get.call_args
    assert call_args[0][0] == "https://large-pup-282364.upstash.io/ping"
    assert call_args.kwargs.get("headers") == {"Authorization": "Bearer test-token"}


def test_probe_tunnel_registry_upstash_direct_get_tunnel_url_success() -> None:
    """Verify Upstash Redis REST explicit /get/tunnel_url path probes directly."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"result": "https://active.trycloudflare.com"}'
    mock_client.get.return_value = mock_resp

    success = probe_tunnel_registry(
        url="https://large-pup-282364.upstash.io/get/tunnel_url",
        auth_token="test-token-456",
        client=mock_client,
    )
    assert success is True
    assert mock_client.get.call_count == 1
    call_args = mock_client.get.call_args
    assert call_args[0][0] == "https://large-pup-282364.upstash.io/get/tunnel_url"
    assert call_args.kwargs.get("headers") == {"Authorization": "Bearer test-token-456"}


def test_probe_tunnel_registry_head_fallback_success() -> None:
    """Verify HEAD request fallback when GET / and GET /ping return non-200."""
    mock_client = MagicMock(spec=httpx.Client)

    def mock_get(target_url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 405
        resp.text = "Method Not Allowed"
        return resp

    mock_head_resp = MagicMock(spec=httpx.Response)
    mock_head_resp.status_code = 200
    mock_head_resp.text = ""

    mock_client.get.side_effect = mock_get
    mock_client.head.return_value = mock_head_resp

    success = probe_tunnel_registry(
        url="https://custom-webhook.com/endpoint",
        auth_token="test-token",
        client=mock_client,
    )
    assert success is True
    mock_client.head.assert_called_once()
    assert mock_client.head.call_args.kwargs.get("headers") == {"Authorization": "Bearer test-token"}


def test_probe_tunnel_registry_401_unauthorized_fails_fast() -> None:
    """Verify HTTP 401 raises PreflightValidationError with auth token failure message."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = '{"error": "WRONGPASS invalid or missing auth token"}'
    mock_client.get.return_value = mock_resp

    with pytest.raises(PreflightValidationError) as exc_info:
        probe_tunnel_registry(
            url="https://registry.example.com",
            auth_token="bad-token",
            client=mock_client,
        )
    assert "HTTP 401 Unauthorized" in str(exc_info.value)
    assert "TUNNEL_REGISTRY_AUTH_TOKEN" in str(exc_info.value)


def test_probe_tunnel_registry_403_forbidden_fails_fast() -> None:
    """Verify HTTP 403 raises PreflightValidationError with permission failure message."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden"
    mock_client.get.return_value = mock_resp

    with pytest.raises(PreflightValidationError) as exc_info:
        probe_tunnel_registry(
            url="https://registry.example.com",
            auth_token="restricted-token",
            client=mock_client,
        )
    assert "HTTP 403 Forbidden" in str(exc_info.value)
    assert "TUNNEL_REGISTRY_AUTH_TOKEN" in str(exc_info.value)


def test_probe_tunnel_registry_connection_error_fails_fast() -> None:
    """Verify connection errors raise PreflightValidationError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(PreflightValidationError) as exc_info:
        probe_tunnel_registry(
            url="https://registry.example.com",
            auth_token="token",
            client=mock_client,
        )
    assert "Failed to connect to TUNNEL_REGISTRY_WEBHOOK_URL" in str(exc_info.value)


# ==============================================================================
# Port Check Tests
# ==============================================================================


def test_check_port_open_success() -> None:
    """Verify check_port_open returns True when port can be bound."""
    # Find an available ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        available_port = s.getsockname()[1]

    assert check_port_open(port=available_port, host="127.0.0.1") is True


def test_check_port_open_occupied_raises_error() -> None:
    """Verify check_port_open raises PreflightValidationError when port is already in use."""
    # Bind an ephemeral port and keep it open
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        occupied_port = s.getsockname()[1]

        with pytest.raises(PreflightValidationError) as exc_info:
            check_port_open(port=occupied_port, host="127.0.0.1")
        assert f"Port conflict detected: Local port {occupied_port}" in str(exc_info.value)
        assert "BLOCKED / already in use" in str(exc_info.value)


def test_check_port_open_permission_denied_raises_error() -> None:
    """Verify check_port_open raises descriptive error on permission denial."""
    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = PermissionError(13, "Permission denied")
        mock_sock_cls.return_value.__enter__.return_value = mock_sock

        with pytest.raises(PreflightValidationError) as exc_info:
            check_port_open(port=80, host="127.0.0.1")
        assert "Permission denied attempting to bind port 80" in str(exc_info.value)


# ==============================================================================
# End-to-End CLI & Pre-flight Flow Tests
# ==============================================================================


def test_run_preflight_checks_success(monkeypatch: pytest.MonkeyPatch, valid_env_dict: Dict[str, str]) -> None:
    """Verify run_preflight_checks completes successfully when all components are valid."""
    for k, v in valid_env_dict.items():
        monkeypatch.setenv(k, v)

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"status": "ok"}'
    mock_client.get.return_value = mock_resp

    with patch("scripts.verify_env_and_auth.check_port_open", return_value=True):
        res = run_preflight_checks(client=mock_client, port=17000)
        assert res is True


def test_main_cli_exit_zero_on_success(monkeypatch: pytest.MonkeyPatch, valid_env_dict: Dict[str, str]) -> None:
    """Verify main() CLI returns exit code 0 when all checks pass."""
    for k, v in valid_env_dict.items():
        monkeypatch.setenv(k, v)

    with patch("scripts.verify_env_and_auth.probe_tunnel_registry", return_value=True), \
         patch("scripts.verify_env_and_auth.check_port_open", return_value=True):
        exit_code = main()
        assert exit_code == 0


def test_main_cli_exit_one_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() CLI returns exit code 1 when any pre-flight check fails."""
    # Wipe out all env vars
    monkeypatch.delenv("TUNNEL_REGISTRY_WEBHOOK_URL", raising=False)
    exit_code = main()
    assert exit_code == 1
