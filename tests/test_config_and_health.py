"""Tests for VoiceGen configuration engine, port allocation guardrails, and health probe."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import FORBIDDEN_PORTS, Settings, get_settings
from src.main import app


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure a clean environment for each test and clear cached settings."""
    for key in [
        "HOST",
        "APP_HOST",
        "PORT",
        "APP_PORT",
        "APP_NAME",
        "ENVIRONMENT",
        "WORKER_PORT",
        "METRICS_PORT",
        "WEBHOOK_PORT",
        "WEBSOCKET_PORT",
        "DOCS_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    app.dependency_overrides.clear()
    yield
    get_settings.cache_clear()
    app.dependency_overrides.clear()


def test_default_port_assignment():
    """Test 1: Default Port Assignment without env variables defaults to 17000."""
    settings = Settings()
    assert settings.port == 17000
    assert settings.host == "127.0.0.1"
    assert settings.app_name == "VoiceGen"
    assert settings.environment == "development"
    assert settings.worker_port == 17001
    assert settings.metrics_port == 17002
    assert settings.webhook_port == 17003
    assert settings.websocket_port == 17004
    assert settings.docs_port == 17005


def test_host_configuration_precedence(monkeypatch):
    """Test host configuration from default, HOST, and APP_HOST."""
    # 1. HOST env variable override
    monkeypatch.setenv("HOST", "0.0.0.0")
    settings = Settings()
    assert settings.host == "0.0.0.0"

    # 2. APP_HOST env variable override when HOST is absent
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("APP_HOST", "192.168.1.100")
    settings = Settings()
    assert settings.host == "192.168.1.100"

    # 3. HOST takes precedence over APP_HOST
    monkeypatch.setenv("HOST", "10.0.0.1")
    monkeypatch.setenv("APP_HOST", "10.0.0.2")
    settings = Settings()
    assert settings.host == "10.0.0.1"


def test_environment_variable_precedence(monkeypatch):
    """Test 2: Environment Variable Precedence (PORT overrides APP_PORT, etc.)."""
    # 1. PORT=17050 yields 17050
    monkeypatch.setenv("PORT", "17050")
    settings = Settings()
    assert settings.port == 17050

    # 2. APP_PORT=17060 yields 17060 when PORT is absent
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("APP_PORT", "17060")
    settings = Settings()
    assert settings.port == 17060

    # 3. PORT=17010 overrides APP_PORT=17020
    monkeypatch.setenv("PORT", "17010")
    monkeypatch.setenv("APP_PORT", "17020")
    settings = Settings()
    assert settings.port == 17010


@pytest.mark.parametrize(
    "invalid_port", [16999, 8000, 8080, 5000, 3000, 17100, 18000]
)
def test_port_range_boundary_enforcement(invalid_port):
    """Test 3: Port Range Boundary Enforcement for ports < 17000 or > 17099."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        Settings(port=invalid_port)
    assert "Operational port violation" in str(exc_info.value)


@pytest.mark.parametrize("forbidden_port", sorted(FORBIDDEN_PORTS))
def test_forbidden_standard_ports_rejected_in_env(monkeypatch, forbidden_port):
    """Test standard forbidden ports explicitly rejected when loaded via environment variable."""
    monkeypatch.setenv("PORT", str(forbidden_port))
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        Settings()
    assert "Operational port violation" in str(exc_info.value)


def test_port_collision_validation():
    """Test root validator prevents active service port collisions."""
    # Collision between primary port and worker_port (both 17001)
    with pytest.raises(ValidationError) as exc_info:
        Settings(port=17001, worker_port=17001)
    assert "Operational port collision" in str(exc_info.value)

    # Collision between secondary ports (worker_port and metrics_port both 17020)
    with pytest.raises(ValidationError) as exc_info:
        Settings(worker_port=17020, metrics_port=17020)
    assert "Operational port collision" in str(exc_info.value)


def test_port_collision_via_env_vars(monkeypatch):
    """Test port collision triggered via environment variables is rejected."""
    monkeypatch.setenv("PORT", "17002")  # Collides with default metrics_port=17002
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    assert "Operational port collision" in str(exc_info.value)


def test_health_probe_http_contract():
    """Test 4: Health Probe HTTP Contract."""
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["service"] == "VoiceGen"
        assert payload["host"] == "127.0.0.1"
        assert payload["port"] == 17000
        assert payload["environment"] == "development"


def test_health_probe_dynamic_configuration(monkeypatch):
    """Test 5: Health Probe Dynamic Configuration updates /healthz payload faithfully."""
    monkeypatch.setenv("PORT", "17042")
    monkeypatch.setenv("HOST", "0.0.0.0")
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["service"] == "VoiceGen"
        assert payload["host"] == "0.0.0.0"
        assert payload["port"] == 17042
        assert payload["environment"] == "development"


def test_secondary_port_slots_env_override(monkeypatch):
    """Test secondary port slots override via environment variables."""
    monkeypatch.setenv("WORKER_PORT", "17011")
    monkeypatch.setenv("METRICS_PORT", "17012")
    monkeypatch.setenv("WEBHOOK_PORT", "17013")
    monkeypatch.setenv("WEBSOCKET_PORT", "17014")
    monkeypatch.setenv("DOCS_PORT", "17015")
    settings = Settings()
    assert settings.worker_port == 17011
    assert settings.metrics_port == 17012
    assert settings.webhook_port == 17013
    assert settings.websocket_port == 17014
    assert settings.docs_port == 17015


@pytest.mark.parametrize(
    "field,invalid_port",
    [
        ("worker_port", 8000),
        ("metrics_port", 17100),
        ("webhook_port", 3000),
        ("websocket_port", 16999),
        ("docs_port", 8080),
    ],
)
def test_secondary_ports_boundary_enforcement(field, invalid_port):
    """Test secondary port slots reject forbidden and out-of-range ports."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        Settings(**{field: invalid_port})
    assert "Operational port violation" in str(exc_info.value)


def test_lifespan_startup_failure_trapping(monkeypatch, capsys):
    """Test Rule 4: Lifespan startup error trapping logs traceback and emits structured error manifest."""
    monkeypatch.setenv("PORT", "8000")  # Forbidden port triggers startup error
    get_settings.cache_clear()

    with pytest.raises((ValidationError, ValueError)):
        with TestClient(app) as client:
            pass

    captured = capsys.readouterr()
    assert "STRUCTURED_ERROR_MANIFEST:" in captured.err
    assert "Operational port violation" in captured.err
    assert "Traceback (most recent call last):" in captured.err


