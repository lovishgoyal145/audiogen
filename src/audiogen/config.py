"""Configuration engine for AudioGen service platform.

Enforces port allocation strategy (17000-17099), environment precedence,
and operational guardrails.
"""

from __future__ import annotations

import functools
import os
from typing import Any, List, Optional, Set

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

# Load .env file backing if present
load_dotenv()

MIN_PORT: int = 17000
MAX_PORT: int = 17099
FORBIDDEN_PORTS: Set[int] = {3000, 5000, 8000, 8080, 8090}


class MissingKaggleCredentialsError(ValueError):
    """Raised when required Kaggle API credentials are not provided in environment."""

    pass


class Settings(BaseModel):
    """AudioGen operational settings model."""

    app_name: str = "AudioGen"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 17000
    worker_port: int = 17001
    metrics_port: int = 17002
    webhook_port: int = 17003
    websocket_port: int = 17004
    docs_port: int = 17005
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    kaggle_kernel_slug: Optional[str] = "avidok/vco-worker"

    @model_validator(mode="before")
    @classmethod
    def load_env_vars_and_precedence(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            data = {}
        else:
            data = dict(data)

        # Application metadata
        if "app_name" not in data or data["app_name"] is None:
            data["app_name"] = os.getenv("APP_NAME", "AudioGen")
        if "environment" not in data or data["environment"] is None:
            data["environment"] = os.getenv("ENVIRONMENT", "development")

        # Host configuration:
        # 1. Direct model arg 'host' if provided
        # 2. Environment variable 'HOST'
        # 3. Environment variable 'APP_HOST'
        # 4. Default '127.0.0.1'
        if "host" not in data or data["host"] is None:
            if os.getenv("HOST") is not None and os.getenv("HOST") != "":
                data["host"] = os.getenv("HOST")
            elif os.getenv("APP_HOST") is not None and os.getenv("APP_HOST") != "":
                data["host"] = os.getenv("APP_HOST")
            else:
                data["host"] = "127.0.0.1"

        # Port strategy & precedence:
        # 1. Direct model arg 'port' if provided
        # 2. Direct model arg 'app_port' if provided
        # 3. Environment variable 'PORT'
        # 4. Environment variable 'APP_PORT'
        # 5. Default 17000
        if "port" not in data or data["port"] is None:
            if "app_port" in data and data["app_port"] is not None:
                data["port"] = data["app_port"]
            elif os.getenv("PORT") is not None and os.getenv("PORT") != "":
                data["port"] = os.getenv("PORT")
            elif os.getenv("APP_PORT") is not None and os.getenv("APP_PORT") != "":
                data["port"] = os.getenv("APP_PORT")
            else:
                data["port"] = 17000

        # Secondary port slots (17001-17009 block)
        secondary_slots = [
            ("worker_port", "WORKER_PORT", 17001),
            ("metrics_port", "METRICS_PORT", 17002),
            ("webhook_port", "WEBHOOK_PORT", 17003),
            ("websocket_port", "WEBSOCKET_PORT", 17004),
            ("docs_port", "DOCS_PORT", 17005),
        ]
        for field_name, env_var, default_port in secondary_slots:
            if field_name not in data or data[field_name] is None:
                val = os.getenv(env_var)
                data[field_name] = val if (val is not None and val != "") else default_port

        # Kaggle credentials and kernel configuration precedence
        if "kaggle_username" not in data or data["kaggle_username"] is None:
            data["kaggle_username"] = os.getenv("KAGGLE_USERNAME")
        if "kaggle_key" not in data or data["kaggle_key"] is None:
            data["kaggle_key"] = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_KEY")
        if "kaggle_kernel_slug" not in data or data["kaggle_kernel_slug"] is None:
            val = os.getenv("KAGGLE_KERNEL_SLUG") or os.getenv("KAGGLE_KERNEL_ID")
            data["kaggle_kernel_slug"] = val if (val is not None and val != "") else "avidok/vco-worker"

        return data

    def get_missing_kaggle_credentials(self) -> List[str]:
        """Return list of missing required Kaggle credential names."""
        missing: List[str] = []
        if not self.kaggle_username or not str(self.kaggle_username).strip():
            missing.append("KAGGLE_USERNAME")
        if not self.kaggle_key or not str(self.kaggle_key).strip():
            missing.append("KAGGLE_KEY")
        if not self.kaggle_kernel_slug or not str(self.kaggle_kernel_slug).strip():
            missing.append("KAGGLE_KERNEL_SLUG")
        return missing

    def format_missing_credentials_message(self, missing: Optional[List[str]] = None) -> str:
        """Format the canonical error message for missing Kaggle credentials."""
        if missing is None:
            missing = self.get_missing_kaggle_credentials()
        return f"Missing required Kaggle API credentials in .env: {', '.join(missing)}"

    def validate_kaggle_credentials(self) -> None:
        """Validate required Kaggle credentials or raise MissingKaggleCredentialsError."""
        missing = self.get_missing_kaggle_credentials()
        if missing:
            raise MissingKaggleCredentialsError(self.format_missing_credentials_message(missing))

    @field_validator(
        "port",
        "worker_port",
        "metrics_port",
        "webhook_port",
        "websocket_port",
        "docs_port",
        mode="after",
    )
    @classmethod
    def validate_port_allocation(cls, v: int, info: ValidationInfo) -> int:
        if v in FORBIDDEN_PORTS or not (MIN_PORT <= v <= MAX_PORT):
            raise ValueError(
                f"Operational port violation: {info.field_name}={v} is outside allowed range "
                f"{MIN_PORT}–{MAX_PORT} or matches strictly forbidden standard ports "
                f"({', '.join(str(p) for p in sorted(FORBIDDEN_PORTS))})."
            )
        return v

    @model_validator(mode="after")
    def validate_port_collisions(self) -> Settings:
        service_ports = {
            "port": self.port,
            "worker_port": self.worker_port,
            "metrics_port": self.metrics_port,
            "webhook_port": self.webhook_port,
            "websocket_port": self.websocket_port,
            "docs_port": self.docs_port,
        }
        seen_ports: dict[int, str] = {}
        for service_name, port_val in service_ports.items():
            if port_val in seen_ports:
                raise ValueError(
                    f"Operational port collision: {service_name} ({port_val}) "
                    f"collides with {seen_ports[port_val]} ({port_val}). "
                    f"All active service ports must be distinct."
                )
            seen_ports[port_val] = service_name
        return self


@functools.lru_cache()
def get_settings() -> Settings:
    """Singleton factory for cached Settings instance."""
    return Settings()


def validate_kaggle_credentials(settings: Optional[Settings] = None) -> Settings:
    """Validate that required Kaggle API credentials are present in Settings or raise MissingKaggleCredentialsError."""
    if settings is None:
        settings = get_settings()
    settings.validate_kaggle_credentials()
    return settings
