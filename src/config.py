"""Configuration engine for VoiceGen service platform.

Enforces port allocation strategy (17000-17099), environment precedence,
and operational guardrails.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Set

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

# Load .env file backing if present
load_dotenv()

MIN_PORT: int = 17000
MAX_PORT: int = 17099
FORBIDDEN_PORTS: Set[int] = {3000, 5000, 8000, 8080, 8090}


class Settings(BaseModel):
    """VoiceGen operational settings model."""

    app_name: str = "VoiceGen"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 17000
    worker_port: int = 17001
    metrics_port: int = 17002
    webhook_port: int = 17003
    websocket_port: int = 17004
    docs_port: int = 17005

    @model_validator(mode="before")
    @classmethod
    def load_env_vars_and_precedence(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            data = {}
        else:
            data = dict(data)

        # Application metadata
        if "app_name" not in data or data["app_name"] is None:
            data["app_name"] = os.getenv("APP_NAME", "VoiceGen")
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

        return data

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
