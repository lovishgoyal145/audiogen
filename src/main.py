"""AudioGen application entry point & health probe service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import sys
import traceback
from typing import Any, Dict

from fastapi import Depends, FastAPI

from src.config import Settings, get_settings

# Configure structured logging to sys.stderr (Rule 4)
logger = logging.getLogger("audiogen")
if not logger.handlers:
    stderr_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)


def emit_error_manifest(exc: Exception, context: str = "lifespan_startup") -> Dict[str, Any]:
    """Logs tracebacks to stderr and emits a structured error manifest."""
    manifest = {
        "event": "error_manifest",
        "context": context,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_json = json.dumps(manifest)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.write(f"STRUCTURED_ERROR_MANIFEST: {manifest_json}\n")
    sys.stderr.flush()
    logger.error("Startup failure error manifest: %s", manifest_json)
    return manifest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with structured error trapping."""
    try:
        # Enforce configuration validation at startup
        settings = get_settings()
        logger.info(
            "Service initialized: %s on %s:%s (env: %s)",
            settings.app_name,
            settings.host,
            settings.port,
            settings.environment,
        )
    except Exception as exc:
        manifest = emit_error_manifest(exc, context="lifespan_startup")
        app.state.startup_error = manifest
        raise exc
    yield


app = FastAPI(
    title="AudioGen API",
    version="0.1.0",
    description="High-performance audio generation and operational service platform",
    lifespan=lifespan,
)


@app.get("/healthz")
async def health_check(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Health diagnostic endpoint reporting status and assigned port."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "host": settings.host,
        "port": settings.port,
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=False)

