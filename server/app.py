"""FastAPI service exposing Indic speech synthesis with serialized inference and bearer auth."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from typing import Any, AsyncGenerator, Dict, Final, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

# Attempt import from audiogen or core fallback
try:
    from audiogen.engine import Synthesizer
except ImportError:
    from core.engine import Synthesizer  # type: ignore

import voices.registry
from server.watchdog import IdleWatchdog

logger = logging.getLogger("audiogen.server")

DEFAULT_BEARER_ENV_VAR: Final[str] = "SERVER_BEARER_TOKEN"
FALLBACK_SECRET_ENV_VAR: Final[str] = "SHARED_SECRET"
GENERATED_DIR_NAME: Final[str] = "audiogen_generated"
MAX_RETAINED_GENERATED_FILES: Final[int] = 50
MAX_GENERATED_FILE_AGE_SECONDS: Final[float] = 3600.0  # 1 hour


def cleanup_generated_files(
    target_dir: Path,
    max_files: int = MAX_RETAINED_GENERATED_FILES,
    max_age_seconds: float = MAX_GENERATED_FILE_AGE_SECONDS,
) -> None:
    """Prune generated audio files exceeding retention age or count limits."""
    try:
        if not target_dir.exists() or not target_dir.is_dir():
            return

        now = time.time()
        files = [p for p in target_dir.glob("*.wav") if p.is_file()]

        # 1. Delete files older than max_age_seconds
        remaining_files: List[Path] = []
        for f in files:
            try:
                mtime = f.stat().st_mtime
                if now - mtime > max_age_seconds:
                    f.unlink(missing_ok=True)
                else:
                    remaining_files.append(f)
            except OSError:
                pass

        # 2. If remaining files meet or exceed max_files, delete oldest to stay within limit
        if len(remaining_files) >= max_files:
            remaining_files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            excess = (len(remaining_files) - max_files) + 1
            for f in remaining_files[:excess]:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
    except Exception as exc:
        logger.warning("Error during generated files cleanup: %s", exc)


class GenerateRequest(BaseModel):
    """Payload schema for /generate endpoint."""

    text: str = Field(..., description="Text in Hindi or Punjabi to synthesize")
    language: str = Field(..., description="Language code ('hi' or 'pa')")
    speaker_ref_name: str = Field(..., description="Voice name matching voices registry manifest")
    return_uri: bool = Field(default=False, description="If True, returns file URI instead of binary audio")


class HealthResponse(BaseModel):
    """Payload schema for /health endpoint."""

    status: str
    gpu: bool
    idle_seconds_remaining: int


class ErrorResponse(BaseModel):
    """Error payload structure."""

    detail: str


def check_gpu_available() -> bool:
    """Check whether CUDA GPU is available for inference."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def verify_bearer_token(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_server_secret: Optional[str] = Header(None, alias="X-Server-Secret"),
) -> str:
    """Dependency verifying bearer token or secret header.

    Raises:
        HTTPException(status_code=401): If token is missing, invalid, or mismatched.
    """
    provided_token: Optional[str] = None
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided_token = parts[1]
        elif len(parts) == 1:
            provided_token = parts[0]
    elif x_server_secret:
        provided_token = x_server_secret.strip()

    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing authentication token",
        )

    expected_token = getattr(request.app.state, "auth_token", None)
    if not expected_token:
        expected_token = os.environ.get(DEFAULT_BEARER_ENV_VAR) or os.environ.get(
            FALLBACK_SECRET_ENV_VAR
        )

    if not expected_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid authentication token",
        )

    return provided_token


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan managing background watchdog and resource teardown."""
    watchdog = getattr(app.state, "watchdog", None)
    if watchdog is not None and not watchdog.is_alive:
        watchdog.start()
    yield
    if watchdog is not None and watchdog.is_alive:
        watchdog.stop()


def create_app(
    synthesizer: Optional[Synthesizer] = None,
    watchdog: Optional[IdleWatchdog] = None,
    auth_token: Optional[str] = None,
) -> FastAPI:
    """Factory creating and configuring the FastAPI app with dependency injection for testing."""
    app = FastAPI(title="AudioGen API", lifespan=lifespan)

    if synthesizer is None:
        try:
            app.state.synthesizer = Synthesizer()
        except Exception:
            app.state.synthesizer = None
    else:
        app.state.synthesizer = synthesizer

    app.state.watchdog = watchdog
    app.state.auth_token = auth_token
    app.state.inference_lock = asyncio.Lock()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Unauthenticated health check returning GPU status and idle seconds remaining."""
        idle_remaining = 600
        active_watchdog = getattr(app.state, "watchdog", None)
        if active_watchdog is not None:
            idle_remaining = active_watchdog.get_idle_seconds_remaining()

        return HealthResponse(
            status="online",
            gpu=check_gpu_available(),
            idle_seconds_remaining=idle_remaining,
        )

    @app.post(
        "/generate",
        responses={
            200: {
                "content": {
                    "audio/wav": {},
                    "application/json": {},
                },
                "description": "Synthesized audio binary or file URI",
            },
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def generate(
        payload: GenerateRequest,
        request: Request,
        _token: str = Depends(verify_bearer_token),
    ) -> Response:
        """Protected endpoint synthesizing Hindi/Punjabi text with serialized inference."""
        active_watchdog = getattr(app.state, "watchdog", None)
        if active_watchdog is not None:
            active_watchdog.touch()

        # 1. Validate text
        if not payload.text or not payload.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty or whitespace-only.",
            )

        # 2. Validate language
        lang = payload.language.strip().lower()
        if lang not in ("hi", "pa"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language: {payload.language}",
            )

        # 3. Resolve voice reference via voice registry
        try:
            ref_audio_path = voices.registry.get_voice_ref(payload.speaker_ref_name)
        except (voices.registry.VoiceNotFoundError, KeyError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown voice name: {payload.speaker_ref_name}",
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Reference audio file for voice '{payload.speaker_ref_name}' does not exist on disk",
            )

        # 4. Perform serialized inference
        inference_engine = getattr(app.state, "synthesizer", None)
        if inference_engine is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failure: Synthesizer is not initialized.",
            )

        inference_lock: asyncio.Lock = app.state.inference_lock
        try:
            async with inference_lock:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    inference_engine.synthesize,
                    payload.text,
                    lang,
                    ref_audio_path,
                )
                if isinstance(result, tuple):
                    waveform, sample_rate = result
                else:
                    waveform, sample_rate = result, 24000
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failure: {exc}",
            )

        # Touch watchdog on successful completion
        if active_watchdog is not None:
            active_watchdog.touch()

        # 5. Format response
        if payload.return_uri:
            out_dir = Path(tempfile.gettempdir()) / GENERATED_DIR_NAME
            out_dir.mkdir(parents=True, exist_ok=True)
            cleanup_generated_files(out_dir)
            out_file = out_dir / f"synth_{os.urandom(6).hex()}.wav"
            sf.write(str(out_file), np.asarray(waveform, dtype=np.float32), int(sample_rate), subtype="PCM_16")
            return JSONResponse({"file_uri": out_file.resolve().as_uri()})

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(waveform, dtype=np.float32), int(sample_rate), format="WAV", subtype="PCM_16")
        return Response(content=buffer.getvalue(), media_type="audio/wav")

    return app


app: FastAPI = create_app()
