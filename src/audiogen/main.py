"""AudioGen application entry point, health probe, and UI/synthesis endpoints."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import functools
import io
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf

from audiogen.config import MissingKaggleCredentialsError, Settings, get_settings
import voices.registry

try:
    from audiogen.engine import (
        KaggleExecutionBridge,
        KaggleExecutionError,
        KaggleTimeoutError,
        Synthesizer,
    )
except ImportError:
    Synthesizer = None  # type: ignore
    KaggleExecutionBridge = None  # type: ignore
    KaggleExecutionError = RuntimeError  # type: ignore
    KaggleTimeoutError = TimeoutError  # type: ignore

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

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
UI_DIR: Path = Path(__file__).resolve().parent / "ui"
VOICES_DIR: Path = (REPO_ROOT / "voices").resolve()

DEFAULT_ENGLISH_VOICES: Dict[str, Dict[str, Any]] = {
    "narrator_english_neutral": {
        "id": "narrator_english_neutral",
        "speaker_ref_name": "narrator_english_neutral",
        "name": "Narrator English Neutral",
        "language": ["en"],
        "description": "Clear neutral studio voice for English narration and instructional audio",
        "ref_text": "Welcome to AudioGen speech synthesis platform.",
        "path": "voices/refs/anchor_female_calm.wav",
    },
    "anchor_english_energetic": {
        "id": "anchor_english_energetic",
        "speaker_ref_name": "anchor_english_energetic",
        "name": "Anchor English Energetic",
        "language": ["en"],
        "description": "Energetic broadcast voice suitable for English announcements",
        "ref_text": "Welcome to the latest edition of AudioGen audio generation.",
        "path": "voices/refs/anchor_male_energetic.wav",
    },
}


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


def get_manifest_voices() -> Dict[str, Dict[str, Any]]:
    """Loads voices from the voice registry manifest with structured error handling."""
    try:
        raw_manifest = voices.registry.load_manifest()
    except Exception as exc:
        manifest_err = emit_error_manifest(exc, context="voice_manifest_loading")
        logger.error("Internal failure loading voice registry manifest: %s", manifest_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: Voice manifest loading failed: {exc}",
        )

    manifest_voices: Dict[str, Dict[str, Any]] = {}
    for v_id, meta in raw_manifest.items():
        manifest_voices[v_id] = {
            "id": v_id,
            "speaker_ref_name": v_id,
            "name": v_id.replace("_", " ").title(),
            "language": [str(l).strip().lower() for l in meta.get("language", [])],
            "description": meta.get("description", ""),
            "ref_text": meta.get("ref_text", ""),
            "path": meta.get("path", ""),
        }
    return manifest_voices


def get_all_registered_voices() -> Dict[str, Dict[str, Any]]:
    """Retrieves all registered voices in unified precedence order:

    1. voices.registry manifest
    2. DEFAULT_ENGLISH_VOICES
    3. app.state.custom_voices
    """
    all_voices: Dict[str, Dict[str, Any]] = {}

    # 1. Manifest voices
    all_voices.update(get_manifest_voices())

    # 2. Built-in default English voices
    for v_id, meta in DEFAULT_ENGLISH_VOICES.items():
        if v_id not in all_voices:
            all_voices[v_id] = dict(meta)

    # 3. Dynamic custom voices
    custom_map = getattr(app.state, "custom_voices", {})
    for v_id, meta in custom_map.items():
        all_voices[v_id] = dict(meta)

    return all_voices


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with structured error trapping."""
    try:
        # Enforce configuration validation at startup
        settings = get_settings()
        if not hasattr(app.state, "custom_voices"):
            app.state.custom_voices = {}
        if not hasattr(app.state, "inference_lock") or app.state.inference_lock is None:
            app.state.inference_lock = asyncio.Lock()

        # Check Kaggle credentials availability
        missing_creds = settings.get_missing_kaggle_credentials()
        app.state.missing_kaggle_credentials = missing_creds
        if missing_creds:
            logger.warning(
                "Kaggle API credentials not configured in environment (%s). Synthesis endpoint will require credentials.",
                ", ".join(missing_creds),
            )
        else:
            logger.info("Kaggle API credentials successfully verified.")

        # Initialize Synthesizer in production if available
        if not hasattr(app.state, "synthesizer") or app.state.synthesizer is None:
            if Synthesizer is not None:
                try:
                    app.state.synthesizer = Synthesizer()
                    logger.info("Synthesizer successfully initialized.")
                except Exception as exc:
                    logger.info("Synthesizer model auto-load skipped (fallback available): %s", exc)
                    app.state.synthesizer = None

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

# Initialize application runtime state
app.state.custom_voices = {}
app.state.inference_lock = asyncio.Lock()
app.state.synthesizer = None
app.state.missing_kaggle_credentials = []

# Mount static asset directory
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# Pydantic Schemas for UI API
class VoiceCreateRequest(BaseModel):
    """Payload schema for creating a new voice profile."""

    id: str = Field(..., description="Unique voice identifier")
    name: Optional[str] = Field(default=None, description="Display name for the voice profile")
    language: Union[List[str], str] = Field(..., description="Language codes supported (e.g. ['en'] or 'hi')")
    description: Optional[str] = Field(default=None, description="Short voice description")
    ref_text: str = Field(..., description="Reference audio transcript")
    path: Optional[str] = Field(default=None, description="Path to reference audio file in voices/ directory")


class GenerateRequest(BaseModel):
    """Payload schema for audio synthesis request."""

    text: str = Field(..., description="Target text to synthesize into speech")
    language: Optional[str] = Field(default="en", description="Target language code (e.g. 'en', 'hi', 'pa')")
    speaker_ref_name: Optional[str] = Field(default=None, description="Voice profile identifier")
    voice: Optional[str] = Field(default=None, description="Alternative voice identifier parameter")
    voice_id: Optional[str] = Field(default=None, description="Alternative voice identifier parameter")


@app.get("/", response_class=FileResponse)
async def serve_ui() -> FileResponse:
    """Serve the minimalist dark web UI interface."""
    index_path = UI_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AudioGen UI index template not found.",
        )
    return FileResponse(index_path, media_type="text/html")


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


@app.get("/api/voices")
async def list_api_voices(language: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available voices using unified precedence, with optional language filtering."""
    voices_map = get_all_registered_voices()
    all_voices = list(voices_map.values())

    if language is not None and language.strip():
        target_lang = language.strip().lower()
        all_voices = [
            v for v in all_voices
            if target_lang in [str(item).lower() for item in v.get("language", [])]
        ]

    return all_voices


@app.post("/api/voices", status_code=status.HTTP_201_CREATED)
async def create_api_voice(payload: VoiceCreateRequest) -> Dict[str, Any]:
    """Register a new custom voice profile in memory with strict path sanitization."""
    v_id = payload.id.strip()
    if not v_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice identifier cannot be empty.",
        )

    if not payload.ref_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reference text cannot be empty.",
        )

    if isinstance(payload.language, list):
        langs = [str(l).strip().lower() for l in payload.language if str(l).strip()]
    else:
        langs = [payload.language.strip().lower()] if payload.language.strip() else []

    if not langs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one language must be specified.",
        )

    display_name = payload.name.strip() if (payload.name and payload.name.strip()) else v_id.replace("_", " ").title()
    desc = payload.description.strip() if (payload.description and payload.description.strip()) else f"Custom voice {display_name}"

    # Path sanitization and traversal prevention
    if payload.path:
        raw_path = Path(payload.path.strip())
        candidate = (REPO_ROOT / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()

        # Enforce that path resides within VOICES_DIR
        try:
            candidate.relative_to(VOICES_DIR)
        except ValueError:
            logger.warning("Path traversal attempt detected in voice creation: %s", payload.path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path traversal violation: Voice reference audio path must be inside the approved voices/ directory.",
            )

        # Enforce .wav extension
        if candidate.suffix.lower() != ".wav":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio format: Reference audio must be a .wav file.",
            )

        # Verify on-disk file existence
        if not candidate.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Reference audio file does not exist on disk: {payload.path}",
            )

        stored_path = str(candidate.relative_to(REPO_ROOT))
    else:
        stored_path = "voices/refs/anchor_female_calm.wav"

    voice_entry = {
        "id": v_id,
        "speaker_ref_name": v_id,
        "name": display_name,
        "language": langs,
        "description": desc,
        "ref_text": payload.ref_text.strip(),
        "path": stored_path,
    }

    if not hasattr(app.state, "custom_voices"):
        app.state.custom_voices = {}

    app.state.custom_voices[v_id] = voice_entry
    return voice_entry


@app.post(
    "/api/generate",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized audio binary (WAV format)",
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def generate_api_audio(payload: GenerateRequest) -> Response:
    """Synthesize text into speech audio waveform."""
    # 1. Validate text
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty or whitespace-only.",
        )

    # 2. Resolve voice parameter
    voice_name = payload.speaker_ref_name or payload.voice or payload.voice_id
    if not voice_name or not voice_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice profile must be specified.",
        )
    voice_name = voice_name.strip()

    # 3. Validate language
    lang = (payload.language or "").strip().lower()
    if not lang:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Language must be specified.",
        )

    # 4. Find voice record using unified precedence
    voices_map = get_all_registered_voices()
    if voice_name not in voices_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown voice name: {voice_name}",
        )
    found_voice = voices_map[voice_name]

    supported_langs = [str(item).strip().lower() for item in found_voice.get("language", [])]
    if lang not in supported_langs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice '{voice_name}' does not support language '{lang}'. Supported languages: {supported_langs}",
        )

    # 5. Validate on-disk existence of reference audio file
    raw_ref_path = found_voice.get("path")
    if not raw_ref_path:
        logger.error("Voice record '%s' missing reference audio path.", voice_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server configuration error: Reference audio file path for voice '{voice_name}' is empty.",
        )

    ref_audio_file = Path(raw_ref_path)
    if not ref_audio_file.is_absolute():
        ref_audio_file = (REPO_ROOT / ref_audio_file).resolve()

    if not ref_audio_file.is_file():
        logger.error("Reference audio file does not exist on disk: %s", ref_audio_file)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server configuration error: Reference audio file for voice '{voice_name}' does not exist on disk: {ref_audio_file}",
        )

    ref_text = found_voice.get("ref_text", "")
    if not ref_text or not ref_text.strip():
        logger.error("Voice record '%s' missing reference transcript.", voice_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server configuration error: Reference text is missing for voice '{voice_name}'.",
        )

    # 6. Resolve synthesizer or enforce Kaggle GPU execution credentials
    settings = get_settings()
    synthesizer = getattr(app.state, "synthesizer", None)
    if synthesizer is None:
        missing_creds = settings.get_missing_kaggle_credentials()
        if missing_creds:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=settings.format_missing_credentials_message(missing_creds),
            )
        bridge = getattr(app.state, "kaggle_bridge", None)
        if bridge is None:
            bridge = KaggleExecutionBridge(settings=settings)
            app.state.kaggle_bridge = bridge
        synthesis_runner = functools.partial(
            bridge.synthesize,
            payload.text.strip(),
            ref_audio_path=str(ref_audio_file),
            ref_text=ref_text.strip(),
            language=lang,
        )
    else:
        synthesis_runner = functools.partial(
            synthesizer.synthesize,
            payload.text.strip(),
            ref_audio_path=str(ref_audio_file),
            ref_text=ref_text.strip(),
            language=lang,
        )

    inference_lock = getattr(app.state, "inference_lock", None)
    if inference_lock is None:
        app.state.inference_lock = asyncio.Lock()
        inference_lock = app.state.inference_lock

    try:
        async with inference_lock:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, synthesis_runner)
            if isinstance(result, tuple):
                waveform, sr = result
            else:
                waveform, sr = result, 24000

        buf = io.BytesIO()
        sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except MissingKaggleCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except (KaggleExecutionError, KaggleTimeoutError) as exc:
        manifest = emit_error_manifest(exc, context="kaggle_execution_failure")
        logger.error("Kaggle execution failure: %s", manifest)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kaggle execution failure: {exc}",
        )
    except Exception as exc:
        manifest = emit_error_manifest(exc, context="synthesis_engine_inference")
        logger.error("Synthesis engine failure: %s", manifest)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis engine failure: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("audiogen.main:app", host=settings.host, port=settings.port, reload=False)
