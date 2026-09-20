"""FastAPI Gateway service for manual GPU session control and request proxying.

Implements the strict 4-state lifecycle: IDLE, STARTING, READY, ERROR.
Per project rules and TICKET-005:
- Starts sessions MANUALLY via POST /session/start.
- /generate rejects requests with HTTP 409 if session is not READY.
- /generate never auto-triggers session startup.
- Reads voices from voices.registry (read-only import).
- Reads tunnel URL from registry webhook/KV store.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, AsyncGenerator, Callable, Dict, Final, List, Optional
import httpx
from pydantic import BaseModel, Field
import yaml

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator import session_manager
from orchestrator.session_manager import (
    TERMINAL_FAILURE_STATUSES,
    KagglePushError,
    KaggleStatusError,
    _kaggle_push,
    get_kaggle_status,
    is_tunnel_healthy,
)
import voices.registry

logger = logging.getLogger("audiogen.orchestrator.gateway")

DEFAULT_CONFIG_PATH: Final[Path] = REPO_ROOT / "config" / "orchestrator_config.yaml"
UI_INDEX_PATH: Final[Path] = REPO_ROOT / "ui" / "index.html"

ENV_BEARER_TOKEN: Final[str] = "SERVER_BEARER_TOKEN"
ENV_SHARED_SECRET: Final[str] = "SHARED_SECRET"
ENV_REGISTRY_URL: Final[str] = "TUNNEL_REGISTRY_WEBHOOK_URL"


class StatusResponse(BaseModel):
    """Schema for /session/start and /session/status responses."""

    status: str
    message: str


class GeneratePayload(BaseModel):
    """Payload schema for POST /generate."""

    text: str = Field(..., description="Text to synthesize")
    language: str = Field(..., description="Language code ('en', 'hi', or 'pa')")
    speaker_ref_name: str = Field(..., description="Voice name matching voices registry")
    return_uri: bool = Field(default=False, description="If True, returns file URI instead of binary audio")


class SessionGateway:
    """Manages the in-memory state machine, background poller, and proxying."""

    def __init__(
        self,
        poll_interval_seconds: float = 5.0,
        startup_timeout_seconds: float = 600.0,
        registry_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        push_fn: Optional[Callable[..., None]] = None,
        status_fn: Optional[Callable[..., str]] = None,
        health_fn: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.registry_url = registry_url
        self.bearer_token = bearer_token
        self._push_fn = push_fn
        self._status_fn = status_fn
        self._health_fn = health_fn

        self._state: str = "IDLE"
        self._status_message: str = "Idle"
        self._tunnel_url: Optional[str] = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._polling_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def tunnel_url(self) -> Optional[str]:
        return self._tunnel_url

    def reset(self) -> None:
        """Reset state machine to IDLE (primarily for test fixture isolation)."""
        if self._polling_task is not None and not self._polling_task.done():
            self._polling_task.cancel()
        self._polling_task = None
        self._state = "IDLE"
        self._status_message = "Idle"
        self._tunnel_url = None

    def _call_kaggle_push(self) -> None:
        if self._push_fn is not None:
            self._push_fn()
            return
        session_manager._kaggle_push()

    def _call_get_kaggle_status(self) -> str:
        if self._status_fn is not None:
            return self._status_fn()
        return session_manager.get_kaggle_status()

    def _call_is_tunnel_healthy(self) -> Optional[str]:
        if self._health_fn is not None:
            return self._health_fn()
        return session_manager.is_tunnel_healthy(registry_url=self.registry_url)

    async def start_session(self) -> Dict[str, str]:
        """Trigger manual GPU session startup adhering to the exact state machine.

        Returns immediately (non-blocking) with STARTING status while background
        task executes Kaggle push and polling.
        """
        async with self._lock:
            # 1. If STARTING: no-op, returns current status, does not push again
            if self._state == "STARTING":
                logger.info("Session is already STARTING; duplicate click ignored.")
                return {"status": self._state, "message": self._status_message}

            # 2. If READY: no-op, returns READY immediately, does not push again
            if self._state == "READY":
                logger.info("Session is already READY; no-op.")
                return {"status": self._state, "message": self._status_message}

            # 3. If IDLE or ERROR: reset to STARTING and retry from scratch
            self._state = "STARTING"
            self._status_message = "GPU session startup initiated. Polling for tunnel readiness..."
            self._tunnel_url = None

            # Cancel any lingering polling task
            if self._polling_task is not None and not self._polling_task.done():
                self._polling_task.cancel()

            # Spawn background task to push and poll (fully non-blocking)
            self._polling_task = asyncio.create_task(self._run_startup_and_polling_loop())

            return {"status": self._state, "message": self._status_message}

    async def _run_startup_and_polling_loop(self) -> None:
        """Background task: executes Kaggle push in thread executor, then enters polling loop."""
        try:
            await asyncio.to_thread(self._call_kaggle_push)
        except Exception as exc:
            logger.error("Kaggle push failed during session start: %s", exc)
            async with self._lock:
                self._state = "ERROR"
                self._status_message = f"Kaggle push failed: {exc}"
            return

        await self._run_polling_loop()

    async def _run_polling_loop(self) -> None:
        """Background loop polling Kaggle status and tunnel health until terminal state."""
        start_time = time.time()
        logger.info("Background polling loop started (timeout=%ss, interval=%ss)",
                    self.startup_timeout_seconds, self.poll_interval_seconds)
        try:
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.startup_timeout_seconds:
                    async with self._lock:
                        self._state = "ERROR"
                        self._status_message = (
                            f"Session startup timed out after {self.startup_timeout_seconds:.1f}s without ready tunnel."
                        )
                    logger.warning("Session startup timed out: %s", self._status_message)
                    return

                # 1. Check Kaggle status for terminal failure or non-transient status errors
                try:
                    k_status = await asyncio.to_thread(self._call_get_kaggle_status)
                    if k_status in TERMINAL_FAILURE_STATUSES:
                        async with self._lock:
                            self._state = "ERROR"
                            self._status_message = (
                                f"Kaggle kernel reported terminal failure status: {k_status}"
                            )
                        logger.error("Kaggle terminal failure detected: %s", k_status)
                        return
                except session_manager.KaggleStatusError as exc:
                    async with self._lock:
                        self._state = "ERROR"
                        self._status_message = f"Kaggle status check failed: {exc}"
                    logger.error("Non-transient Kaggle status error: %s", exc)
                    return
                except Exception as exc:
                    logger.debug("Transient error checking Kaggle status: %s", exc)

                # 2. Check tunnel health
                try:
                    tunnel = await asyncio.to_thread(self._call_is_tunnel_healthy)
                    if tunnel is not None:
                        async with self._lock:
                            self._tunnel_url = tunnel
                            self._state = "READY"
                            self._status_message = f"Session ready at {tunnel}"
                        logger.info("Session became READY: %s", tunnel)
                        asyncio.create_task(self._sync_voices_to_worker(tunnel))
                        return
                except Exception as exc:
                    logger.debug("Error checking tunnel health: %s", exc)

                # Sleep before next poll
                await asyncio.sleep(self.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.debug("Polling loop task cancelled.")
        except Exception as exc:
            logger.exception("Unexpected error in polling loop: %s", exc)
            async with self._lock:
                self._state = "ERROR"
                self._status_message = f"Unexpected error in background poller: {exc}"

    async def _sync_voices_to_worker(self, tunnel_url: str) -> None:
        """Synchronize locally registered custom voices to remote worker after restart."""
        try:
            token = self.bearer_token or os.environ.get(ENV_BEARER_TOKEN) or os.environ.get(ENV_SHARED_SECRET)
            headers: Dict[str, str] = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
                headers["X-Server-Secret"] = token

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{tunnel_url.rstrip('/')}/voices", headers=headers)
                if resp.status_code != 200:
                    logger.warning("Could not query remote voices for sync: %s", resp.status_code)
                    return
                remote_voices = set(resp.json().get("voices", []))

            manifest = voices.registry.load_manifest()
            for v_name, v_meta in manifest.items():
                if v_name not in remote_voices:
                    try:
                        voice_rec = voices.registry.get_voice_ref(v_name)
                        audio_file = Path(voice_rec.path)
                        if not audio_file.is_file():
                            continue
                        raw_audio = audio_file.read_bytes()
                        files = {"file": (audio_file.name, raw_audio, "audio/wav")}
                        lang = (
                            v_meta.get("language", ["hi"])[0]
                            if isinstance(v_meta.get("language"), list)
                            else v_meta.get("language", "hi")
                        )
                        data: Dict[str, str] = {
                            "voice_id": v_name,
                            "language": lang,
                            "ref_text": voice_rec.ref_text,
                        }
                        if voice_rec.description:
                            data["description"] = voice_rec.description

                        async with httpx.AsyncClient(timeout=60.0) as client:
                            sync_resp = await client.post(
                                f"{tunnel_url.rstrip('/')}/voices/clone",
                                files=files,
                                data=data,
                                headers=headers,
                            )
                            if sync_resp.status_code == 201:
                                logger.info("Successfully synced local voice '%s' to remote worker.", v_name)
                            else:
                                logger.warning(
                                    "Failed to sync voice '%s' to remote worker: %s", v_name, sync_resp.text
                                )
                    except Exception as err:
                        logger.warning("Error syncing voice '%s': %s", v_name, err)
        except Exception as exc:
            logger.warning("Worker voice sync failed: %s", exc)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to defaults."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    if target_path.is_file():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to parse config file %s: %s", target_path, exc)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    yield
    gateway: Optional[SessionGateway] = getattr(app.state, "gateway", None)
    if gateway is not None:
        gateway.reset()


def normalize_and_save_audio(audio_bytes: bytes, destination_path: Path) -> None:
    """Normalize uploaded audio to 24 kHz mono 16-bit PCM WAV and persist to destination_path."""
    from pydub import AudioSegment
    import soundfile as sf
    import numpy as np
    import io

    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        sf.write(str(destination_path), samples, 24000, subtype="PCM_16")
    except Exception as exc:
        logger.warning("Audio normalization fallback to raw write: %s", exc)
        destination_path.write_bytes(audio_bytes)


def create_app(
    gateway: Optional[SessionGateway] = None,
    config_path: Optional[Path] = None,
    poll_interval_seconds: Optional[float] = None,
    startup_timeout_seconds: Optional[float] = None,
) -> FastAPI:
    """Factory creating and configuring the FastAPI gateway application."""
    config = load_config(config_path)

    orchestrator_cfg = config.get("orchestrator", {})
    registry_cfg = config.get("registry", {})
    remote_cfg = config.get("remote_server", {})

    resolved_interval = (
        poll_interval_seconds
        if poll_interval_seconds is not None
        else float(orchestrator_cfg.get("poll_interval_seconds", 5.0))
    )
    resolved_timeout = (
        startup_timeout_seconds
        if startup_timeout_seconds is not None
        else float(orchestrator_cfg.get("startup_timeout_seconds", 600.0))
    )
    resolved_registry_url = os.environ.get(ENV_REGISTRY_URL) or registry_cfg.get("webhook_url")
    resolved_bearer = (
        os.environ.get(ENV_BEARER_TOKEN)
        or os.environ.get(ENV_SHARED_SECRET)
        or remote_cfg.get("bearer_token")
    )

    app = FastAPI(title="AudioGen Session Gateway", lifespan=lifespan)

    active_gateway = gateway or SessionGateway(
        poll_interval_seconds=resolved_interval,
        startup_timeout_seconds=resolved_timeout,
        registry_url=resolved_registry_url,
        bearer_token=resolved_bearer,
    )
    app.state.gateway = active_gateway

    @app.post("/session/start", response_model=StatusResponse)
    async def start_session() -> StatusResponse:
        """Trigger session startup or return current status."""
        result = await active_gateway.start_session()
        return StatusResponse(**result)

    @app.get("/session/status", response_model=StatusResponse)
    async def get_session_status() -> StatusResponse:
        """Query current session state and status message."""
        return StatusResponse(
            status=active_gateway.state,
            message=active_gateway.status_message,
        )

    @app.get("/voices")
    async def get_voices(language: Optional[str] = Query(None)) -> Dict[str, List[str]]:
        """List available voices filtered by language ('en', 'hi', or 'pa')."""
        try:
            if language is not None:
                clean_lang = language.strip().lower()
                if clean_lang not in ("en", "hi", "pa"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported language filter: '{language}'. Supported: ['en', 'hi', 'pa']",
                    )
                voice_list = voices.registry.list_voices(clean_lang)
            else:
                voice_list = voices.registry.list_voices(None)
            return {"voices": voice_list}
        except HTTPException:
            raise
        except TypeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception as exc:
            logger.error("Error reading voices from registry: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list voices: {exc}",
            )

    @app.post("/generate")
    async def generate_proxy(payload: GeneratePayload, request: Request) -> Response:
        """Proxy synthesis requests to Kaggle GPU server only when session is READY."""
        current_state = active_gateway.state
        if current_state != "READY":
            # Strict compliance: HTTP 409 when not ready
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"error": "session not ready", "status": current_state},
            )

        tunnel_url = active_gateway.tunnel_url
        if not tunnel_url:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"error": "session not ready", "status": current_state},
            )

        remote_endpoint = f"{tunnel_url.rstrip('/')}/generate"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        token = active_gateway.bearer_token or os.environ.get(ENV_BEARER_TOKEN) or os.environ.get(ENV_SHARED_SECRET)
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Server-Secret"] = token

        try:
            try:
                req_body = await request.json()
            except Exception:
                req_body = payload.model_dump()
            async with httpx.AsyncClient(timeout=120.0) as client:
                remote_resp = await client.post(remote_endpoint, json=req_body, headers=headers)

            if remote_resp.status_code == 400 and (
                "Unknown voice name" in remote_resp.text or "Reference audio file" in remote_resp.text
            ):
                try:
                    manifest = voices.registry.load_manifest()
                    if payload.speaker_ref_name in manifest:
                        v_rec = voices.registry.get_voice_ref(payload.speaker_ref_name)
                        a_file = Path(v_rec.path)
                        if a_file.is_file():
                            v_lang = payload.language
                            files = {"file": (a_file.name, a_file.read_bytes(), "audio/wav")}
                            c_data = {
                                "voice_id": payload.speaker_ref_name,
                                "language": v_lang,
                                "ref_text": v_rec.ref_text,
                            }
                            if v_rec.description:
                                c_data["description"] = v_rec.description
                            async with httpx.AsyncClient(timeout=60.0) as c_client:
                                c_resp = await c_client.post(
                                    f"{tunnel_url.rstrip('/')}/voices/clone",
                                    files=files,
                                    data=c_data,
                                    headers=headers,
                                )
                            if c_resp.status_code == 201:
                                async with httpx.AsyncClient(timeout=120.0) as retry_client:
                                    remote_resp = await retry_client.post(
                                        remote_endpoint, json=req_body, headers=headers
                                    )
                except Exception as sync_exc:
                    logger.warning("On-demand voice sync during generate failed: %s", sync_exc)

            content_type = remote_resp.headers.get("content-type", "application/json")
            return Response(
                content=remote_resp.content,
                status_code=remote_resp.status_code,
                media_type=content_type,
            )
        except Exception as exc:
            logger.error("Failed to proxy /generate to remote server %s: %s", remote_endpoint, exc)
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"error": f"Failed to forward request to Kaggle server: {exc}"},
            )

    @app.post("/voices/clone")
    async def proxy_clone_voice(
        file: UploadFile = File(...),
        voice_id: str = Form(...),
        language: str = Form(...),
        ref_text: str = Form(...),
        name: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
    ) -> Response:
        """Proxy voice cloning request to remote GPU server only when session is READY."""
        current_state = active_gateway.state
        if current_state != "READY":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "session not ready",
                    "status": current_state,
                    "detail": "GPU session must be started and ready before cloning a voice.",
                },
            )

        tunnel_url = active_gateway.tunnel_url
        if not tunnel_url:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "session not ready",
                    "status": current_state,
                    "detail": "GPU session must be started and ready before cloning a voice.",
                },
            )

        # 1. Sanitize voice_id and enforce strict path safety
        clean_voice_id = re.sub(r"[^a-zA-Z0-9_-]", "", voice_id.strip())
        if not clean_voice_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Invalid voice_id",
                    "detail": "Voice identifier must contain alphanumeric characters.",
                },
            )

        local_refs = (REPO_ROOT / "voices" / "refs").resolve()
        local_refs.mkdir(parents=True, exist_ok=True)
        local_audio_path = (local_refs / f"{clean_voice_id}.wav").resolve()
        try:
            local_audio_path.relative_to(local_refs)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Path traversal detected", "detail": "Invalid voice path."},
            )

        raw_audio = await file.read()
        remote_endpoint = f"{tunnel_url.rstrip('/')}/voices/clone"
        token = active_gateway.bearer_token or os.environ.get(ENV_BEARER_TOKEN) or os.environ.get(ENV_SHARED_SECRET)
        headers: Dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Server-Secret"] = token

        files = {"file": (file.filename or f"{clean_voice_id}.wav", raw_audio, file.content_type or "audio/wav")}
        data: Dict[str, str] = {
            "voice_id": clean_voice_id,
            "language": language,
            "ref_text": ref_text,
        }
        if name:
            data["name"] = name
        if description:
            data["description"] = description

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                remote_resp = await client.post(remote_endpoint, files=files, data=data, headers=headers)

            if remote_resp.status_code == status.HTTP_201_CREATED:
                # Persist locally as normalized 24 kHz mono 16-bit PCM WAV
                normalize_and_save_audio(raw_audio, local_audio_path)

                voices.registry.register_voice(
                    voice_name=clean_voice_id,
                    ref_audio_path=local_audio_path,
                    ref_text=ref_text,
                    languages=[language.strip().lower()],
                    description=description or name,
                )

            return Response(
                content=remote_resp.content,
                status_code=remote_resp.status_code,
                media_type="application/json",
            )
        except Exception as exc:
            logger.error("Failed to proxy /voices/clone to remote worker: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"error": f"Failed to forward clone request to Kaggle server: {exc}"},
            )

    @app.get("/", response_class=HTMLResponse)
    async def get_index() -> HTMLResponse:
        """Serve the minimal self-contained session control UI."""
        if not UI_INDEX_PATH.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="UI index.html not found.",
            )
        html_content = UI_INDEX_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, media_type="text/html; charset=utf-8")

    return app


# Default module-level application
app: FastAPI = create_app()

__all__ = [
    "app",
    "create_app",
    "SessionGateway",
    "StatusResponse",
    "GeneratePayload",
]
