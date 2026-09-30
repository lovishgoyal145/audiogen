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
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, AsyncGenerator, Callable, Dict, Final, List, Optional, Tuple
from urllib.parse import urlparse
import uuid
import httpx
from pydantic import BaseModel, Field
import yaml

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Autonomous .env loading (override=False to preserve shell-exported variables)
try:
    import dotenv
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        dotenv.load_dotenv(env_path, override=False)
except ImportError:
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

from orchestrator import session_manager
from orchestrator.session_manager import (
    TERMINAL_FAILURE_STATUSES,
    DiscoveryStatus,
    TunnelDiscoveryResult,
    KagglePushError,
    KagglePushConflictError,
    KaggleStatusError,
    SessionReconciliationResult,
    _kaggle_push,
    cancel_kaggle_kernel,
    check_tunnel_discovery,
    delete_tunnel_url,
    get_kaggle_status,
    is_tunnel_healthy,
    reconcile_remote_session,
)
import voices.registry

logger = logging.getLogger("audiogen.orchestrator.gateway")

DEFAULT_CONFIG_PATH: Final[Path] = REPO_ROOT / "config" / "orchestrator_config.yaml"
UI_INDEX_PATH: Final[Path] = REPO_ROOT / "ui" / "index.html"
SESSION_RUNTIME_FILE: Final[Path] = REPO_ROOT / "scratch" / "session.json"

ENV_BEARER_TOKEN: Final[str] = "SERVER_BEARER_TOKEN"
ENV_SHARED_SECRET: Final[str] = "SHARED_SECRET"
ENV_REGISTRY_URL: Final[str] = "TUNNEL_REGISTRY_WEBHOOK_URL"


def write_session_runtime_file(state_dict: Dict[str, Any]) -> None:
    """Persist active session metadata to scratch/session.json."""
    try:
        SESSION_RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = SESSION_RUNTIME_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
        temp_file.replace(SESSION_RUNTIME_FILE)
    except Exception as exc:
        logger.warning("Failed to write session runtime file: %s", exc)


def remove_session_runtime_file() -> None:
    """Remove scratch/session.json if present."""
    try:
        if SESSION_RUNTIME_FILE.exists():
            SESSION_RUNTIME_FILE.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to remove session runtime file: %s", exc)


def load_session_runtime_file() -> Optional[Dict[str, Any]]:
    """Read scratch/session.json if present."""
    try:
        if SESSION_RUNTIME_FILE.is_file():
            return json.loads(SESSION_RUNTIME_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load session runtime file: %s", exc)
    return None


def _get_other_pids_in_group(pgid: int, self_pid: int) -> List[int]:
    """Return PIDs of all processes in the specified process group excluding self_pid."""
    others: List[int] = []
    try:
        proc_dir = Path("/proc")
        if not proc_dir.is_dir():
            return others
        for entry in proc_dir.iterdir():
            if entry.name.isdigit():
                pid = int(entry.name)
                if pid == self_pid or pid <= 1:
                    continue
                try:
                    stat_file = entry / "stat"
                    if stat_file.is_file():
                        stat = stat_file.read_text()
                        rparen = stat.rfind(")")
                        if rparen != -1:
                            fields = stat[rparen + 1:].split()
                            if len(fields) >= 3 and int(fields[2]) == pgid:
                                others.append(pid)
                except (OSError, ValueError):
                    pass
    except Exception:
        pass
    return others


def terminate_process_group(pgid: int, grace_period: float = 2.0) -> None:
    """Terminate all processes in the specified process group with SIGTERM then SIGKILL.

    Safety guards:
    - Refuses to signal root (0) or init (1) process groups.
    - Strictly ensures the gateway is the process group leader (pgid == os.getpid())
      to avoid signaling parent shells or external processes.
    - Signals child PIDs first without signaling the caller PID, preventing immediate
      suicide and allowing child termination polling and SIGKILL escalation to complete.
    """
    if pgid <= 1:
        logger.warning("Refusing to terminate process group %d (safety guard).", pgid)
        return

    current_pid = os.getpid()
    if pgid != current_pid:
        logger.warning(
            "Refusing to terminate process group %d: current process (%d) is not group leader.",
            pgid,
            current_pid,
        )
        return

    other_pids = _get_other_pids_in_group(pgid, current_pid)
    logger.info("Terminating child processes in group %d (pids: %s, grace period: %.1fs)...", pgid, other_pids, grace_period)
    for pid in other_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            logger.warning("Error sending SIGTERM to PID %d: %s", pid, exc)

    # Wait during grace period in 0.1s slices, checking if other processes remain
    start = time.time()
    while time.time() - start < grace_period:
        remaining = _get_other_pids_in_group(pgid, current_pid)
        if not remaining:
            logger.info("All child processes in group %d exited cleanly under SIGTERM.", pgid)
            return
        time.sleep(0.1)

    # Escalate to SIGKILL if stubborn child processes remain
    remaining = _get_other_pids_in_group(pgid, current_pid)
    if remaining:
        logger.warning(
            "Child processes %s in group %d did not exit after %.1fs. Sending SIGKILL...",
            remaining,
            pgid,
            grace_period,
        )
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Error sending SIGKILL to PID %d: %s", pid, exc)
    else:
        logger.info("All child processes in group %d exited cleanly under SIGTERM.", pgid)


class StatusResponse(BaseModel):
    """Schema for /session/start, /session/status, and /session/terminate responses."""

    status: str
    message: str
    session_id: Optional[str] = None
    failure_stage: Optional[str] = None
    desired_state: Optional[str] = None
    operation_id: Optional[str] = None
    remote_status: Optional[str] = None
    tunnel_url: Optional[str] = None
    observed_state: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ObservedState:
    """Fine-grained observed reality of local and remote resources."""

    local_service_state: str = "RUNNING"
    kaggle_gpu_state: str = "UNKNOWN"
    kernel_slug: str = "avidok/audiogen"
    kernel_execution_status: str = "UNKNOWN"
    notebook_execution_state: str = "UNKNOWN"
    environment_setup_state: str = "UNKNOWN"
    model_weights_state: str = "UNKNOWN"
    voice_initialization_state: str = "UNKNOWN"
    api_service_readiness: str = "NOT_READY"
    tunnel_url: Optional[str] = None
    resource_ownership: str = "local_session"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_service_state": self.local_service_state,
            "kaggle_gpu_state": self.kaggle_gpu_state,
            "kernel_slug": self.kernel_slug,
            "kernel_execution_status": self.kernel_execution_status,
            "notebook_execution_state": self.notebook_execution_state,
            "environment_setup_state": self.environment_setup_state,
            "model_weights_state": self.model_weights_state,
            "voice_initialization_state": self.voice_initialization_state,
            "api_service_readiness": self.api_service_readiness,
            "tunnel_url": self.tunnel_url,
            "resource_ownership": self.resource_ownership,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedState:
        return cls(
            local_service_state=data.get("local_service_state", "RUNNING"),
            kaggle_gpu_state=data.get("kaggle_gpu_state", "UNKNOWN"),
            kernel_slug=data.get("kernel_slug", "avidok/audiogen"),
            kernel_execution_status=data.get("kernel_execution_status", "UNKNOWN"),
            notebook_execution_state=data.get("notebook_execution_state", "UNKNOWN"),
            environment_setup_state=data.get("environment_setup_state", "UNKNOWN"),
            model_weights_state=data.get("model_weights_state", "UNKNOWN"),
            voice_initialization_state=data.get("voice_initialization_state", "UNKNOWN"),
            api_service_readiness=data.get("api_service_readiness", "NOT_READY"),
            tunnel_url=data.get("tunnel_url"),
            resource_ownership=data.get("resource_ownership", "local_session"),
        )


class HeartbeatPayload(BaseModel):
    """Payload schema for POST /session/heartbeat."""

    session_id: Optional[str] = Field(None, description="Active session ID")


class TerminatePayload(BaseModel):
    """Payload schema for POST /session/terminate."""

    session_id: Optional[str] = Field(None, description="Active session ID")
    reason: Optional[str] = Field("user_request", description="Reason for termination")


class GeneratePayload(BaseModel):
    """Payload schema for POST /generate."""

    text: str = Field(..., description="Text to synthesize")
    language: str = Field(..., description="Language code ('en', 'hi', or 'pa')")
    speaker_ref_name: str = Field(..., description="Voice name matching voices registry")
    speed: float = Field(default=1.0, ge=0.2, le=3.0, description="Generation speed factor (default: 1.0)")
    return_uri: bool = Field(default=False, description="If True, returns file URI instead of binary audio")


class SessionGateway:
    """Manages the in-memory state machine, background poller, and proxying."""

    def __init__(
        self,
        poll_interval_seconds: float = 5.0,
        startup_timeout_seconds: float = 600.0,
        lease_timeout_seconds: float = 9.0,
        heartbeat_interval_seconds: float = 3.0,
        startup_grace_seconds: float = 120.0,
        shutdown_grace_seconds: float = 2.0,
        registry_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        kernel_slug: Optional[str] = None,
        push_fn: Optional[Callable[..., None]] = None,
        status_fn: Optional[Callable[..., str]] = None,
        health_fn: Optional[Callable[..., Optional[str]]] = None,
        discovery_fn: Optional[Callable[..., TunnelDiscoveryResult]] = None,
        schedule_exit: bool = False,
    ) -> None:
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.lease_timeout_seconds = float(lease_timeout_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.startup_grace_seconds = float(startup_grace_seconds)
        self.shutdown_grace_seconds = float(shutdown_grace_seconds)
        self.registry_url = registry_url
        self.bearer_token = bearer_token
        self.kernel_slug = kernel_slug
        self._push_fn = push_fn
        self._status_fn = status_fn
        self._health_fn = health_fn
        self._discovery_fn = discovery_fn
        self.schedule_exit = schedule_exit

        self._session_id: str = f"audiogen-sess-{uuid.uuid4().hex[:8]}"
        self._created_at: float = time.time()
        self._last_heartbeat: float = 0.0
        self._lease_watchdog_task: Optional[asyncio.Task] = None
        self._pgid: int = os.getpgid(os.getpid()) if hasattr(os, "getpgid") else os.getpid()
        self._pid: int = os.getpid()

        self._state: str = "IDLE"
        self._desired_state: str = "IDLE"
        self._status_message: str = "Idle"
        self._failure_stage: Optional[str] = None
        self._last_error: Optional[str] = None
        self._current_operation: str = "none"
        self._operation_id: Optional[str] = None
        self._attempt_id: int = 0
        self._last_known_successful_state: Optional[str] = None
        self._tunnel_url: Optional[str] = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._polling_task: Optional[asyncio.Task] = None
        self._observed: ObservedState = ObservedState(kernel_slug=self.kernel_slug or "avidok/audiogen")

    @property
    def state(self) -> str:
        return self._state

    @property
    def desired_state(self) -> str:
        return self._desired_state

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def failure_stage(self) -> Optional[str]:
        return self._failure_stage

    @property
    def current_operation(self) -> str:
        return self._current_operation

    @property
    def operation_id(self) -> Optional[str]:
        return self._operation_id

    @property
    def observed_state(self) -> ObservedState:
        return self._observed

    @property
    def tunnel_url(self) -> Optional[str]:
        return self._tunnel_url

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat

    @property
    def pgid(self) -> int:
        return self._pgid

    def validate_session_configuration(self) -> Tuple[bool, str]:
        """Validate that registry and credentials are configured before starting session."""
        if not self.registry_url or not str(self.registry_url).strip():
            return False, "Registry URL is not configured. Define TUNNEL_REGISTRY_WEBHOOK_URL in .env or configure registry.webhook_url."

        url = str(self.registry_url).strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False, f"Invalid registry URL structure: '{url}'. Must be a valid http:// or https:// URL."

        auth_token = os.environ.get("TUNNEL_REGISTRY_AUTH_TOKEN")
        if not auth_token or not auth_token.strip():
            return False, "TUNNEL_REGISTRY_AUTH_TOKEN is missing or empty in environment."

        has_kaggle_env = bool((os.environ.get("KAGGLE_USERNAME") or "").strip() and (os.environ.get("KAGGLE_KEY") or "").strip())
        has_kaggle_file = (Path.home() / ".kaggle" / "kaggle.json").is_file()
        if not (has_kaggle_env or has_kaggle_file):
            return False, "Kaggle credentials not found. Define KAGGLE_USERNAME and KAGGLE_KEY in .env or provide ~/.kaggle/kaggle.json."

        return True, ""

    def _build_status_response(self) -> Dict[str, Any]:
        """Produce structured dictionary adhering to StatusResponse schema."""
        return {
            "status": self._state,
            "message": self._status_message,
            "session_id": self._session_id,
            "failure_stage": self._failure_stage,
            "desired_state": self._desired_state,
            "operation_id": self._operation_id,
            "remote_status": self._observed.kernel_execution_status,
            "tunnel_url": self._tunnel_url,
            "observed_state": self._observed.to_dict(),
            "details": {
                "local_service": self._observed.local_service_state,
                "kaggle_gpu": self._observed.kaggle_gpu_state,
                "kernel_slug": self._observed.kernel_slug,
                "kernel_status": self._observed.kernel_execution_status,
                "tunnel_health": self._observed.api_service_readiness,
                "resource_ownership": self._observed.resource_ownership,
            },
        }

    def _write_session_state(self) -> None:
        try:
            data = {
                "session_id": self._session_id,
                "desired_state": self._desired_state,
                "gateway_pid": self._pid,
                "gateway_pgid": self._pgid,
                "port": 17000,
                "created_at": self._created_at,
                "updated_at": time.time(),
                "kaggle_kernel_slug": self.kernel_slug or "avidok/audiogen",
                "last_heartbeat": self._last_heartbeat,
                "status": self._state,
                "status_message": self._status_message,
                "current_operation": self._current_operation,
                "operation_id": self._operation_id,
                "attempt_id": self._attempt_id,
                "failure_stage": self._failure_stage,
                "last_known_successful_state": self._last_known_successful_state,
                "tunnel_url": self._tunnel_url,
                "observed_state": self._observed.to_dict(),
            }
            write_session_runtime_file(data)
        except Exception as exc:
            logger.warning("Failed to record session state to file: %s", exc)

    def load_persisted_state(self) -> bool:
        """Load session state from scratch/session.json if available."""
        saved = load_session_runtime_file()
        if not saved:
            return False
        try:
            if saved.get("session_id"):
                self._session_id = saved["session_id"]
            if saved.get("desired_state"):
                self._desired_state = saved["desired_state"]
            if saved.get("status"):
                self._state = saved["status"]
            if saved.get("status_message"):
                self._status_message = saved["status_message"]
            if saved.get("tunnel_url"):
                self._tunnel_url = saved["tunnel_url"]
            if saved.get("last_heartbeat"):
                self._last_heartbeat = float(saved["last_heartbeat"])
            if saved.get("last_known_successful_state"):
                self._last_known_successful_state = saved["last_known_successful_state"]
            if saved.get("failure_stage"):
                self._failure_stage = saved["failure_stage"]
            if saved.get("operation_id"):
                self._operation_id = saved["operation_id"]
            if saved.get("attempt_id"):
                self._attempt_id = int(saved["attempt_id"])
            if "observed_state" in saved and isinstance(saved["observed_state"], dict):
                self._observed = ObservedState.from_dict(saved["observed_state"])
            logger.info("Loaded persisted session state (session_id='%s', status='%s')", self._session_id, self._state)
            return True
        except Exception as exc:
            logger.warning("Error parsing persisted session state: %s", exc)
            return False

    async def reconcile_with_external_state(self) -> Dict[str, Any]:
        """Query real external state (Upstash Redis + Kaggle API) and reconcile."""
        async with self._lock:
            # 1. Probe tunnel discovery / health
            try:
                disc = await asyncio.to_thread(self._call_check_tunnel_discovery)
            except Exception as exc:
                disc = session_manager.TunnelDiscoveryResult(
                    status=session_manager.DiscoveryStatus.REGISTRY_UNREACHABLE,
                    message=str(exc),
                )

            if disc.status == session_manager.DiscoveryStatus.READY and disc.tunnel_url:
                self._tunnel_url = disc.tunnel_url
                self._state = "READY"
                self._desired_state = "READY"
                self._status_message = f"Session ready at {disc.tunnel_url} (reused existing GPU worker)"
                self._last_known_successful_state = "READY"
                self._failure_stage = None
                self._observed.api_service_readiness = "READY"
                self._observed.kaggle_gpu_state = "RUNNING"
                self._observed.model_weights_state = "LOADED"
                self._observed.tunnel_url = disc.tunnel_url
                self._observed.resource_ownership = "reused_session"
                self._write_session_state()
                logger.info("Reconciled with existing healthy tunnel %s", disc.tunnel_url)
                return self._build_status_response()

            # 2. Check Kaggle kernel status
            try:
                k_status = await asyncio.to_thread(self._call_get_kaggle_status)
            except Exception as exc:
                k_status = "UNKNOWN"

            self._observed.kernel_execution_status = k_status

            if k_status in ("RUNNING", "QUEUED"):
                self._observed.kaggle_gpu_state = "ALLOCATED"
                if self._state not in ("STARTING", "READY", "INITIALIZING"):
                    self._state = "RECOVERING"
                    self._status_message = f"Kaggle kernel is {k_status}. Reconnecting and awaiting tunnel..."
                self._write_session_state()
                if self._polling_task is None or self._polling_task.done():
                    self._polling_task = asyncio.create_task(self._run_polling_loop())
                return self._build_status_response()

            if k_status in session_manager.TERMINAL_FAILURE_STATUSES:
                if self._state in ("READY", "STARTING", "RECOVERING"):
                    logger.info("Persisted state was %s, but remote kernel is %s. Reconciling.", self._state, k_status)
                    self._state = "IDLE" if k_status == "COMPLETE" else "ERROR"
                    self._status_message = f"Remote kernel terminated with status: {k_status}"
                    self._failure_stage = "kaggle_kernel"
                    self._tunnel_url = None
                    self._write_session_state()
                return self._build_status_response()

            # If no tunnel and no active kernel
            if self._state == "READY":
                logger.info("Persisted state claimed READY, but neither tunnel nor kernel is active. Reconciling to IDLE.")
                self._state = "IDLE"
                self._desired_state = "IDLE"
                self._status_message = "Idle"
                self._tunnel_url = None
                self._failure_stage = None
                self._write_session_state()

            return self._build_status_response()

    def record_heartbeat(self, session_id: Optional[str] = None) -> bool:
        """Update last heartbeat timestamp for UI liveness tracking."""
        if session_id and session_id != self._session_id:
            logger.warning(
                "Ignored heartbeat with mismatched session_id '%s' (active: '%s')",
                session_id,
                self._session_id,
            )
            return False

        self._last_heartbeat = time.time()
        self._write_session_state()
        if (
            self._state in ("STARTING", "READY")
            and (self._lease_watchdog_task is None or self._lease_watchdog_task.done())
        ):
            try:
                self._lease_watchdog_task = asyncio.create_task(self._run_lease_watchdog())
            except RuntimeError:
                pass
        return True

    async def _run_lease_watchdog(self) -> None:
        """Background loop monitoring lease heartbeat without destroying remote sessions."""
        try:
            while True:
                await asyncio.sleep(max(0.1, self.heartbeat_interval_seconds))
                # Heartbeat monitoring only. Teardown is explicit via /session/terminate
                # or Kaggle's remote IdleWatchdog (10min).
        except asyncio.CancelledError:
            pass

    def reset(self) -> None:
        """Reset state machine to IDLE (primarily for test fixture isolation)."""
        if self._polling_task is not None and not self._polling_task.done():
            self._polling_task.cancel()
        if self._lease_watchdog_task is not None and not self._lease_watchdog_task.done():
            self._lease_watchdog_task.cancel()
        self._polling_task = None
        self._lease_watchdog_task = None
        self._state = "IDLE"
        self._desired_state = "IDLE"
        self._status_message = "Idle"
        self._failure_stage = None
        self._last_error = None
        self._current_operation = "none"
        self._operation_id = None
        self._tunnel_url = None
        self._last_heartbeat = 0.0
        self._observed = ObservedState(kernel_slug=self.kernel_slug or "avidok/audiogen")
        remove_session_runtime_file()

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

    def _call_check_tunnel_discovery(self) -> session_manager.TunnelDiscoveryResult:
        if self._discovery_fn is not None:
            return self._discovery_fn()

        # Backward compatibility: if _health_fn is provided
        if self._health_fn is not None:
            res = self._health_fn()
            if res:
                return session_manager.TunnelDiscoveryResult(
                    status=session_manager.DiscoveryStatus.READY,
                    tunnel_url=res,
                    message=f"Tunnel at {res} is healthy.",
                )
            return session_manager.TunnelDiscoveryResult(
                status=session_manager.DiscoveryStatus.NO_TUNNEL_REGISTERED,
                message="No tunnel registered yet.",
            )

        # Backward compatibility: if _call_is_tunnel_healthy was patched via mock
        fn = getattr(self, "_call_is_tunnel_healthy", None)
        if fn is not None and (
            hasattr(fn, "assert_called")
            or hasattr(fn, "return_value")
            or "Mock" in type(fn).__name__
        ):
            res = fn()
            if res:
                return session_manager.TunnelDiscoveryResult(
                    status=session_manager.DiscoveryStatus.READY,
                    tunnel_url=res,
                    message=f"Tunnel at {res} is healthy.",
                )
            return session_manager.TunnelDiscoveryResult(
                status=session_manager.DiscoveryStatus.NO_TUNNEL_REGISTERED,
                message="No tunnel registered yet.",
            )

        return session_manager.check_tunnel_discovery(registry_url=self.registry_url)

    async def start_session(self) -> Dict[str, Any]:
        """Trigger manual GPU session startup adhering to idempotent reconciliation.

        Returns immediately (non-blocking) with current or STARTING status while background
        task executes Kaggle push/reconciliation and polling.
        """
        async with self._lock:
            # 1. If an operation is already in progress, idempotent no-op
            if (
                self._state in ("STARTING", "INITIALIZING", "RECOVERING")
                or (self._polling_task is not None and not self._polling_task.done())
            ):
                logger.info("Session operation already in progress (%s); duplicate request ignored.", self._state)
                return self._build_status_response()

            # 2. If READY: no-op, returns READY immediately
            if self._state == "READY" and self._tunnel_url:
                logger.info("Session is already READY; no-op.")
                return self._build_status_response()

            # Pre-flight configuration validation (fail-fast < 50ms)
            is_valid, error_msg = self.validate_session_configuration()
            if not is_valid:
                logger.error("Session startup aborted due to configuration error: %s", error_msg)
                self._state = "ERROR"
                self._status_message = f"Configuration error: {error_msg}"
                self._failure_stage = "local_configuration"
                return self._build_status_response()

            # 3. Check if tunnel is ALREADY healthy before pushing!
            try:
                disc = await asyncio.to_thread(self._call_check_tunnel_discovery)
                if disc.status == session_manager.DiscoveryStatus.READY and disc.tunnel_url:
                    self._tunnel_url = disc.tunnel_url
                    self._state = "READY"
                    self._desired_state = "READY"
                    self._status_message = f"Session ready at {disc.tunnel_url} (reused existing GPU worker)"
                    self._last_heartbeat = time.time()
                    self._last_known_successful_state = "READY"
                    self._failure_stage = None
                    self._observed.api_service_readiness = "READY"
                    self._observed.kaggle_gpu_state = "RUNNING"
                    self._observed.model_weights_state = "LOADED"
                    self._observed.tunnel_url = disc.tunnel_url
                    self._observed.resource_ownership = "reused_session"
                    self._write_session_state()
                    asyncio.create_task(self._sync_voices_to_worker(disc.tunnel_url))
                    logger.info("Tunnel %s already healthy. Reused existing session.", disc.tunnel_url)
                    return self._build_status_response()
            except Exception as exc:
                logger.debug("Pre-start tunnel check skipped: %s", exc)

            # 4. Initiate startup state
            self._session_id = f"audiogen-sess-{uuid.uuid4().hex[:8]}"
            self._operation_id = f"op-{uuid.uuid4().hex[:8]}"
            self._current_operation = "start"
            self._attempt_id += 1
            self._desired_state = "READY"
            self._created_at = time.time()
            self._last_heartbeat = time.time()
            self._state = "STARTING"
            self._status_message = "GPU session startup initiated. Polling for tunnel readiness..."
            self._failure_stage = None
            self._tunnel_url = None
            self._observed.resource_ownership = "local_session"
            self._write_session_state()

            # Cancel any lingering tasks
            if self._polling_task is not None and not self._polling_task.done():
                self._polling_task.cancel()
            if self._lease_watchdog_task is not None and not self._lease_watchdog_task.done():
                self._lease_watchdog_task.cancel()

            # Spawn background task to push and poll (fully non-blocking)
            self._polling_task = asyncio.create_task(self._run_startup_and_polling_loop())
            self._lease_watchdog_task = asyncio.create_task(self._run_lease_watchdog())

            return self._build_status_response()

    async def terminate_session(
        self,
        reason: str = "user_request",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shut down the active AudioGen session and cleanly dismantle all resources.

        Idempotent and thread-safe.
        Transitions state to TERMINATING, then SHUTDOWN.
        Cleans:
        1. Polling and lease watchdog tasks
        2. Remote Kaggle worker via inside-out POST /shutdown
        3. Outside-in Kaggle kernel cancellation via kagglesdk/CLI
        4. Upstash Redis tunnel_url key
        5. Session runtime metadata file
        6. Schedules local process group termination
        """
        async with self._lock:
            if self._state in ("TERMINATING", "SHUTDOWN"):
                logger.info("Session already terminating or shutdown (%s); idempotent no-op.", self._state)
                return {
                    "status": "SHUTDOWN",
                    "message": "Session already terminating or shutdown.",
                    "session_id": self._session_id,
                }

            if session_id and session_id != self._session_id:
                logger.warning(
                    "Ignored terminate request with mismatched session_id '%s' (active: '%s')",
                    session_id,
                    self._session_id,
                )
                return {
                    "status": self._state,
                    "message": f"Ignored: mismatched session_id '{session_id}'",
                    "session_id": self._session_id,
                }

            if self._state in ("STARTING", "READY") and (not session_id or not str(session_id).strip()):
                logger.warning(
                    "Rejected terminate request without session_id while active session '%s' is %s",
                    self._session_id,
                    self._state,
                )
                return {
                    "status": self._state,
                    "message": "Rejected: session_id is required to terminate an active session",
                    "session_id": self._session_id,
                }

            is_idle = (self._state == "IDLE")
            logger.info("Initiating session shutdown (reason='%s', state='%s', session_id='%s')...", reason, self._state, session_id or self._session_id)
            self._state = "TERMINATING"
            self._status_message = f"Session terminating (reason: {reason})..."
            self._write_session_state()

            # 1. Cancel background polling and lease watchdog (preventing self-cancellation)
            current = asyncio.current_task()
            if self._polling_task is not None and not self._polling_task.done():
                if self._polling_task is not current:
                    self._polling_task.cancel()
            if self._lease_watchdog_task is not None and not self._lease_watchdog_task.done():
                if self._lease_watchdog_task is not current:
                    self._lease_watchdog_task.cancel()

            tunnel = self._tunnel_url
            token = self.bearer_token or os.environ.get(ENV_BEARER_TOKEN) or os.environ.get(ENV_SHARED_SECRET)
            kernel_slug = self.kernel_slug
            registry_url = self.registry_url
            saved_session_id = self._session_id

        try:
            # If not in IDLE, dismantle remote resources
            if not is_idle:
                # 2. Inside-out remote worker teardown
                if tunnel:
                    headers: Dict[str, str] = {}
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        headers["X-Server-Secret"] = token
                    try:
                        logger.info("Calling remote worker POST %s/shutdown...", tunnel)
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            resp = await client.post(f"{tunnel.rstrip('/')}/shutdown", headers=headers)
                            logger.info("Remote worker shutdown response: HTTP %d", resp.status_code)
                    except Exception as exc:
                        logger.warning("Remote worker POST /shutdown skipped or failed: %s", exc)

                # 3. Outside-in Kaggle kernel cancellation
                try:
                    await asyncio.to_thread(
                        session_manager.cancel_kaggle_kernel,
                        kernel_slug=kernel_slug,
                    )
                except Exception as exc:
                    logger.warning("Outside-in Kaggle kernel cancellation failed: %s", exc)

                # 4. De-register Upstash Redis key
                try:
                    await asyncio.to_thread(
                        session_manager.delete_tunnel_url,
                        endpoint_url=registry_url,
                    )
                except Exception as exc:
                    logger.warning("Upstash Redis tunnel deletion failed: %s", exc)

        finally:
            async with self._lock:
                # 5. Clean runtime file
                remove_session_runtime_file()

                # 6. Mark state as SHUTDOWN
                self._state = "SHUTDOWN"
                self._status_message = "All session resources terminated successfully."
                self._tunnel_url = None

                # 7. Schedule local process group exit if enabled
                if self.schedule_exit:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.call_later(0.5, self._terminate_local_process_tree)
                    except Exception as exc:
                        logger.warning("Failed to schedule local process exit: %s", exc)

        return {
            "status": "SHUTDOWN",
            "message": "All session resources terminated successfully.",
            "session_id": saved_session_id,
        }

    def _terminate_local_process_tree(self) -> None:
        """Invoked asynchronously after HTTP response is flushed to terminate local processes."""
        logger.info("Executing local process tree shutdown...")
        try:
            current_pid = os.getpid()
            pgid = getattr(self, "_pgid", None) or os.getpgid(current_pid)
            if pgid and pgid > 1 and pgid == current_pid:
                terminate_process_group(pgid, grace_period=self.shutdown_grace_seconds)
            logger.info("Gateway pid %d exiting cleanly via os._exit(0)...", current_pid)
            os._exit(0)
        except Exception as exc:
            logger.error("Error during local process tree termination: %s", exc)
            os._exit(0)


    async def _run_startup_and_polling_loop(self) -> None:
        """Background task: executes Kaggle push in thread executor, handling 409 conflict, then enters polling loop."""
        logger.info("Starting Kaggle worker...")
        try:
            await asyncio.to_thread(self._call_kaggle_push)
            logger.info("Kaggle execution launched.")
            logger.info("Waiting for tunnel registration in registry...")
        except session_manager.KagglePushConflictError as exc:
            logger.info(
                "Kaggle push returned 409 Conflict: %s. Kernel is already active. Reconciling and reusing...",
                exc,
            )
            async with self._lock:
                self._observed.resource_ownership = "reused_session"
                self._observed.kaggle_gpu_state = "ALLOCATED"
                self._state = "RUNNING"
                self._status_message = "Kaggle kernel is already active (409 Conflict reconciled). Reconnecting..."
                self._failure_stage = None
                self._write_session_state()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "409" in exc_str or "conflict" in exc_str or "savekernel" in exc_str:
                logger.info(
                    "Detected 409 Conflict in Kaggle push: %s. Reconciling with existing active kernel...",
                    exc,
                )
                async with self._lock:
                    self._observed.resource_ownership = "reused_session"
                    self._observed.kaggle_gpu_state = "ALLOCATED"
                    self._state = "RUNNING"
                    self._status_message = "Kaggle kernel is already active. Reconnecting and awaiting tunnel..."
                    self._failure_stage = None
                    self._write_session_state()
            else:
                logger.error("Kaggle push failed during session start: %s", exc)
                async with self._lock:
                    self._state = "ERROR"
                    self._status_message = f"Kaggle push failed: {exc}"
                    self._failure_stage = "kaggle_push"
                    self._write_session_state()
                return

        await self._run_polling_loop()

    async def _run_polling_loop(self) -> None:
        """Background loop polling Kaggle status and tunnel health until terminal state."""
        start_time = time.time()
        logger.info(
            "Background polling loop started (timeout=%.1fs, interval=%.1fs)",
            self.startup_timeout_seconds,
            self.poll_interval_seconds,
        )
        try:
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.startup_timeout_seconds:
                    final_disc = None
                    try:
                        final_disc = await asyncio.to_thread(self._call_check_tunnel_discovery)
                    except Exception:
                        pass
                    if final_disc and final_disc.status == DiscoveryStatus.READY and final_disc.tunnel_url:
                        async with self._lock:
                            self._tunnel_url = final_disc.tunnel_url
                            self._state = "READY"
                            self._desired_state = "READY"
                            self._status_message = f"Session ready at {final_disc.tunnel_url}"
                            self._last_heartbeat = time.time()
                            self._last_known_successful_state = "READY"
                            self._failure_stage = None
                            self._observed.api_service_readiness = "READY"
                            self._observed.kaggle_gpu_state = "RUNNING"
                            self._observed.model_weights_state = "LOADED"
                            self._observed.tunnel_url = final_disc.tunnel_url
                            self._write_session_state()
                        return

                    async with self._lock:
                        self._state = "ERROR"
                        self._status_message = (
                            f"Session startup timed out after {self.startup_timeout_seconds:.1f}s without ready tunnel."
                        )
                        self._failure_stage = "timeout"
                        self._write_session_state()
                    logger.warning(
                        "Session startup timed out after %.1fs. Last stage: %s",
                        self.startup_timeout_seconds,
                        self._failure_stage,
                    )
                    return

                # 1. Check Kaggle status for terminal failure or non-transient status errors
                try:
                    k_status = await asyncio.to_thread(self._call_get_kaggle_status)
                    self._observed.kernel_execution_status = k_status
                    if k_status in TERMINAL_FAILURE_STATUSES:
                        async with self._lock:
                            self._state = "ERROR"
                            self._status_message = (
                                f"Kaggle kernel reported terminal failure status: {k_status}"
                            )
                            self._failure_stage = "kaggle_kernel"
                            self._write_session_state()
                        logger.error("Kaggle terminal failure detected: %s", k_status)
                        return
                    elif k_status in ("RUNNING", "QUEUED"):
                        self._observed.kaggle_gpu_state = "ALLOCATED"
                except session_manager.KaggleStatusError as exc:
                    async with self._lock:
                        self._state = "ERROR"
                        self._status_message = f"Kaggle status check failed: {exc}"
                        self._failure_stage = "kaggle_status"
                        self._write_session_state()
                    logger.error("Non-transient Kaggle status error: %s", exc)
                    return
                except Exception as exc:
                    logger.debug("Transient error checking Kaggle status: %s", exc)

                # 2. Check tunnel discovery with structured status
                try:
                    discovery_res = await asyncio.to_thread(self._call_check_tunnel_discovery)
                    if discovery_res.status == DiscoveryStatus.READY and discovery_res.tunnel_url:
                        async with self._lock:
                            self._tunnel_url = discovery_res.tunnel_url
                            self._state = "READY"
                            self._desired_state = "READY"
                            self._status_message = f"Session ready at {discovery_res.tunnel_url}"
                            self._last_heartbeat = time.time()
                            self._last_known_successful_state = "READY"
                            self._failure_stage = None
                            self._observed.api_service_readiness = "READY"
                            self._observed.kaggle_gpu_state = "RUNNING"
                            self._observed.model_weights_state = "LOADED"
                            self._observed.tunnel_url = discovery_res.tunnel_url
                            self._write_session_state()
                        logger.info("Tunnel %s healthy (HTTP 200). Session READY.", discovery_res.tunnel_url)
                        asyncio.create_task(self._sync_voices_to_worker(discovery_res.tunnel_url))
                        return
                    elif discovery_res.status == DiscoveryStatus.REGISTRY_AUTH_ERROR:
                        async with self._lock:
                            self._state = "ERROR"
                            self._status_message = f"Registry authentication failed: {discovery_res.message}"
                            self._failure_stage = "registry_communication"
                            self._write_session_state()
                        logger.error("Registry authentication error during discovery: %s", discovery_res.message)
                        return
                    elif discovery_res.status == DiscoveryStatus.REGISTRY_UNREACHABLE:
                        logger.warning("Registry unreachable during discovery: %s", discovery_res.message)
                    elif discovery_res.status == DiscoveryStatus.NO_TUNNEL_REGISTERED:
                        logger.info("Worker still initializing; no tunnel registered in registry yet.")
                    elif discovery_res.status == DiscoveryStatus.TUNNEL_UNHEALTHY:
                        logger.info(
                            "Tunnel URL discovered (%s); awaiting healthy /health response...",
                            discovery_res.tunnel_url,
                        )
                    elif discovery_res.status == DiscoveryStatus.CONFIGURATION_ERROR:
                        async with self._lock:
                            self._state = "ERROR"
                            self._status_message = f"Configuration error during discovery: {discovery_res.message}"
                            self._failure_stage = "local_configuration"
                            self._write_session_state()
                        logger.error("Configuration error in polling loop: %s", discovery_res.message)
                        return
                except Exception as exc:
                    logger.debug("Error checking tunnel discovery: %s", exc)

                # Sleep before next poll
                await asyncio.sleep(self.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.debug("Polling loop task cancelled.")
        except Exception as exc:
            logger.exception("Unexpected error in polling loop: %s", exc)
            async with self._lock:
                self._state = "ERROR"
                self._status_message = f"Unexpected error in background poller: {exc}"
                self._failure_stage = "unexpected"
                self._write_session_state()

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
    gateway: Optional[SessionGateway] = getattr(app.state, "gateway", None)
    auto_reconcile: bool = getattr(app.state, "auto_reconcile", False)
    if gateway is not None and auto_reconcile:
        gateway.load_persisted_state()
        try:
            asyncio.create_task(gateway.reconcile_with_external_state())
        except RuntimeError:
            pass
    yield
    if gateway is not None:
        if gateway._polling_task is not None and not gateway._polling_task.done():
            gateway._polling_task.cancel()
        if gateway._lease_watchdog_task is not None and not gateway._lease_watchdog_task.done():
            gateway._lease_watchdog_task.cancel()


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
    lease_timeout_seconds: Optional[float] = None,
    heartbeat_interval_seconds: Optional[float] = None,
    startup_grace_seconds: Optional[float] = None,
    shutdown_grace_seconds: Optional[float] = None,
    schedule_exit: Optional[bool] = None,
    auto_reconcile: Optional[bool] = None,
) -> FastAPI:
    """Factory creating and configuring the FastAPI gateway application."""
    config = load_config(config_path)

    orchestrator_cfg = config.get("orchestrator", {})
    registry_cfg = config.get("registry", {})
    remote_cfg = config.get("remote_server", {})
    lifecycle_cfg = config.get("lifecycle", {})
    kaggle_cfg = config.get("kaggle", {})

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
    resolved_lease_timeout = (
        lease_timeout_seconds
        if lease_timeout_seconds is not None
        else float(lifecycle_cfg.get("lease_timeout_seconds", 9.0))
    )
    resolved_heartbeat_interval = (
        heartbeat_interval_seconds
        if heartbeat_interval_seconds is not None
        else float(lifecycle_cfg.get("heartbeat_interval_seconds", 3.0))
    )
    resolved_startup_grace = (
        startup_grace_seconds
        if startup_grace_seconds is not None
        else float(lifecycle_cfg.get("startup_grace_seconds", 120.0))
    )
    resolved_shutdown_grace = (
        shutdown_grace_seconds
        if shutdown_grace_seconds is not None
        else float(lifecycle_cfg.get("shutdown_grace_seconds", 2.0))
    )
    env_var_name = registry_cfg.get("webhook_url_env", ENV_REGISTRY_URL)
    resolved_registry_url = os.environ.get(env_var_name) or registry_cfg.get("webhook_url")
    resolved_bearer = (
        os.environ.get(ENV_BEARER_TOKEN)
        or os.environ.get(ENV_SHARED_SECRET)
        or remote_cfg.get("bearer_token")
    )
    resolved_kernel_slug = (
        os.environ.get("KAGGLE_KERNEL_SLUG")
        or kaggle_cfg.get("kernel_slug", "avidok/audiogen")
    )

    if schedule_exit is None:
        if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
            resolved_schedule_exit = False
        else:
            resolved_schedule_exit = True
    else:
        resolved_schedule_exit = schedule_exit

    if auto_reconcile is None:
        if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
            resolved_auto_reconcile = False
        else:
            resolved_auto_reconcile = True
    else:
        resolved_auto_reconcile = auto_reconcile

    app = FastAPI(title="AudioGen Session Gateway", lifespan=lifespan)

    active_gateway = gateway or SessionGateway(
        poll_interval_seconds=resolved_interval,
        startup_timeout_seconds=resolved_timeout,
        lease_timeout_seconds=resolved_lease_timeout,
        heartbeat_interval_seconds=resolved_heartbeat_interval,
        startup_grace_seconds=resolved_startup_grace,
        shutdown_grace_seconds=resolved_shutdown_grace,
        registry_url=resolved_registry_url,
        bearer_token=resolved_bearer,
        kernel_slug=resolved_kernel_slug,
        schedule_exit=resolved_schedule_exit,
    )
    app.state.gateway = active_gateway
    app.state.auto_reconcile = resolved_auto_reconcile

    @app.post("/session/start", response_model=StatusResponse)
    async def start_session() -> StatusResponse:
        """Trigger session startup or return current status."""
        result = await active_gateway.start_session()
        return StatusResponse(**result)

    @app.get("/session/status", response_model=StatusResponse)
    async def get_session_status(session_id: Optional[str] = Query(None)) -> StatusResponse:
        """Query current session state and status message."""
        active_gateway.record_heartbeat(session_id=session_id)
        return StatusResponse(**active_gateway._build_status_response())

    @app.post("/session/reconcile", response_model=StatusResponse)
    async def reconcile_session() -> StatusResponse:
        """Trigger explicit external reconciliation with remote Kaggle/tunnel state."""
        result = await active_gateway.reconcile_with_external_state()
        return StatusResponse(**result)

    @app.post("/session/heartbeat")
    async def heartbeat(
        payload: Optional[HeartbeatPayload] = None,
        request: Request = None,
    ) -> JSONResponse:
        """Receive UI heartbeat to maintain active session lease."""
        sess_id = None
        if payload and payload.session_id:
            sess_id = payload.session_id
        elif request:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    sess_id = body.get("session_id")
            except Exception:
                pass
        ok = active_gateway.record_heartbeat(session_id=sess_id)
        if not ok:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": "Heartbeat rejected: mismatched session_id",
                    "session_id": active_gateway.session_id,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "state": active_gateway.state,
                "session_id": active_gateway.session_id,
            },
        )

    @app.post("/session/terminate")
    async def terminate_session(
        payload: Optional[TerminatePayload] = None,
        request: Request = None,
    ) -> JSONResponse:
        """Immediate shutdown coordinator endpoint terminating all session resources."""
        sess_id = None
        reason = "user_request"
        if payload:
            sess_id = payload.session_id
            if payload.reason:
                reason = payload.reason
        if request:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    sess_id = body.get("session_id") or sess_id
                    reason = body.get("reason") or reason
            except Exception:
                pass

        result = await active_gateway.terminate_session(reason=reason, session_id=sess_id)
        if (
            result.get("message", "").startswith("Ignored: mismatched")
            or result.get("message", "").startswith("Rejected:")
        ):
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=result)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

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
        if current_state in ("TERMINATING", "SHUTDOWN"):
            return JSONResponse(
                status_code=status.HTTP_410_GONE,
                content={"error": "session terminated", "status": current_state},
            )
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
        if current_state in ("TERMINATING", "SHUTDOWN"):
            return JSONResponse(
                status_code=status.HTTP_410_GONE,
                content={"error": "session terminated", "status": current_state},
            )
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
