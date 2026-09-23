# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and web UI services MUST bind within the 17000–17099 port block (e.g., Core API & Web UI: 17000, Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control (TICKET-013: End-to-End Speed Control Option for AudioGen UI & IndicF5 Kaggle Pipeline):
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket:
     - `ui/index.html` (Speed select control immediately above Generate Audio button; parameter passing in `handleGenerate()`)
     - `orchestrator/gateway.py` (`GeneratePayload` schema with default `speed = 1.0`; proxying)
     - `server/app.py` (`GenerateRequest` schema with default `speed = 1.0`; pass `speed` to `inference_engine.synthesize`)
     - `src/audiogen/engine.py` (`Synthesizer.synthesize` signature, `self._backend.config.speed` update, `backend_fn` call, fallback scaling)
     - `tests/test_server_endpoints.py` (Unit tests for `/generate` with default and custom speed)
     - `tests/test_gateway.py` (Unit tests for gateway proxying speed)
     - `tests/test_engine_mock.py` (Unit tests for `Synthesizer.synthesize` speed parameter)
     - `.agent/ticket.md`
     - `.agent/PLAN.md`
     - `.agent/RULES.md`
   - STRICTLY OFF-LIMITS:
     - `orchestrator/session_manager.py` — stable session lifecycle and tunnel discovery logic.
     - `scripts/*` (`scripts/launch_audiogen.sh`, `scripts/start.sh`, `scripts/sync_secrets_dataset.py`, `scripts/verify_env_and_auth.py`).
     - `config/*` (`config/orchestrator_config.yaml`, `config/kernel-metadata.json`).
     - `server/interactive_notebook.ipynb` — Kaggle worker notebook template; imports `server.app.app` dynamically.
     - `server/watchdog.py`, `server/tunnel.py`, `server/registry.py` — stable daemons and tunnel infrastructure.
     - `voices/*` (`voices/registry.py`, `voices/registry_schema.json`, `voices/refs/*`) — stable voice cloning subsystem.
     - `src/audiogen/normalizer.py` — preserved for Indic normalizer test suite integrity.
     - `batch/*` — unrelated batch runner templates.
     - Port allocations: strictly 17000–17099.
     - Existing test assertions must never be weakened, skipped, or deleted. All 422 unit tests must pass 100%.
     - Do NOT run `git commit` or `git push`.

4. Speed Control Feature & IndicF5 Parameter Invariants:
   - **Single Control Constraint**: The only new UI control is the "Speed" selector, placed strictly and immediately above the existing "Generate Audio" button in `ui/index.html`.
   - **No Unrelated Controls**: Do NOT add pitch, duration, pause duration, audio quality, voice preset, or advanced settings. The surrounding UI and Generate Audio button must remain otherwise identical.
   - **Authentic Generation Parameter**: The selected speed MUST be passed into the IndicF5 generation pipeline in Kaggle and applied at native synthesis time via IndicF5 / F5-TTS flow matching (`infer_process(..., speed=...)`). It MUST NOT be simulated via post-generation playback rate changes, HTML audio element `playbackRate`, or audio DSP resamplers.
   - **Safe Default & Baseline Identity**: The default speed MUST be `1.0`. When `speed == 1.0` (or when omitted by callers), synthesis must behave 100% identically to current baseline.
   - **Backward Compatibility**: Request payloads without `speed` must remain completely valid across both gateway and server schemas, defaulting cleanly to `1.0`.
   - **Parameter Range & Validation**: Supported speed values exposed in UI must be sensible and supported by IndicF5 (recommended range: 0.5x to 1.5x, e.g. 0.5x, 0.75x, 1.0x, 1.25x, 1.5x). Backend schemas should enforce `ge=0.2, le=3.0` to prevent division-by-zero or runaway synthesis lengths.
   - **End-to-End Verification**: Verify the full chain: UI selection -> Gateway POST /generate -> Remote Server POST /generate -> `Synthesizer.synthesize` -> `self._backend.config.speed` / `backend_fn(..., speed=...)`.

5. Secrets & Environment Isolation (Strict Non-Negotiable Guardrails):
   - **ZERO SECRETS IN VERSION CONTROL**: Never hardcode tokens, keys, passwords, bearer secrets, or private URLs in git-tracked files.
   - Read configurations exclusively from environment variables via a local `.env` file backed by a safe `.env.example`.
   - **Authoritative Secret Sourcing**:
     - The local `.env` file is the authoritative source for the three required runtime secrets:
       - `TUNNEL_REGISTRY_WEBHOOK_URL`
       - `TUNNEL_REGISTRY_AUTH_TOKEN`
       - `SERVER_BEARER_TOKEN`
     - The Builder must never require the user to manually retype or recreate secrets that already exist in `.env`.
   - **Private Kaggle Secret Dataset Delivery**:
     - Headless delivery of runtime secrets to remote Kaggle kernels must occur exclusively via a dedicated **private Kaggle Dataset** (slug: `<KAGGLE_USERNAME>/audiogen-secrets`).
     - `kaggle kernels push` pushes code only and does NOT transfer arbitrary auxiliary runtime files into the container. Staging ephemeral `runtime_secrets.json` into push directories is broken and STRICTLY FORBIDDEN.
     - The private secret dataset must be created/updated programmatically from `.env` prior to kernel push.
     - The dataset must be attached to the kernel via `dataset_sources` in `kernel-metadata.json` and mounted read-only at `/kaggle/input/audiogen-secrets/`.
     - The dataset must be marked strictly private (`is_private = True`, `--public` / `-u` flag is strictly forbidden).
   - **Kaggle Notebook Secret Loading & Independence**:
     - The git-tracked template `server/interactive_notebook.ipynb` MUST NEVER contain hardcoded credentials, auth tokens, or private webhook endpoints.
     - Notebook code must NEVER depend on interactive-only Kaggle APIs (`UserSecretsClient`) in headless CLI-pushed execution. Any call to `UserSecretsClient` must be strictly a secondary fallback wrapped in non-crashing exception handling (`try...except Exception:`) for optional manual Kaggle UI runs.
     - Startup secrets must be loaded directly from `/kaggle/input/audiogen-secrets/` (supporting `secrets.json` and individual variable files).
   - **Deterministic Startup Secret Assertion**:
     - Before attempting FastAPI startup or Cloudflare tunnel registration, the notebook must verify all three required secrets are present and non-empty.
     - Safe logging is mandatory:
       ```text
       TUNNEL_REGISTRY_WEBHOOK_URL: PRESENT
       TUNNEL_REGISTRY_AUTH_TOKEN: PRESENT
       SERVER_BEARER_TOKEN: PRESENT
       ```
     - If any secret is missing, the notebook must immediately abort execution (`sys.exit(1)` or `raise RuntimeError(...)`) logging the missing variable name(s).
     - Partial secret configuration must be treated as fatal failure.
     - NEVER print, log, or expose secret values in stdout, stderr, or loggers.

6. Remote Startup & Polling Invariants:
   - Status parsing in `session_manager.py` must support dot-qualified Kaggle enum values (`KernelWorkerStatus.*`) and normalize them cleanly to standard uppercase statuses (`ERROR`, `RUNNING`, `QUEUED`, `CANCELLED`, `COMPLETE`).
   - `gateway.py` polling loops must detect terminal failures (`KernelWorkerStatus.ERROR`) immediately and transition to `ERROR` state; it must never get stuck waiting in `STARTING` for 600 seconds when remote startup has failed.
   - Immediate fail-fast: By failing remote execution immediately upon missing secrets, Kaggle marks the kernel `ERROR`, allowing the gateway to report startup failure within seconds instead of hitting the 600-second timeout.
   - Pre-flight / Pre-push validation: The local orchestrator must verify that required secrets exist in `.env` and the secret dataset is synced before invoking `kaggle kernels push`.

7. Voice Audio Reference & Cloning Invariants:
   - Every reference audio file in `voices/refs/` must be an authentic, valid audio recording (e.g. 24 kHz 16-bit mono WAV) with duration >= 1.0 second (recommended >= 3.0 seconds, max 30.0 seconds) and audible, non-zero speech amplitude (`max(abs(audio_samples)) > 0`).
   - Dummy or placeholder files of 0.25-second silence (or pure zeroes) are strictly forbidden in voice registries.
   - Reference transcripts in `voices/registry_schema.json` must precisely correspond to the spoken words in the referenced audio file.
   - **Language Canonization**: The canonical supported languages are strictly `en` (English), `hi` (Hindi), and `pa` (Punjabi). Any other language code must be rejected with HTTP 400.
   - **Zero-Shot Condition Requirement**: F5-TTS / IndicF5 models require both reference audio AND reference transcript (`ref_text`) for conditioning. Voice cloning requests without `ref_text` must be rejected with HTTP 400.
   - **Audio Validation & Normalization**: Uploaded audio files must be validated for format and duration, and converted/resampled to canonical 24 kHz 16-bit mono PCM WAV before persistence.

8. Error Handling & Per-Item Failure Isolation:
   - No silent exception swallowing: all failures must be logged and reported.
   - Batch execution MUST preserve per-item failure isolation: a failure on task N must record `status: "failed"` with a clear error in `manifest_output.json` and continue to task N+1 without aborting the batch.
   - Error messages for unresolvable voices (`VoiceNotFoundError`) and missing on-disk audio assets (`FileNotFoundError`) must be distinctly distinguishable in API error responses.
   - Single model initialization: The synthesizer model weights MUST be loaded exactly once outside the task iteration loop, never reloaded per item.

9. Testing Integrity & Verification:
   - No mock masking: Unit tests must assert genuine application logic and real failures rather than trivial mocks.
   - Audio validation: Voice registry tests must verify real audio properties (duration >= 1.0s, valid headers, non-silent waveform).
   - Mock assertions on headers and JSON body arguments are mandatory in automated tests for any new or modified HTTP client invocation.
   - Full test suite verification command: `.venv/bin/pytest tests/` must execute with 100% pass rate before handoff.

10. Layout & Aesthetic Guardrails:
    - Left-aligned pinning: UI container strictly pinned to the left (`max-width: 520px; margin-left: 0; padding: 48px; min-height: 100vh`).
    - Clear right viewport: The right half of the screen remains completely clear and uncluttered.
    - Monochromatic Batman aesthetic:
      - Near-black background (`#09090b` or `#0a0a0a`)
      - Dark borders (`#27272a`)
      - Crisp muted gray typography (`#a1a1aa`)
      - Clean headings (`#f4f4f5`)
      - High-contrast active accents (`#ffffff`)
      - Sharp, understated borders on cards, inputs, and buttons.

11. Multi-Step & Language Accordion Flow Guardrails:
    - Top-Level Language Categories: Three prominent language choices: **English** (`en`), **Hindi** (`hi`), and **Punjabi** (`pa`).
    - Dynamic Voice Expansion: Selecting a language reveals the list of cloned voices available for that language.
    - Voice Selection: Each voice item provides clear selection feedback and passes the selected voice into generation.
    - Prominent Cloning Action: At the bottom of the revealed voice list, a dedicated `+ Clone New Voice` action is provided.
    - Cloning Input Mechanism: Clicking `Clone New Voice` opens an understated modal or drawer allowing the user to provide an authentic audio file (`.wav`, `.mp3`, `.flac`), voice identifier, target language, and reference transcript.
    - Seamless Immediate Availability: Upon successful cloning, the new voice is immediately added to the active voice list and selected for subsequent generation without page refresh.

12. GPU Session Lifecycle & Real Worker Verification:
    - GPU sessions are started MANUALLY, via one explicit UI action ("Start Session"), never auto-detected or silently triggered by a `/generate` or `/voices/clone` call.
    - Consequence: Both `/generate` and `/voices/clone` MUST reject requests with HTTP 409 if no session is currently `ready` — they must NEVER themselves trigger a session start.
    - Cloning is Real Work: Voice cloning is NOT a frontend-only mock. The remote worker must execute authentic audio validation, 24kHz mono resampling, and GPU model feature extraction/verification under `app.state.inference_lock`.
    - Quota Protection: Idle watchdog timeout remains 600s of inactivity; requests touch the watchdog to keep the session alive while actively used.

13. Voice Persistence & Worker Restart Resilience:
    - Authoritative Durable Storage: Local disk (`voices/registry_schema.json` and `voices/refs/<id>.wav`) is the permanent source of truth for all cloned voices.
    - Worker Restart Synchronization: Because Kaggle worker kernels are ephemeral and fresh instances boot with only default git assets, the gateway must automatically synchronize all custom local voices to the remote worker upon session readiness (`READY`) and provide on-demand sync fallback during `/generate`.
    - Surviving Backend Restarts: On gateway startup, `voices.registry.load_manifest()` automatically discovers all previously cloned voices without requiring database migrations.
