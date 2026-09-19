# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and web UI services MUST bind within the 17000–17099 port block (e.g., Core API & Web UI: 17000, Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control (TICKET-009: Private Kaggle Secret Dataset & Fast-Fail Startup):
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket:
     - `config/kernel-metadata.json`
     - `scripts/sync_secrets_dataset.py` (new sync utility)
     - `scripts/verify_env_and_auth.py`
     - `orchestrator/session_manager.py`
     - `server/interactive_notebook.ipynb`
     - `tests/test_session_manager.py`
     - `tests/test_sync_secrets_dataset.py` (new test file)
     - `.gitignore`
     - `.agent/ticket.md`
     - `.agent/PLAN.md`
     - `.agent/RULES.md`
   - STRICTLY OFF-LIMITS:
     - `core/*`, `batch/*` — unrelated.
     - `server/app.py`, `server/tunnel.py`, `server/watchdog.py` — unrelated to secret propagation.
     - `voices/*` — voice assets and schema are stable from TICKET-008.
     - Existing test assertions must never be weakened, skipped, or deleted. All unit tests must pass 100%.
     - Do NOT run `git commit` or `git push`.

4. Secrets & Environment Isolation (Strict Non-Negotiable Guardrails):
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

5. Remote Startup & Polling Invariants:
   - Status parsing in `session_manager.py` must support dot-qualified Kaggle enum values (`KernelWorkerStatus.*`) and normalize them cleanly to standard uppercase statuses (`ERROR`, `RUNNING`, `QUEUED`, `CANCELLED`, `COMPLETE`).
   - `gateway.py` polling loops must detect terminal failures (`KernelWorkerStatus.ERROR`) immediately and transition to `ERROR` state; it must never get stuck waiting in `STARTING` for 600 seconds when remote startup has failed.
   - Immediate fail-fast: By failing remote execution immediately upon missing secrets, Kaggle marks the kernel `ERROR`, allowing the gateway to report startup failure within seconds instead of hitting the 600-second timeout.
   - Pre-flight / Pre-push validation: The local orchestrator must verify that required secrets exist in `.env` and the secret dataset is synced before invoking `kaggle kernels push`.

6. Voice Audio Reference Invariants:
   - Every reference audio file in `voices/refs/` must be an authentic, valid audio recording (e.g. 24 kHz 16-bit mono WAV) with duration >= 1.0 second (recommended >= 3.0 seconds) and audible, non-zero speech amplitude (`max(abs(audio_samples)) > 0`).
   - Dummy or placeholder files of 0.25-second silence (or pure zeroes) are strictly forbidden in voice registries.
   - Reference transcripts in `voices/registry_schema.json` must precisely correspond to the spoken words in the referenced audio file.

7. Error Handling & Per-Item Failure Isolation:
   - No silent exception swallowing: all failures must be logged and reported.
   - Batch execution MUST preserve per-item failure isolation: a failure on task N must record `status: "failed"` with a clear error in `manifest_output.json` and continue to task N+1 without aborting the batch.
   - Error messages for unresolvable voices (`VoiceNotFoundError`) and missing on-disk audio assets (`FileNotFoundError`) must be distinctly distinguishable in `manifest_output.json`.
   - Single model initialization: The synthesizer model weights MUST be loaded exactly once outside the task iteration loop, never reloaded per item.

8. Testing Integrity & Verification:
   - No mock masking: Unit tests must assert genuine application logic and real failures rather than trivial mocks.
   - Obsolete tests for `runtime_secrets.json` push staging must be removed and replaced with comprehensive tests for secret dataset synchronization, metadata verification, and `/kaggle/input` loading.
   - Audio validation: Voice registry tests must verify real audio properties (duration >= 1.0s, valid headers, non-silent waveform).
   - Mock assertions on the headers argument are mandatory in automated tests for any new or modified HTTP client invocation.
   - Full test suite verification command: `.venv/bin/pytest tests/` must execute with 100% pass rate before handoff.

9. Layout & Aesthetic Guardrails:
   - Left-aligned pinning: UI container strictly pinned to the left (`max-width: 520px; margin-left: 0; padding: 48px; min-height: 100vh`).
   - Clear right viewport: The right half of the screen remains completely clear and uncluttered.
   - Monochromatic Batman aesthetic:
     - Near-black background (`#09090b` or `#0a0a0a`)
     - Dark borders (`#27272a`)
     - Crisp muted gray typography (`#a1a1aa`)
     - Clean headings (`#f4f4f5`)
     - High-contrast active accents (`#ffffff`)
     - Sharp, understated borders on cards and buttons.

10. Multi-Step Flow Guardrails:
    - Step 1 (Select Language): "Select Language" header, 3 vertically stacked buttons with distinct gaps and margins (English, Hindi, Punjabi).
    - Step 2 (Select Voice): "Select Voice" header, lists voices for selected language, prominent `+` button to create/upload a new voice profile, selecting a voice card progresses to Step 3.
    - Step 3 (Script Input & Generation): Text area for target script, live character counter, "Generate Audio" button.
    - Step 4 (Progress & Output): Minimal progress bar or pulsating indicator during synthesis. On completion: embedded `<audio controls>` player, direct download link, and "Reset / New Generation" action.
    - Transitions must occur smoothly without full-page reloads.

11. GPU Session Lifecycle Policy (standing decision — do not relitigate per-ticket):
    - GPU sessions are started MANUALLY, via one explicit UI action ("Start Session"), never auto-detected or silently triggered by a `/generate` call.
    - Rationale: this repo runs on a 30hr/week free Kaggle GPU quota. Auto-detecting and cold-starting behind every request produces unpredictable per-request latency; a single manual trigger per work session produces one predictable wait, then fast generation for the rest of that sitting.
    - Consequence: `/generate` must reject requests with HTTP 409 if no session is currently `ready` — it must NEVER itself trigger a session start.
    - Shutdown remains automatic via the existing idle watchdog (600s inactivity) — only the START side is manual. Do not add auto-shutdown-on-response-sent logic; do not shorten the idle timeout as a side effect of any future ticket without this being the ticket's stated purpose.
    - Kaggle CLI invocation must be wrapped in a single internal function (`_kaggle_push()`) so the underlying command can change later without touching callers.
    - `kernels status` is informational only. The single source of truth for "is the session actually ready to serve" is: tunnel URL present in the existing registry/KV AND a real `GET /health` call against it succeeds. Never treat a Kaggle `RUNNING` status alone as readiness.
