# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and web UI services MUST bind within the 17000–17099 port block (e.g., Core API & Web UI: 17000, Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control (TICKET-008):
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket:
     - `server/interactive_notebook.ipynb`
     - `orchestrator/session_manager.py`
     - `voices/refs/anchor_male_energetic.wav`
     - `voices/refs/storyteller_punjabi_elder.wav`
     - `voices/refs/anchor_female_calm.wav`
     - `voices/registry_schema.json`
     - `.gitignore` (only to exclude ephemeral push staging secrets)
     - `tests/test_session_manager.py`
     - `tests/test_voice_registry.py`
     - `.agent/ticket.md`
     - `.agent/PLAN.md`
     - `.agent/RULES.md`
   - STRICTLY OFF-LIMITS:
     - `core/*`, `batch/*` — unrelated.
     - `server/app.py`, `server/tunnel.py`, `server/watchdog.py` — unrelated to this ticket.
     - `orchestrator/gateway.py` — session manager handles status parsing normalization; gateway logic remains intact.
     - Existing test assertions must never be weakened, skipped, or deleted. All unit tests must pass 100%.
     - Do NOT run `git commit` or `git push`.

4. Secrets & Environment Isolation (Strict Non-Negotiable Guardrails):
   - **ZERO SECRETS IN VERSION CONTROL**: Never hardcode tokens, keys, passwords, bearer secrets, or private URLs in git-tracked files.
   - Read configurations exclusively from environment variables via a local `.env` file backed by a safe `.env.example`.
   - **Kaggle Notebook & Remote Worker Secret Invariants**:
     - The git-tracked template `server/interactive_notebook.ipynb` MUST NEVER contain hardcoded credentials, auth tokens, or private webhook endpoints.
     - Notebook code must NEVER depend exclusively on interactive-only Kaggle APIs (`UserSecretsClient`). Any call to `UserSecretsClient` must be wrapped in non-crashing exception handling (`try...except Exception:`) with fallback to local environment or runtime configuration.
     - Headless delivery of secrets to remote Kaggle kernels must occur exclusively via ephemeral push staging (e.g. uncommitted runtime config created during `_kaggle_push()`), strictly excluded in `.gitignore`, and cleaned up immediately in a `finally:` block.
   - Remote HTTP calls to external coordinators (KV stores, webhooks, or tunnel proxies) MUST conditionally forward Authorization headers if a corresponding `_AUTH_TOKEN` or `_BEARER_TOKEN` exists in the environment.
   - Never log, print, or expose secret values in stdout, stderr, or debug loggers.

5. Session State & Kernel Status Invariants:
   - Status parsing in `session_manager.py` must support dot-qualified Kaggle enum values (`KernelWorkerStatus.*`, unquoted, single-quoted, double-quoted) and normalize them cleanly to standard uppercase statuses (`ERROR`, `RUNNING`, `QUEUED`, `CANCELLED`, `COMPLETE`).
   - Terminal failure detection (`TERMINAL_FAILURE_STATUSES`) must recognize both normalized statuses and dot-qualified enum forms defensively.
   - `gateway.py` polling loops must detect terminal failures immediately upon receipt and transition to `ERROR` state; it must never get stuck waiting in `STARTING` until timeout when a terminal failure has already occurred.

6. Voice Audio Reference Invariants:
   - Every reference audio file in `voices/refs/` must be an authentic, valid audio recording (e.g. 24 kHz 16-bit mono WAV) with duration >= 1.0 second (recommended >= 3.0 seconds) and audible, non-zero speech amplitude (`max(abs(audio_samples)) > 0`).
   - Dummy or placeholder files of 0.25-second silence (or pure zeroes) are strictly forbidden in voice registries.
   - Reference transcripts in `voices/registry_schema.json` must precisely correspond to the spoken words in the referenced audio file.

7. Error Handling & Per-Item Failure Isolation:
   - No silent exception swallowing: all failures must be logged and reported.
   - Batch execution MUST preserve per-item failure isolation (TICKET-003): a failure on task N must record `status: "failed"` with a clear error in `manifest_output.json` and continue to task N+1 without aborting the batch.
   - Error messages for unresolvable voices (`VoiceNotFoundError`) and missing on-disk audio assets (`FileNotFoundError`) must be distinctly distinguishable in `manifest_output.json`.
   - Single model initialization: The synthesizer model weights MUST be loaded exactly once outside the task iteration loop, never reloaded per item.

8. Testing Integrity & Verification:
   - No mock masking: Unit tests must assert genuine application logic and real failures rather than trivial mocks.
   - Audio validation: Voice registry tests must verify real audio properties (duration >= 1.0s, valid headers, non-silent waveform).
   - Call argument assertions: Synthesizer synthesis calls must be verified against actual IndicF5 signature arguments (`text`, `ref_audio_path`, `ref_text`).
   - Mock assertions on the headers argument are mandatory in automated tests for any new or modified HTTP client invocation.
   - Backend integration tests in `tests/test_ui_routes.py` must verify `GET /` returns `200 OK` with valid HTML content.
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
    - Shutdown remains automatic via the existing idle watchdog (TICKET-002) — only the START side is manual. Do not add auto-shutdown-on-response-sent logic; do not shorten the idle timeout as a side effect of any future ticket without this being the ticket's stated purpose.
    - Kaggle CLI invocation must be wrapped in a single internal function (`_kaggle_push()`) so the underlying command can change later (`kaggle kernels push` -> `kaggle kernels update`) without touching callers.
    - `kernels status` is informational only. The single source of truth for "is the session actually ready to serve" is: tunnel URL present in the existing registry/KV AND a real `GET /health` call against it succeeds. Never treat a Kaggle `RUNNING` status alone as readiness.
