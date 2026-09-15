# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and websocket services MUST bind within the 17000–17099 port block (e.g., Core API: 17000, Webhooks/Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control:
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket (`batch/notebook_template.ipynb`, `tests/test_batch_runner.py`).
   - `core/*` (`src/audiogen/engine.py`), `voices/*`, `server/*`, and `batch/manifest_schema.py` are STRICTLY OFF-LIMITS.
   - Do not modify `batch/runner.py` unless it directly calls engine/registry interfaces (confirmed: it does not; it is out of scope).
   - Existing unit tests must never be weakened, deleted, or muted. All existing test suites must pass 100%.

4. Secrets & Environment Isolation:
   - Never hardcode tokens, keys, passwords, or absolute environment-specific local machine paths.
   - Read configurations exclusively from environment variables via a `.env` file backed by a safe `.env.example`.

5. Error Handling & Per-Item Failure Isolation:
   - No silent exception swallowing: all failures must be logged and reported.
   - Batch execution MUST preserve per-item failure isolation (TICKET-003): a failure on task N must record `status: "failed"` with a clear error in `manifest_output.json` and continue to task N+1 without aborting the batch.
   - Error messages for unresolvable voices (`VoiceNotFoundError`) and missing on-disk audio assets (`FileNotFoundError`) must be distinctly distinguishable in `manifest_output.json`.
   - Single model initialization: The synthesizer model weights MUST be loaded exactly once outside the task iteration loop, never reloaded per item.

6. Testing Integrity & Verification:
   - No mock masking: Unit tests must assert genuine application logic and real failures rather than trivial mocks.
   - Call argument assertions: Synthesizer synthesis calls must be verified against actual IndicF5 signature arguments (`text`, `ref_audio_path`, `ref_text`).
   - Verification command must be executed with `.venv/bin/pytest tests/test_batch_runner.py tests/test_batch_manifest.py -v` and achieve 100% pass rate before handoff.
