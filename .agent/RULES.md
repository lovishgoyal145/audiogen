# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and websocket services MUST bind within the 17000–17099 port block (e.g., Core API: 17000, Webhooks/Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control:
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket.
   - Core modules (`core/*`), server endpoints (`server/*`), and batch pipelines (`batch/*`) are STRICTLY OFF-LIMITS unless explicitly specified in the active ticket.
   - Existing unit tests must never be weakened, deleted, or muted. All existing test suites must pass 100%.

4. Secrets & Environment Isolation:
   - Never hardcode tokens, keys, passwords, or absolute environment-specific local machine paths.
   - Read configurations exclusively from environment variables via a `.env` file backed by a safe `.env.example`.

5. Error Handling & Fail-Fast Contracts:
   - No silent exception swallowing or returning silent `None` when resources are not found.
   - Missing lookup keys must raise explicit, descriptive exceptions (e.g., `VoiceNotFoundError` inheriting from `KeyError`).
   - Missing on-disk assets (e.g., reference audio files) must fail fast at lookup time with `FileNotFoundError`, preventing silent failures deep inside inference pipelines.

6. Testing Integrity & Verification:
   - No mock masking: Unit tests must assert against genuine application logic and real failures rather than trivial mocks.
   - Verification command must be executed with `.venv/bin/pytest` and achieve 100% pass rate before handoff.
