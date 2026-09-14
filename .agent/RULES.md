# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and websocket services MUST bind within the 17000–17099 port block (e.g., Core API: 17000, Webhooks/Workers: 17001, Metrics/Admin: 17002).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY OFF-LIMITS.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Secrets & Environment Isolation:
   - Never hardcode tokens, keys, or passwords.
   - Read configurations exclusively from environment variables via a `.env` file backed by a safe `.env.example`.

4. Error Handling & State Trapping:
   - No silent exception swallowing. All background tasks and workers must log tracebacks to stderr and write explicit structured error manifests.
   - No mock masking: Unit tests must assert against genuine application logic and real failures rather than trivial mocks.
