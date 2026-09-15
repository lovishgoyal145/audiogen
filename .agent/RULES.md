# Project Operational Rules & Guardrails

1. Port Allocation Strategy:
   - All backend, API, webhook, and web UI services MUST bind within the 17000–17099 port block (e.g., Core API & Web UI: 17000, Workers: 17001, Metrics/Admin: 17002, Webhooks: 17003, WebSockets: 17004, Docs: 17005).
   - Standard ports (3000, 5000, 8000, 8080, 8090) are STRICTLY FORBIDDEN.

2. Process & Execution Safety:
   - NEVER issue `kill`, `pkill`, or `fuser -k` against any running PID without explicit user verification.
   - All commands must execute within the project virtual environment (`.venv/bin/...`).

3. Scope Boundaries & Blast Radius Control:
   - Strictly limit file changes to the "Allowed Files" defined in the active ticket:
     - `src/audiogen/config.py`
     - `src/audiogen/engine.py`
     - `src/audiogen/main.py`
     - `.env.example`
     - `tests/test_kaggle_e2e.py`
     - `tests/test_ui_routes.py`
     - `tests/test_engine_mock.py`
     - `.agent/ticket.md`
     - `.agent/PLAN.md`
     - `.agent/RULES.md`
   - STRICTLY OFF-LIMITS:
     - Synthetic sine waves (440 Hz in `main.py`, acoustic formants in `engine.py`), mock audio buffers, or silent fallback generators.
     - `src/audiogen/ui/` layout and styling (preserve Batman aesthetic and client bundle).
     - `voices/registry.py` and `voices/registry_schema.json`.
     - Ports outside the 17000–17099 range (default: 17000).
     - Do NOT commit `.env` or any real API keys/credentials to Git.
     - Do NOT run `git commit` or `git push`.
   - Existing unit tests must never be weakened, deleted, or muted. All test cases must pass 100%.

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
   - Backend integration tests in `tests/test_ui_routes.py` must verify `GET /` returns `200 OK` with valid HTML content.
   - Full test suite verification command: `.venv/bin/pytest tests/` must execute with 100% pass rate before handoff.

7. Layout & Aesthetic Guardrails:
   - Left-aligned pinning: UI container strictly pinned to the left (`max-width: 520px; margin-left: 0; padding: 48px; min-height: 100vh`).
   - Clear right viewport: The right half of the screen remains completely clear and uncluttered.
   - Monochromatic Batman aesthetic:
     - Near-black background (`#09090b` or `#0a0a0a`)
     - Dark borders (`#27272a`)
     - Crisp muted gray typography (`#a1a1aa`)
     - Clean headings (`#f4f4f5`)
     - High-contrast active accents (`#ffffff`)
     - Sharp, understated borders on cards and buttons.

8. Multi-Step Flow Guardrails:
   - Step 1 (Select Language): "Select Language" header, 3 vertically stacked buttons with distinct gaps and margins (English, Hindi, Punjabi).
   - Step 2 (Select Voice): "Select Voice" header, lists voices for selected language, prominent `+` button to create/upload a new voice profile, selecting a voice card progresses to Step 3.
   - Step 3 (Script Input & Generation): Text area for target script, live character counter, "Generate Audio" button.
   - Step 4 (Progress & Output): Minimal progress bar or pulsating indicator during synthesis. On completion: embedded `<audio controls>` player, direct download link, and "Reset / New Generation" action.
   - Transitions must occur smoothly without full-page reloads.
