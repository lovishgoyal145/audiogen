# Ticket 001: Indic TTS Core Engine & Audio Post-Processing

## 1. Objective & Scope
Construct an isolated, model-agnostic Python inference core for Hindi (`hi`) and Punjabi (`pa`), featuring:
1. **Text Normalization (`core/normalizer.py`):** Devanagari and Gurmukhi script cleaning, phonetically critical diacritic preservation (`virama`, `bindi`, `tippi`, `addak`, `chandrabindu`, `danda`), and cardinal number-to-words expansion up to crores.
2. **Inference Synthesizer (`core/engine.py`):** Model-agnostic `Synthesizer` interface with dynamic model weight path resolution, device resolution with graceful CPU fallback when CUDA is unavailable, structured error trapping without silent exception swallowing, strict language gating (`hi`, `pa`), and seamless normalizer integration.
3. **Audio Mastering & Export (`core/audio_processor.py`):** DSP pipeline providing 80 Hz high-pass filtering (sub-bass / rumble attenuation), ITU-R BS.1770-4 loudness normalization to `-14.0 LUFS` with `-1.0 dBTP` true peak ceiling, dimension validation, graceful degradation on silent/clipped audio, and 16-bit PCM WAV export at 24000 Hz or 44100 Hz.
4. **Configuration & Dependency Manifest (`pyproject.toml`, `requirements.txt`):** Ensure all new production dependencies (`numpy`, `soundfile`) are registered in primary project configuration manifests.
5. **Verification & Testing (`tests/`):** Unit test suites (`test_normalizer.py`, `test_audio_processor.py`, `test_engine_mock.py`) executing inside `.venv/bin/pytest` with zero mock-masking, genuine assertion validation, and 100% pass rate.

## 2. Context & Allowed Files
*Modify ONLY the files listed below. Any change outside this list is considered scope creep.*
- `pyproject.toml`
- `requirements.txt`
- `core/engine.py`
- `core/normalizer.py`
- `core/audio_processor.py`
- `tests/test_normalizer.py`
- `tests/test_audio_processor.py`
- `tests/test_engine_mock.py`
- `.agent/PLAN.md`
- `.agent/tickets/TICKET-001.md`

## 3. Strict Off-Limits
*Guaranteed boundaries and system invariants that must never be violated.*
- Do not bind or reference any port outside the 17000–17099 range. Standard ports (3000, 5000, 8000, 8080, 8090) are strictly forbidden.
- Do not modify or delete `.agent/RULES.md`.
- Do not kill external processes, PIDs, or system daemons.
- Do not commit secrets, `.env`, or `.agent/` state files.
- `server/*`, `batch/*`, `voices/*` — strictly off-limits.
- Existing unit tests (`tests/test_config_and_health.py`) must not be weakened, deleted, or muted.
- Zero hardcoded absolute paths to model weights.
- No network calls or downloads during automated testing.

## 4. 3-Agent Operational Protocol

### Agent 1: Builder Agent (Execution & Implementation)
- **Role:** Implements the required code and tests according to `.agent/PLAN.md` and ticket specifications.
- **Protocol:**
  - Strictly adhere to the allowed files list; touch no off-limits files.
  - Non-destructive execution: never weaken or delete existing tests.
  - Run all commands strictly inside `.venv/bin/...`.
  - Provide a concise summary of changes and test execution output.

### Agent 2: Reviewer Agent (Architecture & Invariant Verification)
- **Role:** Reviews the implementation diff against architectural guardrails and project rules.
- **Protocol:**
  - Verify zero port violations (confirm all ports are within 17000–17099).
  - Check for scope adherence: verify no unauthorized files were modified or created.
  - Check code quality, typing, exception handling, and absence of silent failures.

### Agent 3: Auditor / Sentinel Agent (Verification & Test Integrity)
- **Role:** Audits test coverage, assertions, and verification commands.
- **Protocol:**
  - Ensure genuine assertion validation: zero mock-masking or trivial assertion bypasses.
  - Run the exact verification command and confirm 100% test pass rate with 0 failures.
  - Verify error handling and edge cases (e.g. boundary values, forbidden inputs).

## 5. Acceptance Criteria
1. Text normalizer supports Hindi (`hi`) and Punjabi (`pa`) script cleaning, diacritic preservation, and cardinal numbers up to crores.
2. Synthesizer resolves model path from constructor or env vars, validates filesystem existence of model path, falls back to CPU if CUDA is unavailable with structured error logging on unexpected exceptions, validates language, and integrates normalizer.
3. Audio processor implements 80 Hz high-pass filtering, ITU-R BS.1770-4 loudness measurement/normalization (-14 LUFS, -1.0 dB true peak ceiling), 1D audio dimension validation across all public functions, resampling, and 16-bit PCM WAV export.
4. Primary project manifest `pyproject.toml` and `requirements.txt` define all new dependencies (`numpy`, `soundfile`).
5. Fallback inference in `Synthesizer` generates genuine acoustic waveforms and is covered by unit tests.
6. Unit tests pass with 100% success rate and zero mock-masking.

## 6. Verification Command
```bash
.venv/bin/pytest tests/
```

## 7. Sign-off Matrix
- [x] Builder Agent: Implemented allowed files and passed verification command.
- [ ] Reviewer Agent: Diff verified for scope, invariants, and port rules.
- [ ] Auditor Agent: Test integrity verified; 0 failures; 0 mock-masking detected.
