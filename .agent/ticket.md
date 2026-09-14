# Ticket 1: Workspace Scaffolding, Port Strategy (17000 Series), and Baseline Harness

## Goal
Establish a hardened base project structure with isolated virtual environment tooling, dedicated 17000-series port configuration, unified settings loading, baseline health diagnostics, and a passing pytest verification harness.

## Context & Allowed Files
- .env.example
- pyproject.toml (or requirements.txt)
- src/config.py (or core/config.py)
- src/main.py
- tests/test_config_and_health.py
- .agent/TICKET_TEMPLATE.md

## Strict Off-Limits
- Do not bind or reference any port outside the 17000–17099 range.
- Do not modify or delete `.agent/RULES.md`.
- Do not commit secrets, `.env`, or `.agent/` state files.

## Acceptance Criteria
1. Project config defines the primary API server port defaulting to `17000` (configurable via `PORT` or `APP_PORT` in `.env`).
2. Secondary service port slots (workers, metrics) are documented and mapped within the 17001–17009 range in `.env.example`.
3. An application entry point (`main.py`) exposes a lightweight health probe (`/healthz`) reporting status and assigned port.
4. `.agent/TICKET_TEMPLATE.md` is populated to enforce the 3-agent format for all subsequent tickets.
5. Unit tests in `tests/test_config_and_health.py` verify port assignment precedence (env var overrides default 17000) and health probe response.
6. The test command `.venv/bin/pytest tests/` runs and passes with 100% success.

## Verification Command
.venv/bin/pytest tests/
