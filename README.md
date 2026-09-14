# AudioGen

AudioGen is a high-performance audio generation and operational service platform built with FastAPI.

## Operational Guardrails & Port Strategy
AudioGen enforces strict operational guardrails adhering to `.agent/RULES.md`:
- All listening services bind strictly within the `17000–17099` port block.
- Standard ports (`3000`, `5000`, `8000`, `8080`, `8090`) are rejected at startup.
- Safe environment configuration is read via `.env` backed by `.env.example`.

## Quick Start

```bash
# Run verification suite
.venv/bin/pytest tests/

# Start server
.venv/bin/python src/audiogen/main.py
```
