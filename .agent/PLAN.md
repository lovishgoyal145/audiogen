# Blueprint: AudioGen — Workspace Scaffolding, Port Strategy (17000 Series), and Baseline Verification Harness

## 1. Executive Summary & Context

This blueprint specifies the technical design, architectural layout, and execution roadmap for **Ticket 1** of the **AudioGen** project. The objective is to establish a hardened Python service foundation adhering strictly to `.agent/RULES.md` and `.agent/ticket.md`.

### Project Name: `AudioGen`
- **Application Scope:** High-performance audio generation and operational service platform.
- **Root Directory:** `/home/lovish/.gemini/antigravity/scratch/audiogen` (Recommended active workspace)
- **Module Identity:** `audiogen`

### Core Guardrails Enforced:
- **Port Allocation Strategy (Rule 1):** All listening services strictly bind within the `17000–17099` range. Standard ports (`3000`, `5000`, `8000`, `8080`, `8090`) are rejected at startup and configuration time with explicit validation errors.
- **Process & Execution Safety (Rule 2):** Zero uncontrolled process kills (`kill`, `pkill`, `fuser -k`). All runtime commands execute inside `.venv/bin/...`.
- **Secrets & Environment Isolation (Rule 3):** No hardcoded credentials; safe `.env.example` mapping primary and secondary service port allocations without secrets.
- **Error Handling & State Trapping (Rule 4):** Structured error trapping and genuine assertions; no mock masking in unit tests.
- **3-Agent Collaboration Protocol:** Enforced via `.agent/TICKET_TEMPLATE.md` defining Builder, Reviewer, and Auditor roles.

---

## 2. Workspace Analysis & Framework Selection

### Workspace Inspection Findings
- **Host Python Environment:** Python 3.12.3 (`/usr/bin/python3`) with standard library `venv` support.
- **Available Base Libraries:** `uvicorn` (0.51.0), `pydantic` (2.13.4), `python-dotenv` (1.2.2), `starlette` (1.3.1), `httpx` (0.28.1).
- **Port Availability:** The `17000–17099` port block is verified 100% free and unbound.
- **Isolated Tooling:** Standalone virtual environment (`.venv`) is required to guarantee reproducibility and satisfy Rule 2 (`.venv/bin/pytest tests/`).

### Framework Decision: FastAPI + Uvicorn
- **FastAPI / Starlette:** Provides high throughput, asynchronous execution, native OpenAPI generation, and clean lifespan lifecycle management.
- **Pydantic V2 / Settings:** Delivers type validation, environment variable extraction with fallback hierarchies, and port boundary validation.
- **Test Infrastructure:** `pytest` + `httpx.ASGITransport` / `starlette.testclient.TestClient` for in-process end-to-end HTTP health probing without relying on network socket binds during automated test runs.

---

## 3. Target File Layout

```
audiogen/
├── .agent/
│   ├── RULES.md                  # Project operational rules & guardrails (Preserved)
│   ├── ticket.md                 # Current ticket specification (Ticket 1)
│   ├── PLAN.md                   # Implementation blueprint (this document)
│   └── TICKET_TEMPLATE.md        # 3-Agent governance ticket template for future tickets
├── .env.example                  # Documented port slots (17000-17009) & sample configuration
├── pyproject.toml                # Project metadata (AudioGen), dependencies, and pytest config
├── src/
│   ├── __init__.py               # Package marker
│   ├── config.py                 # Pydantic-based configuration and port range validation
│   └── main.py                   # FastAPI AudioGen application instance & /healthz probe endpoint
└── tests/
    ├── __init__.py               # Test package marker
    └── test_config_and_health.py # Unit & integration tests for port precedence and health probe
```

---

## 4. Component Specifications

### 4.1. Configuration Engine (`src/config.py`)
- **Model:** `Settings` model reading from environment variables with `.env` backing.
- **Application Metadata:** `app_name: str = "AudioGen"`
- **Port Strategy & Precedence:**
  1. Inspect `PORT`.
  2. If unset, inspect `APP_PORT`.
  3. If unset, default to `17000`.
- **Validation Guardrails:**
  - Value must be an integer between `17000` and `17099` inclusive.
  - If a forbidden port (`3000`, `5000`, `8000`, `8080`, `8090`) or any port outside `17000–17099` is provided, raise a `ValueError` detailing the operational port violation.
- **Secondary Port Slots (Documented & Accessible):**
  - `WORKER_PORT`: Default `17001` (AudioGen worker & synthesis pipelines)
  - `METRICS_PORT`: Default `17002` (Prometheus metrics & monitoring)
  - `WEBHOOK_PORT`: Default `17003` (Callback webhooks)
  - `WEBSOCKET_PORT`: Default `17004` (Real-time audio streaming)
  - `DOCS_PORT`: Default `17005` (Documentation portal)
- **Singleton Factory:** `get_settings()` with caching (`functools.lru_cache`) to avoid redundant filesystem reads.

### 4.2. Application & Health Probe (`src/main.py`)
- **Application Instance:** `FastAPI(title="AudioGen API", version="0.1.0", ...)` configured with lifespan management.
- **Health Diagnostic Endpoint (`/healthz`):**
  - **Method:** `GET`
  - **Status Code:** `200 OK`
  - **Payload Contract:**
    ```json
    {
      "status": "healthy",
      "service": "AudioGen",
      "port": 17000,
      "environment": "development"
    }
    ```
- **CLI Bootstrapping:**
  ```python
  if __name__ == "__main__":
      import uvicorn
      settings = get_settings()
      uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=False)
  ```

### 4.3. Environment Blueprint (`.env.example`)
Explicitly maps and documents the AudioGen 17000 series port architecture:
```ini
# AudioGen Primary Core API Service (Default: 17000)
PORT=17000
APP_PORT=17000
ENVIRONMENT=development
APP_NAME=AudioGen

# AudioGen Secondary Service Port Allocation Block (17001 - 17009)
WORKER_PORT=17001
METRICS_PORT=17002
WEBHOOK_PORT=17003
WEBSOCKET_PORT=17004
DOCS_PORT=17005
RESERVED_SLOT_6=17006
RESERVED_SLOT_7=17007
RESERVED_SLOT_8=17008
RESERVED_SLOT_9=17009
```

### 4.4. 3-Agent Ticket Template (`.agent/TICKET_TEMPLATE.md`)
Standardized structure enforcing the 3-agent lifecycle:
1. **Builder Agent:** Task execution, code generation, adherence to allowed files.
2. **Reviewer Agent:** Architectural verification, invariant checks, code hygiene.
3. **Auditor / Sentinel Agent:** Verification suite execution, failure-path analysis, and mock-masking audit.

### 4.5. Test Suite (`tests/test_config_and_health.py`)
Assertions directly targeting real application logic (No Mock Masking):
- **Test 1: Default Port Assignment:** Without env variables, `Settings().port` defaults to `17000`.
- **Test 2: Environment Variable Precedence:**
  - `PORT=17050` yields `17050`.
  - `APP_PORT=17060` yields `17060` when `PORT` is absent.
  - `PORT=17010` overrides `APP_PORT=17020`.
- **Test 3: Port Range Boundary Enforcement:**
  - Ports `< 17000` (e.g. 8000, 8080, 5000, 3000, 16999) raise explicit validation errors.
  - Ports `> 17099` (e.g. 17100, 18000) raise explicit validation errors.
- **Test 4: Health Probe HTTP Contract:**
  - `TestClient(app).get("/healthz")` returns HTTP 200.
  - Payload contains `"status": "healthy"`, `"service": "AudioGen"`, and `"port": 17000`.
- **Test 5: Health Probe Dynamic Configuration:**
  - Changing configured port updates `/healthz` response payload faithfully.

---

## 5. Step-by-Step Implementation Roadmap

| Phase | Action Item | Target Files | Verification Check |
|---|---|---|---|
| **Phase 1** | Scaffolding & Governance | `.agent/TICKET_TEMPLATE.md`, `.env.example`, `pyproject.toml` | Verify files exist and document 17000-17009 slots with AudioGen naming |
| **Phase 2** | Virtual Environment Setup | `.venv/` | Initialize `.venv` and verify pytest is accessible via `.venv/bin/pytest --version` |
| **Phase 3** | Core Configuration Engine | `src/config.py`, `src/__init__.py` | Validate port range parsing and env precedence logic |
| **Phase 4** | Application Entry Point & Probe | `src/main.py` | Confirm `/healthz` route binds with dynamic port and AudioGen service name |
| **Phase 5** | Test Suite Implementation | `tests/test_config_and_health.py`, `tests/__init__.py` | Run `.venv/bin/pytest tests/` and verify 100% pass rate |

---

## 6. Verification Plan

### Automated Verification
```bash
.venv/bin/pytest tests/ -v
```
All tests must execute inside `.venv/bin/pytest` and pass with 0 failures, 0 warnings, and 0 mocked bypasses.
