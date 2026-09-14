# Ticket [NUMBER]: [TITLE]

## 1. Objective & Scope
[Brief description of the ticket goal, technical problem statement, and expected deliverable.]

## 2. Context & Allowed Files
*Modify ONLY the files listed below. Any change outside this list is considered scope creep.*
- `path/to/allowed_file_1`
- `path/to/allowed_file_2`

## 3. Strict Off-Limits
*Guaranteed boundaries and system invariants that must never be violated.*
- Do not bind or reference any port outside the 17000–17099 range. Standard ports (3000, 5000, 8000, 8080, 8090) are strictly forbidden.
- Do not modify or delete `.agent/RULES.md`.
- Do not kill external processes, PIDs, or system daemons.
- Do not commit secrets, `.env`, or `.agent/` state files.

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
1. [Criterion 1]
2. [Criterion 2]
3. [Criterion 3]

## 6. Verification Command
```bash
.venv/bin/pytest tests/
```

## 7. Sign-off Matrix
- [ ] Builder Agent: Implemented allowed files and passed verification command.
- [ ] Reviewer Agent: Diff verified for scope, invariants, and port rules.
- [ ] Auditor Agent: Test integrity verified; 0 failures; 0 mock-masking detected.
