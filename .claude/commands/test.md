---
description: Run the test suite and fix any failures
allowed-tools:
  - Bash
  - Read
  - Write
---

Run the project test suite using `uv run pytest -v --tb=short`. This collects:

- `tests/unit/` — subsystem unit tests (`unit/vq2/` holds the VQ2 deployment tests)
- `tests/integration/` — multi-component pipeline tests

Tests marked `simulation` require the live simulator and are excluded by default;
`archive/` is never collected.

If any tests fail:

1. Read the failing test to understand what it expects
2. Read the source code being tested
3. Fix the source code (not the test) unless the test is clearly wrong
4. Re-run only the failing tests to confirm the fix
5. Run the full suite one more time to check for regressions

Report a summary: total passed, failed, and any fixes applied.
