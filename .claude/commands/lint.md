---
description: Lint, format, and type-check the codebase
allowed-tools:
  - Bash
  - Read
  - Write
---

Run the following checks in order:

1. `uv run ruff check --fix .` — fix all auto-fixable lint issues
2. `uv run ruff format .` — format all Python files
3. `uv run mypy drone_racing/ --strict` — strict type checking

For any mypy errors that remain after ruff fixes:
- Read the flagged file
- Add proper type annotations or fix type issues
- Re-run mypy to confirm resolution

Report: number of lint fixes applied, format changes, and remaining type errors.
