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
3. `uv run mypy src/drone_racing/` — type checking (strict for library code; the
   `drone_racing.deployments.vq2` modules have relaxed overrides in pyproject.toml)

For any mypy errors that remain after ruff fixes:
- Read the flagged file
- Add proper type annotations or fix type issues
- Re-run mypy to confirm resolution

Note: `archive/` is historical reference code — excluded from all checks, never lint it.

Report: number of lint fixes applied, format changes, and remaining type errors.
