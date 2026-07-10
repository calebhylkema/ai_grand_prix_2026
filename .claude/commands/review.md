---
description: Safety review for control-critical code changes
allowed-tools:
  - Bash
  - Read
model: opus
---

Review all uncommitted changes to files in `src/drone_racing/control/`,
`src/drone_racing/estimation/`, and `src/drone_racing/deployments/vq2/`
(the active flight runtime, controllers, and estimators):

1. `git diff --name-only` to identify changed files
2. For each changed file, review the diff with `git diff <file>`
3. Check for:
   - Unclamped outputs that could exceed physical limits
   - Missing NaN/inf checks on numerical computations
   - Removed or weakened safety constraints (watchdogs, rate/thrust clamps,
     collision/tilt aborts, envelope protection)
   - Changed coordinate frame conventions without updating downstream
   - Division by zero risks (especially in quaternion normalization)
   - Race conditions in multi-threaded state access
   - Changes to the calibrated command boundary (rate inversion/scaling) — any
     edit here needs explicit justification
4. Verify that all changes have corresponding test updates

Report: PASS (safe to merge) or BLOCK (with specific concerns listed).
