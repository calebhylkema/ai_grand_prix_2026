# Pre-sim round (archived)

**What it was:** Development era before the official AI Grand Prix simulator existed
(pre 2026-06-01). The stack was built and exercised against an in-house mock simulator
(`drone_racing/sim/mock_adapter.py`) — perception, ESKF, MPCC, PID, and RL policy were
all developed here.

- **Simulator:** in-house mock (no official build)
- **Original entry point:** `scripts/run_sim.py` (later `scripts/pre_sim/run_sim.py`)
- **Final known status:** full pipeline ran against the mock sim; RL policy trained to
  400k steps (usable checkpoint) — see `docs/` explanation pages for what survived
- **Why archived:** the official sim (VQ1, 2026-06-01) replaced the mock end-to-end;
  the mock's physics never matched
- **Runnable state:** commit `2b1d566` (original layout, before the 2026-07-09 restructure)
- **CI:** not collected, not supported

Reusable training logic (gate-detector training, RL training, data generation,
benchmarks, flight visualization) was promoted to `src/drone_racing/training/` and
`src/drone_racing/cli/` — only the round-specific runner is archived here.
