---
description: Launch a simulation run and analyze results
allowed-tools:
  - Bash
  - Read
---

Launch the simulator with: `uv run python scripts/run_sim.py --config config/default.toml`

After the run completes, analyze the flight log:
1. Did the drone complete the course? If not, where did it fail?
2. What was the lap time?
3. Gate pass rate (passed / total gates)
4. Peak velocity achieved
5. Any collision events?
6. p95 loop latency

Compare against previous best results if available in docs/benchmarks.md.
