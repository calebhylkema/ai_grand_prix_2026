---
description: Run latency benchmarks on the perception-to-control pipeline
allowed-tools:
  - Bash
  - Read
---

Run performance benchmarks: `uv run aigp-benchmark` (or the underlying
`uv run pytest --benchmark-only -v`).

After results, analyze:
1. Is total pipeline latency under 40ms target?
2. Which stage is the bottleneck?
3. Has any stage regressed from the last run?

Compare against these targets:
- Perception (image → corners): <17ms
- PnP + ESKF update: <3ms
- Planning (MPCC solve): <20ms
- Control (command generation): <1ms
- **Total pipeline: <40ms**

Flag any stage exceeding its hard limit in RED.
