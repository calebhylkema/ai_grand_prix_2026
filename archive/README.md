# Archive

Historical code kept for reference. **Nothing in this tree is supported, maintained,
or collected by CI** — it is excluded from packaging, Ruff, mypy, pytest, and coverage.
Sibling imports and stale paths are expected; do not "fix" archived code.

Everything here was last runnable in its original location at commit `2b1d566`
(the last commit before the 2026-07-09 repository restructure). To run any of it,
check out that commit.

| Directory | Contents |
| --- | --- |
| `rounds/pre_sim/` | Mock-simulator era runner (before the official sim dropped 2026-06-01) |
| `rounds/vq1/` | VQ1 qualifier stack + official VQ1 specification |
| `vq2_experiments/race_v1/` | First VQ2 controller (`race.py`) — superseded by the reactive controller |
| `vq2_experiments/reactive_fly/` | Reactive VQ2 controller (`fly.py`) — superseded by the map-and-plan mission |
| `vq2_experiments/probes/` | One-off VQ2 diagnostic probes whose findings are recorded in docs |

The active deployment lives in `src/drone_racing/deployments/vq2/`.
