# Command line

Supported console scripts (installed via `pyproject.toml`; run with `uv run <cmd>`):

| Command | Module | Purpose |
|---|---|---|
| `aigp-vq2` | `drone_racing.deployments.vq2.mission` | ⭐ fly the active VQ2 mission (also hosts the `detect-lab`/diagnostic subcommands) |
| `aigp-calibrate` | `drone_racing.deployments.vq2.calibration` | control-primitive probes: `--test rates`/`hover`/`translate`, `--calib` |
| `aigp-replay` | `drone_racing.cli.replay` | flight-log / trace visualization |
| `aigp-train-detector` | `drone_racing.training.gate_detector` | YOLOv8-pose gate detector training + data generation + ONNX export |
| `aigp-benchmark` | `drone_racing.cli.benchmark` | perception→control latency benchmarks |

Every command supports `--help`. These are the only supported entry points — there
are no runnable scripts in the repository root, and `archive/` code is not runnable.

## Development commands

```bash
uv run pytest                     # all active tests (unit + integration + replay)
uv run pytest -m "not simulation" # skip live-sim tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/drone_racing
uv build                          # package build
```
