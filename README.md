# AI Grand Prix — Autonomous Drone Racing Stack

Autonomous FPV drone racing stack for the [Anduril AI Grand Prix 2026](https://theaigrandprix.com/).
Monocular RGB camera + IMU only — no depth, no GPS, no ground-truth telemetry.

**Active phase: VQ2** (Virtual Qualifier 2). See [docs/project/status.md](docs/project/status.md)
for current state.

## Quick start

```bash
# Install dependencies
uv sync --all-extras

# Launch the VQ2 simulator from .local/simulators/vq2/ (Windows, Training mode),
# start a run, then fly the mission:
uv run aigp-vq2 --attempts 3 --secs 60
```

## Supported commands

| Command | Purpose |
|---|---|
| `aigp-vq2` | fly the active VQ2 mission |
| `aigp-calibrate` | verify the rate/thrust command calibration |
| `aigp-replay` | flight-log / trace visualization |
| `aigp-train-detector` | gate-detector training + data generation |
| `aigp-benchmark` | pipeline latency benchmarks |

See [docs/reference/command-line.md](docs/reference/command-line.md).

## Tests

```bash
uv run pytest                     # all active tests
uv run pytest -m "not simulation" # skip live-sim tests
```

## Architecture

```
Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Planner/Guidance → Calibrated Rate Commands (MAVLink)
```

- [Architecture reference](docs/reference/architecture.md)
- [Project status](docs/project/status.md)
- [Official VQ2 spec](docs/reference/specifications/vq2/summary.md)
  ([original PDF](docs/reference/specifications/vq2/original.pdf))
- [Documentation index](docs/index.md)

## License

MIT
