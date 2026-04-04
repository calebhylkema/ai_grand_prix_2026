# AI Grand Prix — Autonomous Drone Racing Stack

## Project Overview

Python-only autonomy stack for the Anduril AI Grand Prix 2026 ($500K autonomous FPV drone racing competition). Identical hardware for all teams — software is the only differentiator. Monocular RGB camera + IMU only. No depth, no GPS, no stereo, no LiDAR.

**Pipeline:** Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Racing Line Planner → MPCC/RL Controller → Throttle/Roll/Pitch/Yaw

## Critical Constraints

- Python 3.12+ (competition confirmed 3.14.2 works). C/Cython extensions permitted.
- Single forward-facing ~12MP wide-angle FPV camera
- 6-axis IMU (gyro + accel) — may saturate above ~16g
- ~100 TOPS onboard compute (likely NVIDIA Jetson Orin NX class)
- Telemetry provides: starting position, velocity, orientation
- Controls: throttle, roll, pitch, yaw (rate commands)
- NO depth, NO engine RPM, NO battery SoC, NO manual steering

## Repository Layout

```
drone_racing/          # Main package
  perception/          # Gate detection (YOLO keypoint + classical CV fallback)
  estimation/          # ESKF: IMU propagation + vision corrections
  planning/            # MPCC racing line + receding horizon
  control/             # RL policy, PID fallback, safety ladder
  sim/                 # Simulator adapter (DCL API interface)
  logging/             # Flight telemetry recording
  utils/               # Coordinate transforms, math helpers
config/                # TOML scenario configs
scripts/               # Entry points: run_sim.py, train_policy.py
tests/                 # pytest suite with fixtures
docs/                  # Research findings, competition notes
```

## Development Rules

### MUST

- MUST run `uv run ruff check --fix . && uv run ruff format .` before committing
- MUST type-hint all function signatures — `mypy --strict` is enforced in CI
- MUST use NumPy-style docstrings for all public functions
- MUST write tests for any new module — minimum 80% coverage
- MUST profile latency for any perception or control change — target <40ms total pipeline
- MUST use `logging` module (not print) for all debug output
- MUST use SI units internally (meters, radians, seconds) — convert at boundaries only

### MUST NOT

- MUST NOT use `time.sleep()` in the control loop — use async or event-driven patterns
- MUST NOT hardcode gate dimensions — read from config
- MUST NOT modify files in `control/safety.py` without explicit human approval
- MUST NOT use OpenCV `imshow` in production code — gate behind `--debug` flag
- MUST NOT commit secrets, API keys, or competition credentials

### Style

- Line length: 100 chars (configured in pyproject.toml)
- Imports: isort with Black-compatible profile
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- Prefer dataclasses or Pydantic models over raw dicts for structured data
- Prefer NumPy vectorized ops over Python loops for numerical code

## Key Commands

```bash
uv run pytest                           # Run all tests
uv run pytest -m "not simulation"       # Fast unit tests only
uv run pytest --benchmark-only          # Latency benchmarks
uv run ruff check --fix .               # Lint + auto-fix
uv run mypy drone_racing/               # Type check
uv run python scripts/run_sim.py        # Launch simulation
uv run python scripts/train_policy.py   # Train RL policy
```

## Coordinate Frames

- **World frame (W):** Right-handed, Z-up, origin at race start
- **Body frame (B):** FRD (Front-Right-Down), origin at drone center of mass
- **Camera frame (C):** OpenCV convention (X-right, Y-down, Z-forward)
- All rotations stored as quaternions (scalar-last: [x, y, z, w]) unless interfacing with scipy (scalar-first)
- Gate poses stored as [x, y, z, qx, qy, qz, qw] in world frame

## Competition Phases

1. **VQ1 (May–June 2026):** Desaturated environment, highlighted gates, high SNR — classical CV viable
2. **VQ2 (June–July 2026):** Photorealistic, lighting distractions, visual aids OFF — need learned detectors
3. **Physical Qualifier (Sept 2026, SoCal):** Real Neros drone, sim-to-real gap is the challenge
4. **Finals (Nov 2026, Columbus OH):** Head-to-head racing, reliability wins

## Architecture Decision Records

When making significant design decisions, create a markdown file in `docs/adr/` following the format:
`NNNN-title.md` with sections: Context, Decision, Consequences, Status.
