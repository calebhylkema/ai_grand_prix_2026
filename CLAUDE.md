# AI Grand Prix — Autonomous Drone Racing Stack

## Project Overview

Python-only autonomy stack for the Anduril AI Grand Prix 2026 ($500K autonomous FPV drone racing competition). Identical hardware for all teams — software is the only differentiator. Monocular RGB camera + IMU only. No depth, no GPS, no stereo, no LiDAR.

**Pipeline:** Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Racing Line Planner → MPCC/RL Controller → Throttle/Roll/Pitch/Yaw

## Critical Constraints

- Python 3.12+ (competition confirmed 3.14.2 works). C/Cython extensions permitted. Any language allowed.
- Single forward-facing FPV camera (resolution/FPS TBD — separate vision spec coming)
- 6-axis IMU (gyro + accel) via HIGHRES_IMU MAVLink message
- Telemetry: attitude, orientation, linear velocities, system status, nav reference data
- Controls: throttle, roll, pitch, yaw via SET_ATTITUDE_TARGET MAVLink message
- Communication: MAVLink v2 over UDP via MAVSDK. Command rate 50–120 Hz, heartbeat min 2 Hz.
- Sim runs locally on **Windows only**, requires internet (anti-cheat)
- ~100 TOPS onboard compute (likely NVIDIA Jetson Orin NX class) — physical stage only
- NO depth, NO GPS, NO absolute position, NO manual steering

## Repository Layout

```
drone_racing/          # Main package (organized by function)
  perception/          # Gate detection (YOLO keypoint + classical CV fallback)
  estimation/          # ESKF: IMU propagation + vision corrections
  planning/            # MPCC racing line + receding horizon
  control/             # RL policy, PID fallback, safety ladder
  sim/                 # Simulator adapter (pymavlink / MAVSDK / mock)
  logging/             # Flight telemetry recording
  utils/               # Coordinate transforms, math helpers
scripts/               # Entry points, organized BY PHASE
  vq2/                 # ⭐ ACTIVE — VQ2 stack (race.py, control_check.py, eskf.py, ...)
  vq1/                 # retired VQ1 runner + probes (ground-truth-rig tooling)
  pre_sim/             # mock-sim era tooling (training, benches)
config/                # TOML configs (note: scripts/vq2 hardcodes VQ2 spec intrinsics)
tests/                 # pytest suite with fixtures
docs/                  # specs/ (official PDFs+refs), adr/, ROADMAP.txt, RESEARCH.txt, CODE_MAP.md
data/ models/ logs/ runs/   # gitignored artifacts (training data, weights, flight logs)
SIMULATOR/             # VQ1 sim v1.0.3364 install — ground-truth validation rig, KEEP
VQ2/                   # VQ2 sim v1.0.3379 install
NEXT_SESSION.md        # ⭐ read-first handoff for the next working session
```

See `docs/CODE_MAP.md` for the per-file map.

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
- MUST NOT weaken safety/failsafe logic (watchdogs, rate/thrust clamps, envelope
  protection — inline in the controllers; a dedicated `control/safety.py` was planned
  but never created) without explicit human approval
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

- **Sim frame:** Local Cartesian (no geographic coords). Convention TBD — MAVLink uses NED but spec says "local Cartesian" without specifying. Clarification pending from organizers.
- **Body frame (B):** FRD (Front-Right-Down), origin at drone center of mass (MAVLink convention)
- **Camera frame (C):** OpenCV convention (X-right, Y-down, Z-forward)
- All rotations stored as quaternions (scalar-last: [x, y, z, w]) unless interfacing with scipy (scalar-first)
- Gate poses stored as [x, y, z, qx, qy, qz, qw] in world frame
- **NOTE:** SET_POSITION_TARGET_LOCAL_NED uses NED. Our internal Z-up convention will need a conversion layer at the MAVSDK boundary.

## Competition Phases

1. **VQ1 (May 2026, open through VQ2):** < 10 gates, minimal distractions, clear gate visibility, full 3D with elevation changes. Focus on **completion**. Unlimited attempts, 8 min max per run.
2. **VQ2 (LIVE 2026-06-30, sim v1.0.3379, spec VADR-TS-003):** < 20 gates, increased complexity (dynamic lighting, 3D-scanned scenes, obstacles). Fastest valid time counts; objective = qualify for the physical round. **⚠️ Telemetry cut: `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and `GATE_INFO`/`TRACK_INFO` are REMOVED — only IMU + camera + RACE_STATUS remain. State estimation + vision are now mandatory (see `docs/specs/vq2_spec.md`).** Training vs Competitive modes; competitive runs are code-auditable.
3. **Physical Qualifier (Sept 2026, SoCal):** Real Neros drone, sim-to-real gap is the challenge
4. **Finals (Nov 2026, Columbus OH):** Head-to-head racing, reliability wins

## Architecture Decision Records

When making significant design decisions, create a markdown file in `docs/adr/` following the format:
`NNNN-title.md` with sections: Context, Decision, Consequences, Status.
