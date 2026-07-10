# AI Grand Prix — Autonomous Drone Racing Stack

Python autonomy stack for the Anduril AI Grand Prix 2026 ($500K autonomous FPV
drone racing competition). Identical hardware for all teams — software is the only
differentiator. Monocular RGB camera + IMU only: no depth, no GPS, no stereo, no
ground-truth telemetry.

**Pipeline:** Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Planner/Guidance → Calibrated Rate Commands (MAVLink)

## Current state — read this first

- **VQ2 is the only supported deployment.** Everything else is archived.
- **`docs/project/status.md`** ⭐ is the single authoritative status document —
  read it at session start, update it in place instead of creating session handoffs.
- Active runner: `uv run aigp-vq2` (`src/drone_racing/deployments/vq2/mission.py`).
- Sim installs live in `.local/simulators/` (vq2 = active; vq1 = ground-truth
  validation rig). Sim in **Training** mode for dev — Qualification runs are
  code-audited.

## Repository layout

```
src/drone_racing/          installed package (import name: drone_racing)
  deployments/vq2/         ⭐ ACTIVE VQ2 deployment (mission, runtime, perception,
                           attitude, eskf, guidance, calibration, mapping, diagnostics)
  perception|estimation|planning|control/   pre-sim-lineage library modules
  training/  cli/          training logic + console entry points
deployments/vq2/           configs (training.toml / qualification.toml) + course maps
tests/                     unit/ integration/ replay/vq2/ simulation/
docs/                      Diátaxis docs — index.md, project/status.md ⭐
archive/                   historical rounds + superseded controllers — NOT active,
                           excluded from all tooling, never "fix" it
.local/                    simulators, datasets, models, runs, logs (gitignored)
```

Full map: `docs/reference/architecture.md`.

## Key commands

```bash
uv run pytest                           # all active tests
uv run pytest -m "not simulation"       # fast unit tests only
uv run ruff check --fix . && uv run ruff format .   # MUST run before committing
uv run mypy src/drone_racing            # type check
uv run aigp-vq2 --help                  # the mission runner
uv run aigp-calibrate --test rates --calib          # verify control calibration
```

## Development rules

### MUST

- MUST run `uv run ruff check --fix . && uv run ruff format .` before committing
- MUST type-hint all function signatures (strict mypy for library code; the VQ2
  deployment modules have relaxed per-module overrides in pyproject.toml)
- MUST use NumPy-style docstrings for public functions
- MUST write tests for new modules; profile latency for perception/control changes
  (<40 ms total pipeline)
- MUST use `logging` (not print) and SI units internally (meters, radians, seconds)

### MUST NOT

- MUST NOT weaken safety/failsafe logic — watchdogs, rate/thrust clamps, tilt and
  collision aborts, envelope protection, or the **calibrated command boundary**
  (`deployments/vq2/calibration.py`) — without explicit human approval. The safety
  ladder lives inline in the controllers; run `/review` on any diff touching
  control or estimation. See `docs/reference/safety.md`.
- MUST NOT use `time.sleep()` in the control loop
- MUST NOT hardcode gate dimensions (config only; the official camera intrinsics
  in the VQ2 deployment are spec constants and are the exception)
- MUST NOT use OpenCV `imshow` in production code (gate behind `--debug`)
- MUST NOT commit secrets or competition credentials
- MUST NOT touch `archive/` code — it is historical reference, excluded from tooling

## Conventions

- Line length 100; isort/Black-compatible via Ruff; `snake_case`/`PascalCase`/`UPPER_CASE`
- Dataclasses or Pydantic over raw dicts; NumPy vectorized ops over loops
- Quaternions scalar-last `[x,y,z,w]` (scipy interfaces are scalar-first)
- Coordinate frames: `docs/reference/coordinate-frames.md` — world NED, body FRD,
  camera OpenCV convention. ⚠️ The camera tilt sign (spec: 20° up; code: 20° down)
  is an open question — see status.md.
- Configuration: `docs/reference/configuration.md` — deployment configs under
  `deployments/vq2/configs/`, generated output under `.local/`
- Significant design decisions get an ADR in `docs/decisions/` (Context, Decision,
  Consequences, Status; supersede — never silently delete)

## Competition context

1. **VQ1** (May 2026) — complete, archived
2. **VQ2** (live since 2026-06-30, sim v1.0.3379, spec VADR-TS-003) — ⭐ ACTIVE.
   Telemetry removed: IMU + camera + RACE_STATUS only. Fastest valid time
   qualifies for the physical round. Deadline mid–late July 2026.
3. **Physical Qualifier** (Sept 2026, SoCal) — real Neros drone, ~100 TOPS onboard
4. **Finals** (Nov 2026, Columbus OH) — head-to-head, reliability wins

Constraints: Python 3.12+ (sim confirmed on 3.14.2), Windows-only sim (internet
required), MAVLink v2/UDP, command rate < 100 Hz, heartbeat ≥ 2 Hz.
