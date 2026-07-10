# Architecture

## Pipeline

```
Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Planner/Guidance → Calibrated Rate Commands (MAVLink)
```

## Repository layout

```
src/drone_racing/        installed package (import name: drone_racing)
  perception/            gate detection (classical, neural, PnP, tracker, hybrid)
  estimation/            16-state ESKF + gate mapper (pre-sim lineage)
  planning/              B-spline racing line + MPCC (p95 22 ms warm-started)
  control/               PID cascade, RL policy, MPCC adapter, rate limiter
  sim/                   sim adapters: pymavlink (live), MAVSDK (legacy), mock
  logging/               flight logger + frame recorder
  utils/                 config, transforms, types
  training/              gate-detector / RL / data-generation training entry logic
  deployments/vq2/       ⭐ ACTIVE deployment (see below)
  cli/                   console-script entry points (aigp-*)
deployments/vq2/         deployment data: configs + course maps
tests/                   unit / integration / replay/vq2 / simulation
archive/                 historical rounds + superseded VQ2 controllers (not in CI)
.local/                  simulators, datasets, models, runs, logs (gitignored)
```

## The VQ2 deployment (`drone_racing.deployments.vq2`)

| Module | Role |
|---|---|
| `mission.py` | ⭐ active runner — map-and-plan phased executor (ADR-0003) |
| `runtime.py` | Flight runtime: sensor stepping, calibrated command boundary, level-rate controller, tilt compensation, aborts/failsafes, thrust helpers, trace recorder |
| `perception.py` | glow-resistant gate detection (ring test) + cyan-line detection + PnP |
| `attitude.py` | Mahony attitude filter + velocity integrator |
| `eskf.py` | 15-state visual-inertial ESKF, gate-as-landmark (validated) |
| `guidance.py` | geometric acceleration→attitude controller |
| `calibration.py` | control-primitive probe; source of truth for RATE_GAIN `(−2.55, −2.55, −2.35)` |
| `mapping.py` | offline course-map miner (co-visible PnP gate pairs from traces) |
| `diagnostics.py` | recon, offline detector lab, frame inspection |

### Why the deployment duplicates package modules

`deployments/vq2/` has its **own** perception and ESKF, distinct from
`drone_racing.perception` / `drone_racing.estimation`. This is deliberate: the
deployment versions are flight-validated on sim v1.0.3379; the package versions are
the pre-sim lineage (the package ESKF has a known frame-conversion drift). Do not
merge them casually — consolidation is a separate, verified task on the roadmap
(pre-physical-qualifier).

## Phase lineage of package modules

"Pre-sim" is a temporal label, not dead code — perception, ESKF, MPCC and RL are the
mandatory stack for the physical qualifier.

| Module | Lineage | Status |
|---|---|---|
| `perception/` | pre-sim | mainline for physical; VQ2 uses deployment-local detection |
| `estimation/eskf.py` | pre-sim | 16-state; known frame-conversion drift — VQ2 filter is deployment-local |
| `estimation/gate_mapper.py` | pre-sim | gate-map discovery; feeds MPCC handoff |
| `planning/` | pre-sim | next up on the speed roadmap |
| `control/` | pre-sim | RL best_model @400k usable; PID gains were mock-sim tuned |
| `sim/mavlink_adapter.py` | vq1 | live pymavlink adapter + RACE_STATUS/vision parsing |
| `sim/mavsdk_adapter.py`, `mock_adapter.py` | pre-sim | legacy MAVSDK path + offline mock |

## Control interface (verified on v1.0.3379)

- **Rate + thrust via SET_ATTITUDE_TARGET is the ONLY sim-honored control.**
  Position/velocity setpoints are ignored (tested — drone tumbles).
- Commands pass through the calibrated boundary exactly once
  (see [calibrate controls](../how-to/calibrate-controls.md)).
- Re-arm every loop or the sim silently ignores commands.
- Command rate < 100 Hz; heartbeat ≥ 2 Hz; physics 120 Hz.

## Latency budget (end-to-end target <40 ms)

| Stage | Target | Hard limit |
|-------|--------|------------|
| Camera capture | 1–5 ms | 5 ms |
| Detection (classical / neural) | 5 / 17 ms | 20 ms |
| PnP + ESKF update | 1–3 ms | 5 ms |
| Planning (MPCC solve) | 10–20 ms | 25 ms |
| Command generation | <1 ms | 2 ms |
