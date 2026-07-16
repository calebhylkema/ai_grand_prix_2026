# Run VQ2

## Prerequisites

1. **Simulator:** launch the VQ2 sim from `.local/simulators/vq2/` (Windows only,
   internet required for anti-cheat). Select **Training** mode for all development —
   Qualification runs count and are code-audited.
2. **Environment:** `uv sync --all-extras` from the repo root.
3. A run must be live in the sim (drone on the pad) before starting the mission.

## Fly the mission

```bash
uv run aigp-vq2 --config training --attempts 3 --secs 60
```

This runs the legacy phased executor (`drone_racing.deployments.vq2.mission`):
STAB → thread gate 0 → brake/climb on the gate-pass event → scan to the mapped
bearing → map-gated accept → thread. `--config` names a deployment config
(`training` / `qualification`) or an explicit TOML path; `--secs` / `--attempts`
override the config's run parameters. Use `--help` for all options.

The active mission does **not** load the new versioned course-map/reference planner.
Its gate-1 range is a legacy camera-depth constant. The typed course map, NED
reference, ESKF, and MPCC remain offline/shadow-only until ADR-0004 promotion gates
pass.

The runner contains a reviewed, Training-only experimental transition-authority flag,
but it is **default-off, rejected by Qualification, and currently blocked from further
use**. The first bounded trial did not preserve a confirmed gate-1 target across the
gate-0 pass. A later authority-off capture preserved the visual track id but exposed a
confirmation-count reset at promotion; that defect is fixed offline and still needs a clean
authority-off replacement capture. Do not add `--experimental-transition-authority` to
ordinary runs; the next work in [project status](../project/status.md) remains shadow-only.

Key behaviors to know:

- The mission **re-arms every loop** — the sim silently ignores commands otherwise.
- All rates pass through the calibrated command boundary
  (see [calibrate controls](calibrate-controls.md)); every rate in the code is a
  *desired actual* body rate.
- Failsafes (tilt-abort, collision-abort, stale-feed abort) end an attempt early;
  the run summary prints the abort reason.

## After a run

Each attempt archives a full trajectory trace (frames + state in filenames). Read
the trace **before** changing anything — see [diagnose a flight](diagnose-a-flight.md).

## Configuration

Deployment config lives in `deployments/vq2/configs/` — `training.toml` for
development, `qualification.toml` for competitive runs. Course maps (mined gate
geometry) live in `deployments/vq2/course_maps/`. See
[configuration](../reference/configuration.md).
