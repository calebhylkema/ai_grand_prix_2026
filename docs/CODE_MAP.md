# Code Map — what lives where and which phase it serves

Updated 2026-07-01 (workspace reorganization; formerly the root `MANIFEST.md`).
Phases: **pre-sim** (built against the mock adapter before 2026-06-01), **vq1**
(real sim v1.0.3364, ground-truth telemetry), **vq2** (sim v1.0.3379, vision + IMU
only — the ACTIVE phase).

> "pre-sim" is a *temporal* label, not dead code. Perception, ESKF, MPCC and RL are
> the mandatory stack for the **physical qualifier** and are progressively becoming
> mainline in VQ2. Don't delete.

## Top level

```
drone_racing/    importable package, organized BY FUNCTION (standard Python layout)
scripts/
  vq2/           ⭐ ACTIVE — standalone VQ2 stack (pymavlink, no package imports)
  vq1/           VQ1 race runner + sim probes (retired; reference + ground-truth rig tooling)
  pre_sim/       mock-sim era tooling (training, benches, visualization)
tests/           pytest suite for the drone_racing package (+ run_race helpers)
config/          default.toml — package config (VQ2 scripts hardcode spec intrinsics)
docs/            specs/ (official), adr/, ROADMAP.txt, RESEARCH.txt, CODE_MAP.md
data/            training data + captured VQ2 frames (gitignored)
models/          weights: gate detector ONNX, yolov8n-pose.pt base (gitignored)
logs/ runs/      flight logs / recorded runs (gitignored artifacts)
SIMULATOR/       VQ1 sim v1.0.3364 install — KEEP: it still broadcasts ground truth
                 (ODOMETRY/ATTITUDE) = our estimator-validation rig
VQ2/             VQ2 sim v1.0.3379 install (folder locked while sim runs)
NEXT_SESSION.md  ⭐ read-first handoff for the next working session
```

## scripts/vq2/ (ACTIVE)

| File | Role |
|---|---|
| `control_check.py` | ⭐ control-primitive probe suite; source of truth for the RATE_GAIN calibration `(-2.55,-2.55,-2.35)` + `desired_to_cmd()` |
| `race.py` | old VQ2 controller — threads gate 0 (6/6), tumbles at gate 1; tuned around the UNcalibrated rate mapping; reference only |
| `eskf.py` | 15-state visual-inertial ESKF, gate-as-landmark (validated) |
| `estimator.py` | Mahony attitude filter + velocity integrator (validated) |
| `guidance.py` | geometric acceleration→attitude controller (offline-validated; needs calibrated rates) |
| `recon.py` | passive telemetry/vision recon (saves frames to `data/vq2/recon_frames/`) |
| `estimate_test.py` | live estimator validation against the sim |
| `guidance_test.py` | offline guidance validation (no sim) |
| `test_estimation.py` | pytest unit tests locking in the ESKF/attitude math |
| `_archive/` | superseded: `control_probe.py` (→ control_check), `eskf_synthetic_test.py` (→ test_estimation) |

## scripts/vq1/ (retired, VQ1 ground-truth era)

`run_race.py` (the VQ1 race loop — completed the course ~30s), `axis_sign_probe.py`,
`hover_probe.py`, `sim_probe.py`. Useful against the OLD sim as a ground-truth rig.

## scripts/pre_sim/

`run_sim.py` (mock sim, `/sim` skill), `train_policy.py` (PPO), `train_gate_detector.py`,
`generate_training_data.py`, `tune_mpcc.py`, `bench_pipeline.py`, `bench_rl_inference.py`,
`visualize_flight.py`.

## drone_racing/ package

| Module | Phase | Status |
|---|---|---|
| `perception/` (classical, neural, pnp, tracker, hybrid) | pre-sim | mainline for VQ2/physical; scripts/vq2 currently uses its own HSV+PnP — consolidation pending |
| `estimation/eskf.py` | pre-sim | 16-state; has known frame-conversion drift (see memory) — the working VQ2 filter is `scripts/vq2/eskf.py` |
| `estimation/gate_mapper.py` | pre-sim | gate-map discovery; feeds MPCC handoff |
| `planning/` (mpcc, spline, dynamics) | pre-sim | MPCC p95 22ms warm-started; next up on the speed roadmap |
| `control/pid.py`, `rl_controller.py`, `rl_env.py`, `mpcc_adapter.py`, `rate_limiter.py` | pre-sim | RL best_model @400k usable; PID gains were mock-sim tuned |
| `sim/mavlink_adapter.py`, `encapsulated_data.py`, `vision_rx.py` | vq1 | live pymavlink adapter + RACE_STATUS/vision parsing |
| `sim/mavsdk_adapter.py`, `mock_adapter.py` | pre-sim | old MAVSDK path + offline mock |
| `logging/` | shared | flight logger + `--record` frame recorder |
| `utils/` | shared | config, transforms, types |

**Note:** `control/safety.py` is referenced by CLAUDE.md as a protected file but was
never actually created — the safety ladder lives inline in the controllers.
