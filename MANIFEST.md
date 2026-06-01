# File Manifest — pre-sim vs v1-sim

Quick map of what each piece was built for. **pre-sim** = built before the sim
dropped (2026-06-01), against the mock adapter / assumptions. **v1-sim** = built
for the real AI-GP sim (the VQ1 completion push). **shared** = used by both.

The importable `drone_racing/` package stays organized **by function** (standard
Python layout — splitting it by phase would break every import). This file is the
phase label so the package layout doesn't have to carry it.

> Note: "pre-sim" is a *temporal* label, not "dead code". Perception, the gate
> mapper, ESKF, MPCC and RL are bypassed for VQ1 but are the mandatory stack for
> the **physical qualifier** (no TRACK_INFO / no ground-truth odometry there).
> See `memory/project_pre_sim_work_preserved.md`.

## scripts/  (now split into folders)

```
scripts/
  run_race.py          v1-sim — THE live entry point (stays put; skills + every
                       run command reference it). Cross-era: it's the v1 race
                       loop but imports the pre-sim perception/ESKF/MPCC/RL.
  sim_v1/              v1-sim diagnostics written against the real sim
    axis_sign_probe.py   identify roll/pitch/yaw rate sign conventions
    hover_probe.py       measure hover throttle
    sim_probe.py         probe SET_POSITION_TARGET velocity-mode behaviour
  pre_sim/             pre-sim tooling (mock sim, training, tuning, analysis)
    run_sim.py           mock-sim runner  (/sim skill points here)
    train_policy.py      RL (PPO) training
    train_gate_detector.py  YOLO keypoint trainer (standalone, no drone_racing)
    generate_training_data.py
    tune_mpcc.py         MPCC weight tuning
    bench_pipeline.py    perception→control latency benchmark
    bench_rl_inference.py
    visualize_flight.py  flight-log plotting (generally useful)
```

## drone_racing/  (package — unchanged, labelled here)

| Module | Phase | Status in VQ1 path |
|---|---|---|
| `perception/` (classical, tracker, neural/PnP) | pre-sim | running (detects gates) but not steering control yet |
| `estimation/eskf.py` | pre-sim | `predict()` runs; output **bypassed** (sim odometry is ground-truth) |
| `estimation/gate_mapper.py` | pre-sim | running; feeds the MPCC-handoff trigger |
| `planning/` (mpcc, spline) | pre-sim | inactive in `--controller hover` |
| `control/pid.py` | pre-sim | overridden by the hover/lead-pursuit controller (had FRD bug) |
| `control/rl_controller.py` | pre-sim | inactive; trained weights trained on mock physics (likely retrain) |
| `control/rate_limiter.py` | shared | used by run_race |
| `control/safety.py` | shared | protected file — human approval to edit |
| `sim/mock_adapter.py`, `sim/mavsdk_adapter.py` | pre-sim | offline/mock + old MAVSDK path |
| `sim/mavlink_adapter.py` | **v1-sim** | live pymavlink adapter (the real sim connection) |
| `sim/encapsulated_data.py` | **v1-sim** | RACE_STATUS + TRACK_INFO parsing |
| `sim/vision_rx.py` | **v1-sim** | sim vision stream receiver |
| `logging/flight_logger.py` | shared | state/command logging |
| `logging/frame_recorder.py` | v1-sim | `--record` frame+telemetry capture |
| `utils/` (config, transforms, types) | shared | math/config/dataclasses |

## Top-level artifacts

| Path | Phase | Notes |
|---|---|---|
| `runs/` | v1-sim | smoke-test output (`--record`). Left at top level so `--record runs/...` paths stay simple. |
| `start_frame.png` | v1-sim | sim start-screen capture |
| `SIMULATOR/` | v1-sim | the AI-GP sim install (untracked) |
| `data/`, `models/`, `yolov8n-pose.pt` | pre-sim | training data + model weights |
| `config/`, `docs/`, `tests/` | shared | config, notes, test suite (organized by function) |
