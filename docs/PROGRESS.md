# AI Grand Prix — Progress Tracker

**Last Updated:** 2026-03-12
**Current Phase:** Phase 0 — Foundation
**Current Step:** Step 1 — Project Initialization

---

## Phase 0: Foundation (Now → Sim Release, March–May 2026)

### Step 1: Project Initialization
- [x] Create folder structure
- [x] Create CLAUDE.md files (root + per-module)
- [x] Create .claude/ configs (mcp.json, settings.json, slash commands)
- [x] Create pyproject.toml with dependencies
- [x] Create config/default.toml
- [x] Create .gitignore
- [x] Create CI workflow (.github/workflows/ci.yml)
- [x] Create docs (RESEARCH.md, PLAN.md, CHATGPT_RESEARCH.md)
- [ ] Open folder in VS Code ← YOU ARE HERE
- [ ] Initialize git repo
- [ ] Install uv (if not already installed)
- [ ] Run `uv sync --all-extras` to install all dependencies
- [ ] Verify: `uv run python -c "import numpy; print(numpy.__version__)"`
- [ ] Initial git commit
- [ ] Create GitHub repo and push

### Step 2: Core Data Types & Interfaces
- [ ] `drone_racing/utils/types.py` — DroneState, DroneCommand, GateDetection, TelemetryData dataclasses
- [ ] `drone_racing/utils/transforms.py` — quaternion helpers, frame conversions
- [ ] `drone_racing/sim/adapter.py` — SimAdapter Protocol + MockAdapter for testing
- [ ] `tests/test_types.py` — validate dataclass construction, serialization
- [ ] `tests/test_transforms.py` — quaternion math correctness (property-based with hypothesis)
- [ ] All tests passing, mypy clean, ruff clean

### Step 3: Perception — Classical Gate Detector (VQ1)
- [ ] `drone_racing/perception/classical.py` — HSV threshold + contour + quad fitting
- [ ] `drone_racing/perception/pnp.py` — OpenCV solvePnP wrapper with confidence scoring
- [ ] `tests/fixtures/perception/` — 10+ synthetic gate images with known corner positions
- [ ] `tests/test_classical.py` — detector finds gates in synthetic images
- [ ] `tests/test_pnp.py` — PnP returns correct pose for known corner inputs
- [ ] Latency benchmark: classical detector <5ms per frame

### Step 4: Estimation — ESKF
- [ ] `drone_racing/estimation/eskf.py` — 16-state Error-State Kalman Filter
- [ ] `drone_racing/estimation/imu.py` — IMU integration with midpoint quaternion propagation
- [ ] `tests/test_eskf.py` — filter converges on simulated trajectory with known ground truth
- [ ] `tests/test_imu.py` — integration accuracy over short horizons
- [ ] Property test: filter uncertainty grows during IMU-only propagation, shrinks on correction
- [ ] Latency benchmark: ESKF predict + update <3ms

### Step 5: Planning — Spline Racing Line
- [ ] `drone_racing/planning/spline.py` — B-spline through gate waypoints
- [ ] `drone_racing/planning/speed_profile.py` — curvature-based speed limits
- [ ] `tests/test_spline.py` — path passes through all gates, is smooth (C2 continuous)
- [ ] `tests/test_speed_profile.py` — speed reduces in tight turns
- [ ] Visualize racing line with matplotlib (debug tool)

### Step 6: Control — PID Cascade (Baseline)
- [ ] `drone_racing/control/pid.py` — cascaded PID (position → velocity → attitude → rate)
- [ ] `drone_racing/control/safety.py` — output clamping, rate limiting, envelope protection
- [ ] `drone_racing/control/command.py` — DroneCommand output with safety checks
- [ ] `tests/test_pid.py` — step response settles within spec, no overshoot >20%
- [ ] `tests/test_safety.py` — commands always within physical limits
- [ ] Latency benchmark: control loop <1ms

### Step 7: Integration — Full Pipeline with Mock Sim
- [ ] `drone_racing/sim/mock_adapter.py` — replays image sequences + simulated IMU + basic physics
- [ ] `scripts/run_sim.py` — entry point that wires everything together
- [ ] Full pipeline: mock camera → detect → PnP → ESKF → plan → PID → command
- [ ] Drone completes a simple 4-gate circuit in mock sim without crashing
- [ ] End-to-end latency profiling: <40ms target
- [ ] Flight data logger working, can replay runs

### Step 8: MPCC Planner (Performance Upgrade)
- [ ] `drone_racing/planning/mpcc.py` — CasADi-based MPCC solver
- [ ] `drone_racing/planning/dynamics.py` — quadrotor dynamics model for MPCC
- [ ] `tests/test_mpcc.py` — solver converges on toy 2D problem, then 3D
- [ ] Swap MPCC in for spline+PID, verify faster lap times
- [ ] Latency benchmark: MPCC solve <20ms

### Step 9: Neural Gate Detector (VQ2 Prep)
- [ ] Synthetic training data generation script
- [ ] YOLOv8-nano keypoint model training pipeline
- [ ] ONNX export + inference wrapper
- [ ] `drone_racing/perception/neural.py` — ONNX Runtime inference
- [ ] `drone_racing/perception/detector.py` — unified interface, auto-selects backend
- [ ] `tests/test_neural.py` — detector finds gates in diverse lighting/clutter
- [ ] Latency benchmark: neural inference <17ms

### Step 10: RL Policy (Parallel Track)
- [ ] `drone_racing/control/rl_policy.py` — G&CNet MLP wrapper
- [ ] `scripts/train_policy.py` — PPO training loop with domain randomization
- [ ] Train on simulator, benchmark against MPCC
- [ ] Hot-swap capability: switch MPCC ↔ RL at runtime

---

## Phase 1: VQ1 (May–June 2026)
- [ ] Simulator adapter for real DCL API
- [ ] Classical detector tuned for VQ1 environment
- [ ] Complete full course without crashing
- [ ] Submit qualifying run

## Phase 2: VQ2 (June–July 2026)
- [ ] Neural detector deployed
- [ ] MPCC operational
- [ ] Qualify for physical round

## Phase 3: Physical Qualifier (September 2026)
- [ ] Camera recalibration tools
- [ ] ESKF auto-tuning
- [ ] Domain randomization in training
- [ ] Qualify for finals

## Phase 4: Finals (November 2026)
- [ ] Adaptive race strategy
- [ ] Multiple config presets
- [ ] WIN
