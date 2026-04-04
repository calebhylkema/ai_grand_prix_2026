# AI Grand Prix — Development Plan

## Mission
Build the fastest, most reliable autonomous drone racing stack for the Anduril AI Grand Prix 2026. Win $500K and a job offer.

---

## Phase 0: Foundation (Now → Sim Release, ~March–May 2026)

### Goals
- Project scaffolding complete and CI green
- Classical CV gate detector working on synthetic images
- ESKF implemented and tested with simulated IMU + fake gate detections
- PID cascade controller tracking simple trajectories
- MPCC solver running on toy 2D problems with CasADi

### Deliverables
- [ ] `pyproject.toml` with all deps, `uv lock` committed
- [ ] GitHub repo with Actions CI (ruff → mypy → pytest)
- [ ] `drone_racing/perception/classical.py` — HSV threshold + contour gate detector
- [ ] `drone_racing/estimation/eskf.py` — 16-state ESKF with IMU propagation
- [ ] `drone_racing/control/pid.py` — Cascaded PID controller
- [ ] `drone_racing/planning/spline.py` — B-spline path through waypoints
- [ ] Test suite with >80% coverage on estimation and control
- [ ] Synthetic gate image dataset (100+ images) for perception testing
- [ ] `docs/adr/0001-eskf-over-ekf.md` — first architecture decision record

### Risk Mitigation
- Interface specs due "second half of March" — design adapter pattern now so sim integration is a thin layer
- Train YOLOv8-nano on synthetic gate images before real sim data exists

---

## Phase 1: VQ1 — Virtual Qualifier Round 1 (May–June 2026)

### Environment
- Desaturated visuals, highlighted gates, high SNR
- Visual aids possibly active

### Strategy
Classical CV is likely sufficient here. Focus on **reliability over speed**.

### Deliverables
- [ ] Simulator adapter (`drone_racing/sim/dcl_adapter.py`) wrapping competition API
- [ ] Classical detector tuned for VQ1 environment
- [ ] Full pipeline running: camera → detect → PnP → ESKF → plan → control → commands
- [ ] Complete at least one full lap without crashing
- [ ] Latency profiling: target <40ms end-to-end
- [ ] Flight data logger recording full telemetry for replay

### Success Criteria
- **Minimum:** Complete the course. Any time counts.
- **Target:** Top 50% of qualifying times.
- **Stretch:** Top 20%.

---

## Phase 2: VQ2 — Virtual Qualifier Round 2 (June–July 2026)

### Environment
- Photorealistic, lighting distractions, lower SNR
- Visual aids OFF — gates blend into environment

### Strategy
Switch to neural gate detection. This is where the competition gets real.

### Deliverables
- [ ] YOLOv8-nano keypoint model trained on VQ1 flight data + augmented synthetic data
- [ ] ONNX export with TensorRT optimization
- [ ] Dual-mode perception: auto-detect environment and switch classical↔neural
- [ ] MPCC planner operational with CasADi/acados
- [ ] Curvature-based speed profile on racing line
- [ ] Bayesian hyperparameter tuning for MPCC weights (TuRBO)
- [ ] Begin RL policy training with PPO in simulator

### Success Criteria
- **Minimum:** Qualify for physical round (top N teams, N TBD).
- **Target:** Top 25% qualifying time.
- **Stretch:** Top 10%.

---

## Phase 3: Physical Qualifier (September 2026, Southern California)

### Challenge
Sim-to-real gap. Everything that "worked in simulation" will break in new ways.

### Strategy
- Arrive with robust, conservative default settings
- Use the 2-week on-site period for rapid calibration
- Camera intrinsics will differ — recalibrate immediately
- IMU noise characteristics will differ — re-tune ESKF process noise
- Lighting, reflections, ground texture — retrain detector on real frames ASAP

### Deliverables
- [ ] Domain randomization in training pipeline (varied lighting, blur, exposure)
- [ ] One-click calibration script for camera intrinsics
- [ ] ESKF auto-tuning: log innovation sequences, optimize Q/R matrices
- [ ] RL policy fine-tuned on real flight data (if time permits)
- [ ] Conservative mode toggle: wider gate tunnels, lower max speed
- [ ] Post-flight analysis dashboard for rapid debugging

### Success Criteria
- **Minimum:** Complete the course physically. Don't crash the drone.
- **Target:** Qualify for finals (top 10).

---

## Phase 4: Championship Finals (November 2026, Columbus, Ohio)

### Format
Head-to-head racing — consistency matters even more.

### Strategy
- Maximize completion rate above all else
- Gradually increase aggressiveness between heats
- If leading: maintain speed. If trailing: push harder.
- Exploit any course-specific advantages found in practice

### Deliverables
- [ ] Race strategy module: adaptive aggressiveness based on position
- [ ] Real-time telemetry dashboard for pit crew
- [ ] Multiple configuration presets: conservative, balanced, aggressive
- [ ] Crash recovery: if drone survives impact, can it resume?

---

## Parallel Workstreams (Ongoing Throughout)

### RL Policy Development
- Train PPO policy in simulator with domain randomization
- Teacher-student distillation (privileged teacher → vision-only student)
- G&CNet architecture: 3×64 MLP, state → motor commands at 500 Hz
- Benchmark against MPCC: faster? More reliable? Both?

### Data Pipeline
- Automated flight data collection in sim
- Gate detection labeling tools (semi-automated with model predictions)
- Training data versioning with DVC or similar
- Ablation study framework: test one change at a time

### Infrastructure
- Docker image for reproducible builds (CUDA + OpenCV + CasADi)
- GitHub Actions: lint → typecheck → unit tests → sim tests
- Performance regression dashboard
- Team knowledge base in `docs/`

---

## Key Decision Points

| Date | Decision | Options |
|------|----------|---------|
| March 2026 | Interface specs released | Adapt adapter layer |
| May 2026 | Sim available | Validate all assumptions |
| June 2026 | VQ1 results | Classical CV sufficient or need neural for VQ1? |
| July 2026 | VQ2 results | MPCC vs RL as primary controller |
| August 2026 | Pre-physical prep | Which configuration to bring as default |

---

## Resource Allocation (Solo Developer)

| Week Budget | Allocation |
|-------------|------------|
| 40% | Core pipeline (perception → estimation → planning → control) |
| 25% | Testing, debugging, profiling |
| 20% | RL training and experimentation |
| 15% | Infrastructure, docs, tooling |

If adding team members, perception and control are the most parallelizable workstreams.
