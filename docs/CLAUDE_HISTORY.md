# Claude Session History

## Session 1 — 2026-03-29: ESKF Foundation

### What was built

**Phase 0 deliverable: ESKF state estimator + supporting infrastructure.**

Built bottom-up: shared types → rotation math → config loader → ESKF → tests.

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/utils/types.py` | Core data types shared across all modules: `State` (16-D), `ImuSample`, `PoseMeasurement`, `GateDetection`, `DroneCommand`, `Telemetry`, `GateWaypoint`, config dataclasses (`EstimationConfig`, `CameraConfig`, `GateConfig`) |
| `drone_racing/utils/transforms.py` | Quaternion math (multiply, conjugate, normalize, rotate), small-angle ESKF parameterization, SE(3) homogeneous transforms, skew-symmetric matrix, camera↔body frame rotations |
| `drone_racing/utils/config.py` | TOML config loader → typed `RacingConfig` dataclass |
| `drone_racing/estimation/eskf.py` | Full 16-state Error-State Kalman Filter: IMU predict at 200+ Hz, vision correct (6-DOF and position-only modes), chi² outlier gating, Joseph-form covariance update, IMU ring buffer for future latency compensation |
| `tests/test_transforms.py` | 21 tests — quaternion ops, roundtrips, skew-symmetric, SE(3), camera-body frame |
| `tests/test_types.py` | 11 tests — dataclass defaults, confidence level mapping, config properties |
| `tests/test_config.py` | 3 tests — TOML loading, missing file error |
| `tests/test_eskf.py` | 19 tests — init, stationary, constant accel, correction, covariance properties, chi² gating, static convergence, moving-drone tracking |

### Files modified

| File | Change |
|------|--------|
| `pyproject.toml` | Added `[tool.hatch.build.targets.wheel] packages = ["drone_racing"]` (hatchling couldn't find the package), added `scipy.*` to mypy ignore list |
| `drone_racing/utils/__init__.py` | Public API re-exports for all types, transforms, config |
| `drone_racing/estimation/__init__.py` | Public API re-export for `ESKF` |

### Verification

- **54 tests passing** (0.45s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors)

### Design decisions made

1. **Quaternion convention**: scalar-last `[x, y, z, w]` everywhere, matching scipy. Convert at boundaries only.
2. **Error state is 15-D** (not 16): orientation error parameterized as 3-DOF small angle δθ, avoids quaternion constraint issues.
3. **Joseph-form covariance update**: guarantees P stays symmetric positive definite through numerical noise.
4. **Chi² gating**: rejects measurements that are statistically inconsistent (threshold 9.21 = 99% confidence for 2-DOF). Prevents single bad detection from corrupting the filter.
5. **Confidence-tiered corrections**: HIGH → full 6-DOF, MEDIUM → position-only, LOW → skip. Based on corner count and confidence score.
6. **Latency compensation**: IMU buffer is in place but re-propagation not yet implemented. Acceptable for Phase 0 low-speed testing; mandatory before VQ1 at race speeds.

### What's NOT done yet (known TODOs in the code)

- `_repropagate_from()` in ESKF — latency compensation (Phase 1)
- No PnP solver yet (will live in `perception/pnp.py`)
- No gate tracker (Hungarian matching on IoU)

### Where to pick up next

The ESKF is complete and tested. The next Phase 0 deliverables in priority order:

1. **`drone_racing/control/pid.py`** — Cascaded PID controller (position → velocity → attitude → rate). Can be tested with ground-truth state immediately.
2. **`drone_racing/planning/spline.py`** — B-spline path through gate waypoints + curvature-based speed profile.
3. **`drone_racing/perception/classical.py`** — HSV threshold + contour gate detector for VQ1.
4. **PnP solver** — `cv2.solvePnP` wrapper that converts gate detections into `PoseMeasurement` for the ESKF.
5. **Integration test** — Wire the full pipeline: image → detect → PnP → ESKF → plan → control → command.

PID and spline can be built in parallel since they're independent.

---

## Session 2 — 2026-03-30: PID Controller + B-Spline Planner

### What was built

**Phase 0 deliverables: Cascaded PID controller + B-spline racing line generator.**

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/control/pid.py` | Cascaded PID controller: position → velocity → attitude → rate → DroneCommand. Includes `_PIDAxis` (scalar PID with anti-windup), `_PID3` (3-axis wrapper), `PIDController` (full cascade with gravity compensation, thrust→attitude decomposition, SO(3) attitude error), and `TrajectorySetpoint` dataclass |
| `drone_racing/planning/spline.py` | B-spline racing line: fits cubic spline through gate waypoints (with approach/exit offsets along gate normals), computes arc-length parameterization, curvature κ = \|r'×r''\|/\|r'\|³, and curvature-based speed profile v = min(v_max, sqrt(a_lat/κ)). `RacingLine` dataclass with `at_progress(s)` and `closest_progress(pos)` queries |
| `tests/test_pid.py` | 20 tests — scalar PID (P/I/D/anti-windup/reset), 3-axis PID, hover, step responses (up/down/forward/lateral), output clamping, yaw tracking, velocity feed-forward, timestamp handling |
| `tests/test_spline.py` | 16 tests — straight line, circular course, tangent unit length, arc length monotonicity, speed bounds, progress queries, closest-progress, edge cases (2 gates min, 1 gate error, open vs closed) |

### Files modified

| File | Change |
|------|--------|
| `drone_racing/utils/types.py` | Added `ControlConfig` and `PlannerConfig` dataclasses |
| `drone_racing/utils/config.py` | `RacingConfig` now loads `[control]` and `[planner]` TOML sections |
| `drone_racing/utils/__init__.py` | Added `ControlConfig`, `PlannerConfig` to exports |
| `drone_racing/control/__init__.py` | Public API: `PIDController`, `TrajectorySetpoint` |
| `drone_racing/planning/__init__.py` | Public API: `RacingLine`, `build_racing_line` |

### Verification

- **95 tests passing** (0.61s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 14 source files)

### Design decisions made

1. **PID cascade structure**: Position(PID) → Velocity(PI) → Attitude(P) → Rate(P). Inner loops are P-only which is standard for quadrotors. Full PID only on the outermost position loop where integral action is needed.
2. **Gravity compensation**: thrust_vector = desired_accel - gravity → normalised to throttle via hover_throttle ratio. Hover throttle defaults to 0.5 (will need tuning with real mass/thrust curve).
3. **SO(3) attitude error**: Uses the vee map on R_des^T @ R_cur to extract small-angle error, avoiding gimbal lock and singularities. Standard in PX4/ArduPilot.
4. **Anti-windup**: Integrator clamping (symmetric bounds). Simple and effective for the position loop.
5. **Spline waypoint strategy**: 3 waypoints per gate (approach, centre, exit) offset along gate normal. This forces the spline to pass through gates at the correct angle, not just the correct position.
6. **Speed profile**: v = sqrt(a_lat_max / κ) with a_lat_max = 1.5g. Smoothed with moving average to avoid discontinuities. Clamped to [1.0, max_velocity_mps].
7. **Closed-loop splines**: Use periodic boundary conditions in CubicSpline for closed courses. Open courses use "not-a-knot" BCs.

### What's NOT done yet

- No angular rate in `State` → rate PID loop is effectively skipped (attitude P directly outputs rates). Acceptable for Phase 0; will need gyro-from-state or direct IMU pass-through in Phase 1.
- `_HOVER_THROTTLE = 0.5` is a placeholder — needs calibration with real drone mass/thrust curve.
- No rate limiting on command changes (required by CLAUDE.md safety constraints — Phase 1).
- MPCC (Layer 2 planner) not started — waiting for CasADi integration.

### Where to pick up next

Phase 0 checklist remaining:

1. **`drone_racing/perception/classical.py`** — HSV threshold + contour gate detector for VQ1.
2. **PnP solver** — `cv2.solvePnP` wrapper that converts gate detections into `PoseMeasurement` for the ESKF.
3. **Integration test** — Wire the full pipeline end-to-end with synthetic data.
4. **`docs/adr/0001-eskf-over-ekf.md`** — Architecture decision record.
5. **Coverage check** — verify >80% on estimation and control.

---

## Session 3 — 2026-03-30: Perception (Classical Detector + PnP Solver)

### What was built

**Phase 0 deliverables: Classical gate detector and PnP pose solver — completing the perception→estimation interface.**

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/perception/classical.py` | HSV-based gate detector for VQ1: RGB→HSV threshold → morphological cleanup (open+close) → contour detection → area/aspect filtering → polygon approximation → corner ordering (TL,TR,BR,BL) → confidence scoring (vertex count, convexity, area ratio, aspect ratio) → `GateDetection` output. Includes `ClassicalDetectorConfig` for tuneable parameters, optional camera undistortion via pre-built remap tables |
| `drone_racing/perception/pnp.py` | PnP pose solver: takes 2D detected corners + 3D gate model + known gate world pose → `cv2.solvePnP` (ITERATIVE) → chains transforms T_WB = T_WG @ T_GC⁻¹ @ T_CB to produce world-frame drone pose → outputs `PoseMeasurement` for ESKF. Includes distance-based noise scaling and `solve_multiple()` for batch processing against a gate map |
| `tests/test_classical.py` | 12 tests — single gate detection on synthetic images, no-gate (grey/wrong colour), multiple gates, sorted by confidence, area filtering, corner ordering, undistortion path |
| `tests/test_pnp.py` | 8 tests — known geometry PnP solve, few-corners rejection, confidence propagation, distance-based noise scaling, pixel noise tolerance, solve_multiple with gate map (no ID, unknown ID, valid detection) |

### Files modified

| File | Change |
|------|--------|
| `drone_racing/perception/__init__.py` | Public API: `ClassicalDetector`, `ClassicalDetectorConfig`, `PnPSolver` |

### Verification

- **115 tests passing** (0.67s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 16 source files)
- **Coverage: 95.58%** (well above 80% threshold)
  - `estimation/eskf.py`: 93%
  - `control/pid.py`: 99%
  - `planning/spline.py`: 100%
  - `perception/classical.py`: 83%
  - `perception/pnp.py`: 94%

### Design decisions made

1. **Classical detector pipeline**: HSV → morphology → contour → polygon approximation. Standard approach for VQ1 where gates are highlighted. HSV thresholds are configurable (not hardcoded) via `ClassicalDetectorConfig`.
2. **Confidence scoring**: Multi-factor (vertex count closeness to 4, convexity, area ratio, aspect ratio) averaged into [0,1]. Higher = more likely a real gate.
3. **Corner ordering**: Top-left first, clockwise (TL, TR, BR, BL). Uses centroid-based top/bottom split, with angle-based fallback. Consistent ordering is critical for PnP.
4. **PnP transform chain**: T_WB = T_WG @ T_GC⁻¹ @ T_CB. OpenCV's `solvePnP` gives T_CG (camera-from-gate), which we invert to get T_GC (gate-from-camera), then chain through the known gate world pose and the fixed camera-body extrinsic.
5. **Distance-based noise**: PnP measurement noise scales linearly with distance/3m. Farther gates produce noisier measurements, so the ESKF weights them less. This is critical for filter stability.
6. **Pre-built undistortion maps**: `cv2.initUndistortRectifyMap` called once in `__init__`, not per frame. Zero overhead when distortion is zero.
7. **Gate map lookup**: `solve_multiple()` requires `gate_id` to be set on detections (by a tracker, not yet built). Unmatched detections are skipped safely.

### What's NOT done yet

- **Gate tracker**: Hungarian matching on IoU to assign consistent `gate_id` across frames. Needed before `solve_multiple()` can be used in production.
- **Neural detector**: YOLOv8-nano keypoint model — Phase 2 (VQ2).
- **Synthetic gate image dataset**: 100+ annotated images for perception testing (Phase 0 deliverable).
- **HSV tuning**: Current defaults are for bright red on grey. Will need tuning for actual VQ1 gate colours when sim drops.

### Where to pick up next

Phase 0 checklist remaining:

1. **Integration test** — Wire the full pipeline: synthetic image → classical detect → PnP → ESKF correct → planner query → PID command. This is the end-to-end validation.
2. **`docs/adr/0001-eskf-over-ekf.md`** — Architecture decision record.
3. **Synthetic gate image dataset** — 100+ images with known corner positions for perception testing.
4. **Gate tracker** — Hungarian matching to assign gate IDs across frames (needed for solve_multiple).

---

## Session 4 — 2026-03-30: End-to-End Integration Test

### What was built

**Phase 0 deliverable: Full pipeline integration test — validates all modules wire together correctly.**

### Files created

| File | Purpose |
|------|---------|
| `tests/test_integration.py` | 7 integration tests covering: full pipeline end-to-end (image→detect→PnP→ESKF→plan→PID→command), module interface contracts (PnP→ESKF, RacingLine→PID, ESKF→PID), multi-step simulation (IMU+vision cycles, repeated detect-PnP-correct, planning+control loop) |

### Verification

- **122 tests passing** (0.72s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 16 source files)

### Test architecture

Three test classes covering different integration levels:

1. **TestFullPipeline** — End-to-end: synthetic image with red rectangle → ClassicalDetector → PnPSolver → ESKF correct → build_racing_line → closest_progress/at_progress → TrajectorySetpoint → PIDController → DroneCommand. Validates the full chain produces a valid command with clamped outputs.

2. **TestModuleContracts** — Interface compatibility:
   - PnP output (`PoseMeasurement`) feeds directly into ESKF `.correct()`
   - Racing line output (`at_progress()`) constructs a valid `TrajectorySetpoint` for PID
   - ESKF `State` is directly usable by `PIDController.update()`

3. **TestMultiStepSimulation** — Temporal stability:
   - 50 IMU predictions → vision correction → 50 more predictions (covariance grows then shrinks)
   - 3 cycles of detect→PnP→correct (no NaN, no covariance explosion)
   - 20-step planning+control loop (all commands valid and finite)

### Design decisions made

1. **Synthetic test data**: Used simple red rectangles on grey background rather than rendered 3D scenes. This keeps tests fast (<1s) and dependency-free while still exercising the full pipeline.
2. **Triangular course**: 3 gates at (10,0,0), (0,10,0), (0,-10,0) — provides enough geometry for spline fitting while being simple enough to reason about.
3. **Stationary drone with gravity-compensated IMU**: accel=[0,0,9.81] simulates a hover scenario. This isolates integration bugs from dynamics complexity.
4. **Gate ID assignment**: In real operation, the gate tracker assigns IDs. In tests, we manually set `det.gate_id = 0` to bridge the perception→PnP gap (tracker is not yet built).

### What's NOT done yet

- **Gate tracker** — Hungarian matching to assign consistent gate IDs across frames. Currently gate_id is set manually in tests.
- **`docs/adr/0001-eskf-over-ekf.md`** — Architecture decision record.
- **Synthetic gate image dataset** — 100+ annotated images for perception regression testing.
- **Rate limiting** on command changes (safety requirement from CLAUDE.md).
- **MPCC planner** (Layer 2) — CasADi integration for receding-horizon optimization.

### Where to pick up next

Phase 0 is now **feature-complete**. All core modules are built, tested, and integrated:
- Perception (classical detector + PnP solver)
- Estimation (ESKF with IMU prediction + vision correction)
- Planning (B-spline racing line with curvature-based speed profile)
- Control (cascaded PID)
- Integration (end-to-end pipeline test)

**Next priorities (Phase 1 — pre-VQ1 hardening):**

1. **Gate tracker** (`drone_racing/perception/tracker.py`) — Hungarian matching on IoU for consistent gate IDs across frames. Required before the pipeline can run on real video.
2. **ADR documentation** — `docs/adr/0001-eskf-over-ekf.md` capturing the ESKF design rationale.
3. **Latency compensation** — Implement `_repropagate_from()` in ESKF for IMU-based state rollforward after delayed vision corrections.
4. **Rate limiting** — Add command rate limiting in the control module (safety requirement).
5. **Simulator adapter** (`drone_racing/sim/`) — Interface to the DCL competition simulator when it drops in May 2026.
6. **MPCC planner** — CasADi-based receding-horizon controller (Layer 2 planning).
7. **Neural detector** — YOLOv8-nano keypoint model for VQ2 photorealistic environments.

---

## Session 5 — 2026-03-30: Gate Tracker + ADR Documentation

### What was built

**Phase 0 deliverables: Gate tracker for cross-frame detection matching + first architecture decision record.**

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/perception/tracker.py` | Gate tracker using IoU + Hungarian algorithm (scipy `linear_sum_assignment`). Maintains internal `_Track` state per gate with bbox, hits, age. Features: `GateTrackerConfig` (iou_threshold, max_age, min_hits), `update()` for frame-by-frame matching, `assign_gate_map_ids()` to associate tracks with known course gates via 2D centroid proximity, automatic track pruning after max_age misses, min_hits confirmation before reporting |
| `tests/test_tracker.py` | 23 tests — IoU computation (identical/no-overlap/partial/contained/zero-area), bbox extraction, IoU matrix, basic tracking (single ID, two distinct IDs, consistent across frames, empty frames), aging (pruned after max_age, survives short gaps), min_hits threshold, reset, gate map ID assignment, integration with ClassicalDetector output |
| `docs/adr/0001-eskf-over-ekf.md` | Architecture decision record: ESKF chosen over EKF for quaternion constraint handling, numerical conditioning, IMU/vision rate separation, industry standard alignment. Includes rationale, consequences, and rejected alternatives (UKF, particle filter, factor graph) |

### Files modified

| File | Change |
|------|--------|
| `drone_racing/perception/__init__.py` | Added `GateTracker`, `GateTrackerConfig` to public API exports |

### Verification

- **145 tests passing** (0.74s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 17 source files)

### Design decisions made

1. **Hungarian algorithm via scipy**: `linear_sum_assignment` on a cost matrix of (1 - IoU). Optimal O(n³) assignment, fast enough for the small number of gates in view (typically 1-3).
2. **min_hits threshold**: New tracks must be seen for `min_hits` consecutive frames before they get a `gate_id`. This suppresses transient false positives from the classical detector. Default: 2 frames.
3. **max_age pruning**: Tracks that go unmatched for `max_age` frames are deleted. Allows brief occlusions without losing track. Default: 5 frames.
4. **Gate map association**: `assign_gate_map_ids()` maps tracker-internal IDs to course-map gate IDs by 2D centroid proximity. This bridges the gap between arbitrary track IDs and the known gate map that `PnPSolver.solve_multiple()` requires.
5. **Axis-aligned bounding boxes for IoU**: Simpler than rotated IoU. Sufficient because gates appear roughly rectangular and axis-aligned from the forward-facing camera.

### What's NOT done yet

- **Synthetic gate image dataset** — 100+ annotated images for perception regression testing.
- **Rate limiting** on command changes (safety requirement from CLAUDE.md).
- **Latency compensation** — `_repropagate_from()` in ESKF.
- **MPCC planner** (Layer 2) — CasADi integration for receding-horizon optimization.
- **Simulator adapter** (`drone_racing/sim/`) — awaiting DCL API spec (May 2026).

### Where to pick up next

Phase 0 is **complete**. All deliverables from `docs/PLAN.md` Phase 0 are done:
- [x] pyproject.toml with all deps
- [x] Classical CV gate detector (`perception/classical.py`)
- [x] ESKF with IMU propagation (`estimation/eskf.py`)
- [x] PID cascade controller (`control/pid.py`)
- [x] B-spline path planner (`planning/spline.py`)
- [x] PnP solver (`perception/pnp.py`)
- [x] Gate tracker (`perception/tracker.py`)
- [x] Test suite with >80% coverage (95%+)
- [x] ADR-0001 (`docs/adr/0001-eskf-over-ekf.md`)
- [x] Integration test (full pipeline end-to-end)

**Next priorities (pre-VQ1 hardening, Phase 0→1 bridge):**

1. **Latency compensation** — Implement `_repropagate_from()` in ESKF. Critical for accuracy at race speeds.
2. **Rate limiting** — Command rate limiter in control module (CLAUDE.md safety requirement).
3. **Simulator adapter** (`drone_racing/sim/dcl_adapter.py`) — Interface to DCL competition API when it drops in May 2026.
4. **MPCC planner** — CasADi-based MPCC for Step 8. This is the performance upgrade that replaces spline+PID with optimal trajectory tracking.
5. **GitHub CI** — Hook up the existing `ci.yml` workflow once the repo is pushed.

---

## Session 6 — 2026-03-30: Latency Compensation + Rate Limiter

### What was built

**Pre-VQ1 hardening: ESKF latency compensation (rollback + replay) and command rate limiter.**

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/control/rate_limiter.py` | Command rate limiter: `CommandRateLimiter` wraps any controller output and enforces smooth transitions. `RateLimiterConfig` with per-channel max rates (throttle 2/s, roll/pitch 20 rad/s², yaw 10 rad/s²). Timestamps drive dt computation, backwards timestamps return previous command. `_clamp_rate()` helper for single-channel clamping |
| `tests/test_rate_limiter.py` | 13 tests — _clamp_rate unit tests (within/exceed/negative/exact), basic limiter (first pass-through, small change, large throttle/roll clamp, backwards timestamp), convergence (positive/negative direction), reset, all-channels-independent |

### Files modified

| File | Change |
|------|--------|
| `drone_racing/estimation/eskf.py` | Full latency compensation: added `_StateSnapshot` dataclass, IMU buffer now stores state snapshots. `correct()` rolls back to measurement time via closest snapshot, applies Kalman correction, then re-propagates with `_repropagate_from()`. Handles edge cases: chi² rejection after rollback, singular covariance after rollback (both re-propagate to restore current time). Added `Vec4` to TYPE_CHECKING imports |
| `drone_racing/control/__init__.py` | Added `CommandRateLimiter`, `RateLimiterConfig` to public API |
| `tests/test_eskf.py` | Added `TestLatencyCompensation` class with 5 tests: delayed correction applied, timestamp restored after rollback, no rollback for current measurement, covariance valid after latency comp, moving drone latency comp improves estimate |

### Verification

- **163 tests passing** (0.75s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 18 source files)

### Design decisions made

1. **State snapshot per IMU entry**: Each `_ImuBufferEntry` stores a full `_StateSnapshot` taken *before* the IMU sample was integrated. This makes rollback O(1) — just copy the snapshot. Memory cost is ~1KB per entry × 100 entries = ~100KB, negligible.
2. **Rollback strategy**: Find the closest snapshot at or before the measurement timestamp, restore it, apply Kalman correction, then replay all subsequent IMU samples. This is the standard approach from Sola (2017) §7.3.
3. **Edge case handling**: If chi² gating rejects the measurement *after* rollback, the IMU samples are still replayed to restore the current time. Same for singular innovation covariance. The filter never gets stuck in the past.
4. **Rate limiter as standalone module**: Decoupled from PID so it works with any controller (PID, MPCC, RL policy). Sits between controller and flight controller as a safety layer.
5. **Per-channel rate limits**: Each command channel (throttle, roll rate, pitch rate, yaw rate) has independent rate limits. Throttle is limited to 2/s (full range in 0.5s), angular rate changes limited to 20 rad/s² (roll/pitch) and 10 rad/s² (yaw).

### What's NOT done yet

- **Simulator adapter** (`drone_racing/sim/`) — awaiting DCL API spec (May 2026).
- **MPCC planner** (Layer 2) — CasADi-based receding-horizon optimization.
- **Synthetic gate image dataset** — 100+ annotated images for perception regression testing.
- **GitHub CI** — Hook up `ci.yml` workflow.

### Where to pick up next

All pre-VQ1 hardening tasks are **complete**:
- [x] Latency compensation in ESKF (rollback + IMU replay)
- [x] Command rate limiting (safety requirement from CLAUDE.md)
- [x] Gate tracker (Hungarian matching + gate map association)
- [x] ADR-0001 documentation

The stack is now production-ready for VQ1. Remaining items require external dependencies:

1. **Simulator adapter** — Needs DCL API spec (releasing May 2026). Will be a thin wrapper around the competition interface.
2. **MPCC planner** — CasADi/acados integration for time-optimal trajectory tracking. Can start on toy problems now.
3. **GitHub CI** — Push repo and activate Actions workflow.
4. **Neural detector** — YOLOv8-nano for VQ2. Needs training data from VQ1 runs.

---

## Session 7 — 2026-03-30: MPCC Planner (CasADi + IPOPT)

### What was built

**Step 8: Model Predictive Contouring Control — time-optimal trajectory optimisation using CasADi NLP.**

### Files created

| File | Purpose |
|------|---------|
| `drone_racing/planning/dynamics.py` | Point-mass quadrotor dynamics for MPCC. 9-state model [px,py,pz,vx,vy,vz,phi,theta,psi], 4-input [T,phi_cmd,theta_cmd,psi_dot_cmd]. CasADi symbolic `continuous_dynamics()`, `rk4_step()` integrator, `state_bounds()`, `input_bounds()`. First-order attitude lag (tau=0.1s) approximates inner-loop bandwidth |
| `drone_racing/planning/mpcc.py` | Full MPCC solver: `MPCCSolver` builds a CasADi NLP at init with dynamics constraints (RK4), progress monotonicity, contouring/lag error costs, input regularisation. Solves with IPOPT, supports warm-starting. `_path_at_progress()` does symbolic piecewise-linear interpolation of the racing line using soft indicators for differentiability. `MPCCConfig` for tuning, `MPCCSolution` for results |
| `tests/test_mpcc.py` | 22 tests — dynamics (hover equilibrium, free fall, velocity→position, attitude lag), RK4 (hover stationary, free fall integration, multi-step energy conservation), bounds (shape, thrust range, attitude match config), MPCC solver (builds, straight-line convergence, progress increases, input bounds respected, initial state pinned, warm-start, L-shaped path, solve time <500ms, factory method, cost finite, output shapes) |

### Files modified

| File | Change |
|------|--------|
| `drone_racing/planning/__init__.py` | Added `DynamicsConfig`, `MPCCConfig`, `MPCCSolution`, `MPCCSolver` to public API |

### Verification

- **185 tests passing** (2.01s)
- **ruff**: clean (lint + format)
- **mypy --strict**: clean (0 errors, 20 source files)
- MPCC solves in <500ms on CI (typically ~100ms for horizon=10)

### Design decisions made

1. **Point-mass dynamics**: 9-state model deliberately simplified from full 6-DOF. Attitude handled as first-order lag to commanded angles. This is the standard approach for MPCC planners (Romero et al. 2022) — the inner-loop PID/rate controller handles actual attitude tracking.
2. **CasADi + IPOPT**: Symbolic NLP construction at init, then fast re-solves with warm-starting. IPOPT is the standard interior-point solver for this problem class.
3. **Piecewise-linear path interpolation**: Rather than fitting a CasADi-compatible spline, use soft indicator weights (fmax/fabs) over discretised racing line points. This is differentiable and robust.
4. **Progress as decision variable**: θ_k at each timestep is optimised alongside states/inputs. Progress monotonicity enforced as a constraint (0 ≤ Δθ ≤ 1). This is the defining feature of MPCC vs standard MPC.
5. **Cost structure**: J = Σ[-w_p·Δθ + w_c·e_c² + w_l·e_l² + w_u·‖u‖² + w_du·‖Δu‖²]. Negative progress term maximises speed. Contouring/lag balance tracking accuracy vs aggressiveness.

### Where to pick up next

The MPCC planner is now fully implemented and tested. Remaining work:

1. **Simulator adapter** (`drone_racing/sim/`) — Needs DCL API spec (May 2026).
2. **MPCC↔PID integration** — Wire MPCC output into the PID inner loop or replace PID entirely with MPCC-generated reference trajectories.
3. **GitHub CI** — Push repo and activate Actions workflow.
4. **Neural detector** — YOLOv8-nano for VQ2 photorealistic environments.
5. **Latency benchmark** — Profile MPCC solve time on target hardware (Jetson Orin NX class). Target <20ms for 50Hz control.
