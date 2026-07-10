# Trajectory & control

How commands get made, and the control lessons that cost days to learn.

## First principles (flight-validated)

- **Yaw does not steer.** Yaw points the nose (and the camera); only tilting
  (roll/pitch) redirects travel. Steering = bank; camera pointing = yaw; a corner
  needs both, decoupled — path geometry and camera swing are separate jobs.
- **The command boundary must be calibrated once, centrally.** The sim's rate
  commands were inverted and 2.5× scaled; before calibration every correction
  diverged into a tumble, and controllers accumulated compensating sign hacks.
  See [calibrate controls](../how-to/calibrate-controls.md).
- **Blind = brake.** When the steering target is lost, brake toward hover — never
  scan-and-drift, never chase a stale estimate. This single rule ended the
  tumble era.
- **Aim then go:** forward pitch only when centred in both axes; otherwise brake
  and let yaw+bank+climb centre first. Re-acquisition from hover must be
  aim-then-go — a coordinated turn is a speed maneuver.
- **Tilt-compensate thrust** (1/(cosR·cosP)) so banking doesn't bleed altitude.
- Open-loop ("blind") maneuvers don't repeat: a fixed-bank arc at ±30% entry speed
  swings its radius ±70% (R = v²/(g·tanφ)) in a corridor metres wide. Anything
  precise must be closed on position or vision.

## Control ladder

1. Primary controller (mission logic / MPCC when it arrives)
2. PID cascade fallback (position → velocity → attitude → rate)
3. Hover/brake when the state estimate is unreliable
4. Abort on envelope violation

Fallback transitions are logged with reason codes. Clamp outputs, rate-limit
command changes, watch for oscillation. See [safety](../reference/safety.md).

## Planning: racing line + MPCC

Two layers: a global cubic B-spline through gate centers with a curvature-based
speed profile, refined online by MPCC (maximize path progress, minimize contouring
error; CasADi solver, 20–40 step horizon, p95 22 ms warm-started).

```
minimize: -w_progress·θ̇ + w_contour·e_c² + w_lag·e_l² + w_input·‖u‖²
s.t. quadrotor dynamics, thrust/rate limits, gate tunnels, progress monotonicity
```

MPCC++ (arXiv:2403.17551, RSS 2024) adds safety constraints and Bayesian tuning —
100% gate success matching RL lap times. The planner must not assume gates are
visible (use the estimated map) and must widen tunnels when the map is uncertain.

## RL: the performance ceiling, not the baseline

- Swift (Nature 2023) beat human champions with PPO + teacher-student distillation,
  mostly via 40 ms sensorimotor latency (vs human 220 ms) — but was brittle to
  condition changes.
- MonoRace (A2RL 2026 winner) used a G&CNet-style 3×64 MLP at 500 Hz.
- Our stance (and the AlphaPilot lesson): **reliability wins championships.**
  Classical + MPCC is the qualifying baseline; RL is layered on only if the
  baseline comfortably qualifies. Our PPO best checkpoint (400k steps) is usable;
  later checkpoints regressed — don't retrain without the real sim distribution.

## Obstacles

Per [ADR-0002](../decisions/0002-vq2-obstacle-avoidance.md): no full 3D avoidance.
The racing line through gate centers inherently clears flanking obstacles; the
real killers were data-association fly-offs and threading precision. A lightweight
optical-flow looming net exists as a safety layer for the physical stage.

## Speed evolution (context)

Winning speeds: ~0.6 m/s (2017, classical) → 8 m/s (AlphaPilot 2019) → 17 m/s
(MPCC era) → 22 m/s (Swift 2023, stereo) → **28 m/s (MonoRace 2026, monocular+IMU —
our sensor suite)**. Speed comes from trajectory quality and latency, not bravado.
