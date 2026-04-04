# Planning Module — Racing Line & Trajectory Optimization

## Purpose

Generate time-optimal trajectories through gate sequences and refine them online using receding-horizon optimization. The planner bridges the gap between "where are the gates" and "what motor commands to send."

## Two-Layer Architecture

### Layer 1: Global Racing Line (offline/once per course)
- Fit a cubic B-spline through gate centers with configurable waypoints before/after each gate
- Apply curvature-based speed profile: slow in tight turns, max speed on straights
- Optional: Use CPC (Complementary Progress Constraints) for theoretically time-optimal trajectories
- Output: Reference path parameterized by arc length, with speed profile

### Layer 2: Receding Horizon Refinement (online, 50–100 Hz)
- MPCC (Model Predictive Contouring Control): maximize progress along reference path while minimizing contouring error
- Horizon: 20–40 steps at 50ms intervals = 1–2 seconds lookahead
- Quadrotor dynamics model: 6-DOF rigid body with thrust and torque inputs
- Solver: CasADi + acados for real-time NMPC (target <20ms solve time)

## MPCC Formulation

```
minimize: -w_progress * θ_dot + w_contour * e_c² + w_lag * e_l² + w_input * ||u||²
subject to:
  - Quadrotor dynamics (point-mass or full 6-DOF)
  - Thrust limits: 0 ≤ T ≤ T_max
  - Angular rate limits: |ω| ≤ ω_max
  - Gate passage constraints (prismatic tunnels around gates)
  - Progress monotonicity: θ_{k+1} > θ_k
```

Key tuning parameters in `config/planner.toml`:
- `w_progress`: Aggressiveness (higher = faster but riskier)
- `w_contour`, `w_lag`: Tracking precision
- Gate tunnel width: Safety margin around gates

## Alternative: RL Policy (parallel development track)

- G&CNet architecture: 3-layer MLP (64 neurons each), state → motor commands
- Trained with PPO in simulation with domain randomization
- Runs at 500 Hz on microcontroller — bypasses MPCC entirely
- Use MPCC as safety fallback if RL policy commands exceed constraints

## Key Constraints

- Planner MUST NOT assume gates are visible — use estimated gate map from ESKF
- Planner MUST handle gate ordering: next gate, gate after next, etc.
- If gate map is uncertain (high covariance), widen tunnel constraints
- Racing line MUST be recomputable when gate estimates are refined
- All trajectories include feasibility check: can the drone physically execute this?
