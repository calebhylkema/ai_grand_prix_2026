# Control Module — Trajectory Tracking & Motor Commands

## Purpose

Convert planned trajectories into throttle/roll/pitch/yaw commands that the drone firmware accepts. This module is the final link before actuation — bugs here crash the drone.

## Control Hierarchy (Safety Ladder)

1. **Primary: MPCC output** — direct from planner when solving successfully
2. **Secondary: PID cascade** — if MPCC solver fails or times out
3. **Tertiary: Hover/brake** — if state estimate is unreliable (high covariance)
4. **Emergency: Kill motors** — if attitude exceeds safe envelope (>80° from horizontal)

Fallback transitions MUST be logged with reason codes.

## PID Cascade Structure (Baseline Controller)

```
Position PID → Velocity PID → Attitude PID → Rate PID → Motor Mixer
```

- Position loop: 50 Hz, outputs desired velocity
- Velocity loop: 100 Hz, outputs desired attitude (roll/pitch) + thrust
- Attitude loop: 250 Hz, outputs desired angular rates
- Rate loop: 500 Hz, outputs motor commands

Each loop runs at progressively higher frequency for stability.

## Output Interface

The simulator receives commands via MAVLink `SET_ATTITUDE_TARGET` (VADR-TS-001).
Our `DroneCommand` maps to MAVSDK's `AttitudeRate`:

```python
@dataclass
class DroneCommand:
    throttle: float     # 0.0 to 1.0 (normalized thrust) → thrust_value
    roll_rate: float    # rad/s → roll_deg_s (converted at boundary)
    pitch_rate: float   # rad/s → pitch_deg_s (converted at boundary)
    yaw_rate: float     # rad/s → yaw_deg_s (converted at boundary)
```

Conversion to MAVSDK: `mavsdk_adapter.py` converts rad/s → deg/s at the boundary.
Alternative: `SET_POSITION_TARGET_LOCAL_NED` for position/velocity control (NED frame).

## Safety Constraints

- MUST clamp all outputs to physical limits before sending
- MUST implement rate limiting on command changes (no instantaneous jumps)
- MUST monitor for oscillation (frequency analysis on recent commands)
- MUST log every command with timestamp for post-flight analysis
- control/safety.py is a PROTECTED FILE — requires human review for changes

## Tuning Notes

- PID gains in `config/control.toml` — NEVER hardcode
- Tune inner loops first (rate), then outer loops (position)
- Use Ziegler-Nichols for initial gains, then fine-tune in simulation
- Log tracking error histograms for each loop to identify the bottleneck
- For MPCC: use TuRBO Bayesian optimization to auto-tune cost weights
