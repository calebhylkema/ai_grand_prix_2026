# Safety

**Rule: never weaken safety or failsafe logic without explicit human approval.**
This covers watchdogs, rate/thrust clamps, tilt/collision aborts, envelope
protection, and the calibrated command boundary. There is no single `safety.py` —
the safety ladder lives inline in the controllers; any diff touching it gets the
`/review` safety review.

## Active failsafes (VQ2 deployment runtime)

| Failsafe | Trigger | Action |
|---|---|---|
| Tilt abort | attitude > 60° from level | end attempt |
| Collision abort | ≥ 3 collision events | end attempt |
| Stale-feed abort | camera/IMU/RACE_STATUS stream goes stale | end attempt |
| Rate/thrust clamps | every command | clamp to limits before send |
| Blind = brake | steering target lost | brake toward hover, never scan-and-drift |

All aborts print a reason code in the run summary.

## Command-boundary rules

- Calibration (`cmd = desired / GAIN`) is applied at exactly **one** boundary.
  Applying it twice, or adding sign hacks downstream, is how controllers historically
  tumbled. Changes here require explicit justification and review.
- Clamp all outputs to physical limits before sending; rate-limit command changes
  (no instantaneous jumps).
- MUST NOT use `time.sleep()` in the control loop.

## Estimator trust rules (safety-relevant, each traced to a crash)

- IMU-integrated vertical velocity is **diagnostic only** — never servo thrust on it.
- Velocity-nulling brakes only while the estimate has a fresh vision anchor;
  stale ⇒ level attitude and let drag work.
- ESKF gravity updates are ground-only (in-flight they erase real pitch).

## Control hierarchy (design ladder)

1. Primary controller (active mission logic)
2. Hover/brake — when the state estimate is unreliable
3. Abort — on any failsafe trigger

Fallback transitions are logged with reason codes.
