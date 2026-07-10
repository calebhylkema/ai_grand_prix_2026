# State estimation

How we estimate attitude/velocity/position from IMU + monocular gate sightings,
and where estimators lie.

## Why an error-state Kalman filter

Full rationale in [ADR-0001](../decisions/0001-eskf-over-ekf.md). Short version:
the ESKF parameterizes orientation error as an unconstrained 3-D small angle
(vs the EKF's constrained quaternion), keeping the covariance full-rank and
well-conditioned, cleanly separating high-rate IMU prediction from low-rate vision
correction, and matching the industry standard for VIO (PX4 EKF2, VINS-Mono,
MSCKF; canonical reference: Sola 2017, arXiv:1711.02508).

## The two implementations

- **Package (`drone_racing.estimation.eskf`):** 16-state
  (p, v, q, accel bias, gyro bias), pre-sim lineage, has a known frame-conversion
  drift — do not fly it until that is fixed.
- **VQ2 deployment (`drone_racing.deployments.vq2.eskf`):** 15-state error
  formulation, gate-as-landmark corrections, chi-squared gating.
  Validated synthetic: pos 0.23 m, vel 0.15 m/s, att 2.6°. This is the flight filter.

They stay separate until a deliberate consolidation (roadmap, pre-physical).

## Design essentials

- IMU predict at sensor rate; vision correct per validated PnP fix; re-anchor on
  active-gate change.
- Latency compensation matters at speed: at 100 km/h, 10 ms = 0.28 m. Buffer IMU
  samples; correct at measurement time; re-propagate.
- Measurement trust tiers: 6+ corners at 2–5 m → full 6-DOF; medium → position-only
  (de-rotated); <4 corners or far → skip, propagate on IMU.
- Use ALL visible gates for updates, not just the next target (AlphaPilot insight).
- Chi-squared gate every innovation; log innovation sequences for offline tuning.

## Where estimators lie (VQ2 flight-validated pitfalls)

- **The cruise lie:** gravity-alignment updates that gate on `|‖a‖−G|` pass their
  own test during cruise (|a| ≈ 1.05 g) and erase real nose-down pitch — the filter
  read 0.3–0.8 m/s mid-corner while true speed was 4–6 m/s. **Gravity updates must
  be ground-only.** Synthetic tests missed this because they had no cruise regime;
  test every regime the vehicle actually flies.
- **IMU-integrated vertical velocity is never control-grade near launch** — the
  launch transient pins it at a ~−3 m/s fiction. Diagnostic only.
- Velocity from finite-differenced PnP poses (static gate ⇒ d(p_rel)/dt = −v) with
  EMA smoothing and a pose-jump guard beats IMU integration for short horizons —
  but only while the vision anchor is fresh (~1.5 s); stale ⇒ trust nothing.
- Gyro-only attitude is fine over ~6 s blind stretches; longer needs vision fixes.

## Ground-truth validation rig

The VQ1 sim (`.local/simulators/vq1/`) still broadcasts ODOMETRY + ATTITUDE.
Fly the same estimator there and compare against ground truth before trusting it
in VQ2 — the only closed-loop validation available without telemetry.
