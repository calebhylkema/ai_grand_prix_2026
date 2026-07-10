# ADR-0001: Error-State Kalman Filter (ESKF) over Extended Kalman Filter (EKF)

**Status:** Accepted
**Date:** 2026-03-29
**Author:** Caleb (with Claude)

---

## Context

The drone racing stack needs a state estimator that fuses high-rate IMU data (200-500 Hz) with asynchronous gate-based PnP pose corrections (20-60 Hz) to produce a continuous 16-dimensional state estimate (position, velocity, orientation, accel bias, gyro bias).

At racing speeds (up to 100 km/h), even 10ms of latency introduces ~0.28m of position error. The estimator must be fast, numerically stable, and handle the large dynamic range between IMU noise (small) and PnP measurement noise (distance-dependent, sometimes large).

Two main candidates were considered:

1. **Extended Kalman Filter (EKF):** Linearise the full nonlinear state dynamics around the current estimate. The state includes the quaternion directly (16-D state, 16-D error).
2. **Error-State Kalman Filter (ESKF):** Propagate a nominal state with the full nonlinear IMU model, and only estimate the *error* (deviation from nominal) with a Kalman filter. Orientation error is parameterised as a 3-D small angle, not a 4-D quaternion (15-D error state).

## Decision

We chose the **Error-State Kalman Filter (ESKF)**.

## Rationale

### 1. Quaternion constraint problem (the primary reason)

A standard EKF with a quaternion in the state vector has a 16-D state but only 15 degrees of freedom — the quaternion must satisfy ||q|| = 1. This unit-norm constraint causes problems:

- The covariance matrix is singular along the constraint direction (rank 15 in a 16-D space).
- Re-normalisation after each update is fragile and can interact badly with the covariance.
- The Jacobians of the dynamics with respect to the quaternion are more complex.

The ESKF sidesteps this entirely by parameterising orientation error as a 3-D small angle vector (delta-theta), which is unconstrained. The nominal quaternion is propagated separately with the full nonlinear rotation model and re-normalised. The error-state covariance is a well-conditioned 15x15 matrix with full rank.

### 2. Better numerical conditioning

The ESKF error state stays near zero between corrections (it gets reset after each Kalman update). This means:

- The linearisation point is always near the origin of the error state, where the small-angle approximation is most accurate.
- The EKF linearises around the full state, which can be far from the linearisation point if the dynamics are fast — exactly the case at racing speeds.
- The ESKF covariance matrix stays small and well-conditioned, while the EKF covariance operates on the full state scale.

### 3. Natural separation of IMU and vision rates

The ESKF cleanly separates the two update rates:

- **IMU predict (200+ Hz):** Propagate the nominal state with the full nonlinear equations. This is a simple integration step — no Jacobians needed for the nominal state.
- **Vision correct (20-60 Hz):** Run the linear Kalman correction on the small error state. Compute Jacobians (H matrix) once per correction, not once per IMU sample.

An EKF would need to compute Jacobians at every IMU prediction step, which is wasteful since most of the computational budget should go to the infrequent but expensive vision corrections.

### 4. Industry standard for IMU-vision fusion

The ESKF is the de facto standard for visual-inertial odometry (VIO) and drone state estimation:

- PX4 autopilot uses ESKF (via ECL/EKF2).
- ArduPilot's EKF3 is an error-state formulation.
- MSCKF, VINS-Mono, and most research VIO systems use ESKF.
- Joan Sola's "Quaternion kinematics for the error-state Kalman filter" (2017) is the canonical reference.

Using the industry-standard approach means:
- More reference implementations to validate against.
- Easier to debug with established tools and theory.
- Lower risk of subtle bugs that would only manifest at race speeds.

### 5. Simpler extension path

The ESKF's modular structure makes it straightforward to add:

- **Latency compensation:** Buffer IMU samples, roll back to measurement time, correct, re-propagate. The nominal/error split makes this clean.
- **Multi-gate corrections:** Each gate detection is an independent correction step. No need to batch.
- **Additional sensors:** Adding barometer, magnetometer, or optical flow just adds more correction steps — the predict side is untouched.

## Consequences

### Positive

- 15x15 covariance matrix is always full-rank and well-conditioned.
- IMU prediction is fast (no Jacobian computation for nominal propagation).
- Joseph-form covariance update guarantees positive-definiteness through numerical noise.
- Chi-squared gating on the error-state innovation is straightforward and well-calibrated.
- Clean separation of concerns between prediction and correction code paths.

### Negative

- Slightly more complex to implement correctly than a naive EKF (error injection, error-state reset, two separate state representations).
- Requires understanding of the error-state parameterisation — new team members need to read the Sola paper.
- The 3-D orientation error approximation (delta-theta ~ 2 * delta-q_vec) is only accurate for small errors. If the filter diverges badly, recovery is harder than with a full-state EKF. Mitigated by chi-squared gating and covariance monitoring.

### Alternatives Rejected

- **Unscented Kalman Filter (UKF):** Better linearisation accuracy but 2-3x more expensive (sigma point propagation). Not justified given the ESKF's small-angle approximation is already accurate at 200+ Hz IMU rates.
- **Particle Filter:** Too expensive for real-time on embedded hardware. Useful for multi-modal distributions (e.g., kidnapped robot), which is not our scenario.
- **Factor Graph (GTSAM/Ceres):** Gold standard for offline SLAM but too slow for real-time state estimation at 200+ Hz. Could be added as an offline smoother for post-flight analysis.

## References

1. Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter." arXiv:1711.02508.
2. Trawny, N. & Roumeliotis, S. (2005). "Indirect Kalman Filter for 3D Attitude Estimation."
3. PX4 ECL/EKF2 documentation: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
