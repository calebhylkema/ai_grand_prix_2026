# Estimation Module — Error-State Kalman Filter

## Purpose

Fuse high-rate IMU measurements (200–500 Hz) with asynchronous gate-based pose corrections (20–60 Hz) to produce a continuous, low-latency state estimate for the planner and controller.

## State Vector (16D)

```
x = [position(3), velocity(3), orientation_quat(4), accel_bias(3), gyro_bias(3)]
```

- Position and velocity in world frame (W)
- Orientation as quaternion [x, y, z, w] (body-to-world rotation)
- Biases modeled as random walks

## Architecture

**Error-State Kalman Filter (ESKF)** — propagate nominal state with IMU, estimate error state:

1. **Prediction (IMU rate, 200+ Hz):** Integrate acceleration and angular velocity. Use midpoint integration for quaternion propagation. Handle IMU saturation (>16g) by clamping and flagging.
2. **Correction (vision rate, 20–60 Hz):** PnP-derived gate pose as measurement. Measurement model depends on PnP quality:
   - High confidence (6+ corners, 2–5m range): Full 6-DOF update
   - Medium confidence: Position-only update with de-rotated method
   - Low confidence (<2 corners or >10m): Skip update, rely on IMU propagation

## Critical Implementation Details

- Buffer IMU measurements with timestamps for latency compensation
- When vision measurement arrives at time t_meas, propagate state from t_meas to t_now using buffered IMU
- At 100 km/h, 10ms delay = 0.28m error — latency compensation is NOT optional
- Process noise Q tuned per-axis: lateral > vertical (wind affects horizontal more)
- Measurement noise R scaled by inverse confidence from perception
- Chi-squared gating on innovation to reject outlier measurements
- Log innovation sequences for offline filter tuning

## Coordinate Frame Conversions

```python
# Body-to-world rotation from quaternion
R_WB = Rotation.from_quat(state.orientation)  # scipy uses [x,y,z,w]

# Camera-to-body is a fixed extrinsic (from calibration)
R_BC = config.camera_extrinsic  # Typically a 90° rotation

# PnP gives camera-to-gate, we need world-to-body:
# T_WB = T_WG @ T_GC.inv() @ T_CB
```

## Failure Modes to Handle

- IMU saturation (>16g): Use model-predicted acceleration as fallback
- No gates visible for >500ms: Extrapolate with IMU only, increase uncertainty
- Multiple gates visible: Use ALL visible gates for update (AlphaPilot key insight)
- First gate acquisition after takeoff: Initialize position from telemetry start position
