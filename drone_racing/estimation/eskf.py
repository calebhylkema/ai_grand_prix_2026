"""Error-State Kalman Filter (ESKF) for drone state estimation.

Fuses high-rate IMU data (200–500 Hz) with asynchronous gate-based PnP pose
corrections (20–60 Hz) to produce a continuous 16-D state estimate.

State vector (nominal):
    [position(3), velocity(3), orientation_quat(4), accel_bias(3), gyro_bias(3)]

Error state (15-D, orientation as 3-DOF small angle):
    [δp(3), δv(3), δθ(3), δba(3), δbg(3)]

References
----------
- Joan Solà, "Quaternion kinematics for the error-state Kalman filter" (2017)
- Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation"
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from drone_racing.utils.transforms import (
    GRAVITY_W,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    skew_symmetric,
    small_angle_to_quat,
)
from drone_racing.utils.types import (
    DetectionConfidence,
    ImuSample,
    PoseMeasurement,
    State,
    Telemetry,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from drone_racing.utils.config import RacingConfig
    from drone_racing.utils.types import CovMat, Vec3, Vec4

logger = logging.getLogger(__name__)

# Error-state indices (15-D)
_P = slice(0, 3)  # position error
_V = slice(3, 6)  # velocity error
_T = slice(6, 9)  # orientation error (small-angle)
_BA = slice(9, 12)  # accel bias error
_BG = slice(12, 15)  # gyro bias error

_STATE_DIM = 15


@dataclass(slots=True)
class _StateSnapshot:
    """Frozen copy of the nominal state for rollback during latency compensation."""

    timestamp_ns: int
    position: Vec3
    velocity: Vec3
    orientation: Vec4
    accel_bias: Vec3
    gyro_bias: Vec3
    covariance: CovMat


@dataclass(slots=True)
class _ImuBufferEntry:
    """Timestamped IMU sample kept for latency compensation."""

    timestamp_ns: int
    accel_mps2: Vec3
    gyro_rps: Vec3
    state_before: _StateSnapshot
    """State snapshot *before* this IMU sample was integrated."""


class ESKF:
    """Error-State Kalman Filter.

    Parameters
    ----------
    config : RacingConfig
        Full racing configuration (estimation + camera params).
    imu_buffer_duration_s : float
        How many seconds of IMU history to keep for latency compensation.
    """

    def __init__(
        self,
        config: RacingConfig,
        imu_buffer_duration_s: float = 0.5,
    ) -> None:
        self._cfg = config.estimation
        self._camera_cfg = config.camera

        # Nominal state
        self._state = State.identity()

        # Process noise continuous-time spectral density
        self._q_c = self._build_continuous_noise()

        # IMU ring-buffer for latency compensation
        max_entries = int(self._cfg.imu_rate_hz * imu_buffer_duration_s) + 1
        self._imu_buffer: deque[_ImuBufferEntry] = deque(maxlen=max_entries)

        self._initialised = False
        self._consecutive_rejections = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Current nominal state (read-only snapshot)."""
        return self._state

    @property
    def initialised(self) -> bool:
        """Whether the filter has been initialised with a starting pose."""
        return self._initialised

    def initialise(self, telemetry: Telemetry) -> None:
        """Set the filter state from the initial telemetry snapshot.

        Parameters
        ----------
        telemetry : Telemetry
            Starting position, velocity, and orientation from race system.
        """
        self._state = State(
            timestamp_ns=telemetry.timestamp_ns,
            position=telemetry.position.copy(),
            velocity=telemetry.velocity.copy(),
            orientation=quat_normalize(telemetry.orientation.copy()),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=self._initial_covariance(),
        )
        self._imu_buffer.clear()
        self._initialised = True
        logger.info("ESKF initialised at t=%d ns", telemetry.timestamp_ns)

    def predict(self, imu: ImuSample) -> State:
        """IMU prediction step — propagate nominal state and covariance.

        Should be called at IMU rate (200+ Hz).

        Parameters
        ----------
        imu : ImuSample
            Raw IMU measurement.

        Returns
        -------
        State
            Updated nominal state after propagation.
        """
        if not self._initialised:
            logger.warning("predict() called before initialise(); ignoring")
            return self._state

        dt = self._dt_from_last(imu.timestamp_ns)
        if dt <= 0.0:
            return self._state

        # Snapshot state *before* this IMU sample is integrated
        snapshot = _StateSnapshot(
            timestamp_ns=self._state.timestamp_ns,
            position=self._state.position.copy(),
            velocity=self._state.velocity.copy(),
            orientation=self._state.orientation.copy(),
            accel_bias=self._state.accel_bias.copy(),
            gyro_bias=self._state.gyro_bias.copy(),
            covariance=self._state.covariance.copy(),
        )

        # Store for latency compensation
        self._imu_buffer.append(
            _ImuBufferEntry(
                timestamp_ns=imu.timestamp_ns,
                accel_mps2=imu.accel_mps2.copy(),
                gyro_rps=imu.gyro_rps.copy(),
                state_before=snapshot,
            )
        )

        # Bias-corrected measurements
        accel = imu.accel_mps2 - self._state.accel_bias
        gyro = imu.gyro_rps - self._state.gyro_bias

        # Clamp if saturated
        if imu.accel_saturated:
            accel_norm = np.linalg.norm(accel)
            if accel_norm > 156.96:  # ~16 g
                accel = accel * (156.96 / accel_norm)
                logger.debug("IMU accel saturated, clamped to 16 g")

        # --- Nominal state propagation ---
        r_wb = quat_to_rotation_matrix(self._state.orientation)

        # Acceleration in world frame (remove gravity)
        accel_w = r_wb @ accel + GRAVITY_W

        # Position & velocity (simple Euler — sufficient at 200+ Hz)
        self._state.position = (
            self._state.position + self._state.velocity * dt + 0.5 * accel_w * dt**2
        )
        self._state.velocity = self._state.velocity + accel_w * dt

        # Orientation (midpoint quaternion integration)
        delta_angle = gyro * dt
        dq = small_angle_to_quat(delta_angle)
        self._state.orientation = quat_normalize(quat_multiply(self._state.orientation, dq))

        # Timestamp
        self._state.timestamp_ns = imu.timestamp_ns

        # --- Error-state covariance propagation ---
        f_x = self._transition_matrix(r_wb, accel, dt)
        q_d = self._discrete_process_noise(r_wb, dt)
        self._state.covariance = f_x @ self._state.covariance @ f_x.T + q_d

        # Covariance capping — prevent runaway growth during IMU-only propagation
        self._cap_covariance()

        return self._state

    def correct(self, measurement: PoseMeasurement) -> State:
        """Vision correction step — fuse PnP pose measurement.

        Handles latency compensation by rolling state forward from the
        measurement timestamp to the current time using buffered IMU data.

        Parameters
        ----------
        measurement : PoseMeasurement
            6-DOF pose from PnP solver.

        Returns
        -------
        State
            Updated nominal state after correction.
        """
        if not self._initialised:
            logger.warning("correct() called before initialise(); ignoring")
            return self._state

        # Skip low-confidence detections
        if measurement.confidence == DetectionConfidence.LOW:
            logger.debug("Skipping LOW confidence measurement (gate=%s)", measurement.gate_id)
            return self._state

        # --- Latency compensation: roll back to measurement time ---
        imu_to_replay: list[_ImuBufferEntry] = []
        rolled_back = False

        if measurement.timestamp_ns < self._state.timestamp_ns and len(self._imu_buffer) > 0:
            # Find the closest buffered state at or before the measurement time
            rollback_entry: _ImuBufferEntry | None = None
            replay_start_idx = 0

            for i, entry in enumerate(self._imu_buffer):
                if entry.state_before.timestamp_ns <= measurement.timestamp_ns:
                    rollback_entry = entry
                    replay_start_idx = i
                else:
                    break

            if rollback_entry is not None:
                # Restore state to the snapshot before this IMU sample
                snap = rollback_entry.state_before
                self._state.timestamp_ns = snap.timestamp_ns
                self._state.position = snap.position.copy()
                self._state.velocity = snap.velocity.copy()
                self._state.orientation = snap.orientation.copy()
                self._state.accel_bias = snap.accel_bias.copy()
                self._state.gyro_bias = snap.gyro_bias.copy()
                self._state.covariance = snap.covariance.copy()

                # Collect IMU samples to replay after correction
                imu_to_replay = list(self._imu_buffer)[replay_start_idx:]
                rolled_back = True

                logger.debug(
                    "Latency comp: rolled back %d ms, will replay %d IMU samples",
                    (self._state.timestamp_ns - snap.timestamp_ns) // 1_000_000,
                    len(imu_to_replay),
                )

        # --- Build measurement and Kalman gain ---
        if measurement.confidence == DetectionConfidence.HIGH:
            # Full 6-DOF update
            innovation, h, r = self._full_6dof_measurement(measurement)
        else:
            # Position-only update (MEDIUM confidence)
            innovation, h, r = self._position_only_measurement(measurement)

        # Chi-squared gating
        s = h @ self._state.covariance @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance — skipping update")
            if rolled_back:
                self._repropagate_from(imu_to_replay)
            return self._state

        chi2 = float(innovation @ s_inv @ innovation)
        if chi2 > self._cfg.chi2_gate_threshold:
            self._consecutive_rejections += 1
            logger.warning(
                "Chi² = %.2f > threshold %.2f — rejecting measurement (gate=%s)",
                chi2,
                self._cfg.chi2_gate_threshold,
                measurement.gate_id,
            )
            # Covariance inflation after many consecutive rejections.
            # This widens the chi² acceptance region so the filter can
            # re-acquire lock after transient divergence.
            if self._consecutive_rejections >= self._cfg.consecutive_rejection_limit:
                self._state.covariance *= self._cfg.covariance_inflation_factor
                self._cap_covariance()
                self._consecutive_rejections = 0
                logger.info(
                    "ESKF: inflated covariance (%d consecutive rejections)",
                    self._cfg.consecutive_rejection_limit,
                )
            if rolled_back:
                self._repropagate_from(imu_to_replay)
            return self._state

        self._consecutive_rejections = 0

        # Kalman gain
        k = self._state.covariance @ h.T @ s_inv

        # For position-only updates (MEDIUM confidence), attenuate gains
        # for non-position states.  Orientation drift is the primary cause
        # of state divergence — allow partial correction to prevent it.
        if measurement.confidence != DetectionConfidence.HIGH:
            k[3:6, :] *= self._cfg.low_confidence_vel_gain
            k[6:9, :] *= self._cfg.low_confidence_ori_gain
            k[9:, :] = 0.0  # still zero bias from position-only

        # Error-state correction
        dx = k @ innovation

        # Cap correction magnitude to prevent catastrophic state jumps
        # from bad PnP solutions (especially with 4 coplanar corners).
        pos_corr = float(np.linalg.norm(dx[_P]))
        vel_corr = float(np.linalg.norm(dx[_V]))
        ori_corr = float(np.linalg.norm(dx[_T]))
        max_pos_corr = self._cfg.max_pos_correction
        max_vel_corr = self._cfg.max_vel_correction
        max_ori_corr = self._cfg.max_ori_correction
        if pos_corr > max_pos_corr or vel_corr > max_vel_corr or ori_corr > max_ori_corr:
            scale = min(
                max_pos_corr / max(pos_corr, 1e-9),
                max_vel_corr / max(vel_corr, 1e-9),
                max_ori_corr / max(ori_corr, 1e-9),
            )
            dx = dx * scale
            logger.debug(
                "ESKF: capping correction magnitude (pos=%.2f vel=%.2f ori=%.2f → scale=%.2f)",
                pos_corr,
                vel_corr,
                ori_corr,
                scale,
            )

        # --- Inject error into nominal state ---
        self._state.position = self._state.position + dx[_P]
        self._state.velocity = self._state.velocity + dx[_V]
        dq = small_angle_to_quat(dx[_T])
        self._state.orientation = quat_normalize(quat_multiply(self._state.orientation, dq))
        self._state.accel_bias = self._state.accel_bias + dx[_BA]
        self._state.gyro_bias = self._state.gyro_bias + dx[_BG]

        # --- Covariance update (Joseph form for numerical stability) ---
        i_kh = np.eye(_STATE_DIM) - k @ h
        self._state.covariance = i_kh @ self._state.covariance @ i_kh.T + k @ r @ k.T

        # Force symmetry
        self._state.covariance = 0.5 * (self._state.covariance + self._state.covariance.T)

        logger.debug(
            "ESKF corrected: gate=%s, chi²=%.2f, confidence=%s, rollback=%s",
            measurement.gate_id,
            chi2,
            measurement.confidence.value,
            rolled_back,
        )

        # --- Re-propagate from measurement time to current time ---
        if rolled_back:
            self._repropagate_from(imu_to_replay)

        return self._state

    # ------------------------------------------------------------------
    # Measurement models
    # ------------------------------------------------------------------

    def _full_6dof_measurement(
        self, meas: PoseMeasurement
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], CovMat]:
        """Build innovation, H matrix, and R for full 6-DOF update.

        Parameters
        ----------
        meas : PoseMeasurement
            Measurement with position and orientation.

        Returns
        -------
        tuple
            (innovation [6], H [6×15], R [6×6]).
        """
        # Position innovation
        pos_innov = meas.position_w - self._state.position

        # Orientation innovation: small-angle difference
        q_meas = quat_normalize(meas.orientation_w)
        q_state = self._state.orientation
        # δq = q_meas ⊗ q_state⁻¹ → extract δθ ≈ 2 * δq_vec
        q_err = quat_multiply(q_meas, np.array([-q_state[0], -q_state[1], -q_state[2], q_state[3]]))
        if q_err[3] < 0:
            q_err = -q_err
        ori_innov = 2.0 * q_err[:3]

        innovation = np.concatenate([pos_innov, ori_innov])

        # H matrix: identity blocks for position and orientation errors
        h = np.zeros((6, _STATE_DIM), dtype=np.float64)
        h[0:3, _P] = np.eye(3)
        h[3:6, _T] = np.eye(3)

        # Measurement noise
        r = np.diag([meas.position_noise_std**2] * 3 + [meas.orientation_noise_std**2] * 3)

        return innovation, h, r

    def _position_only_measurement(
        self, meas: PoseMeasurement
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], CovMat]:
        """Build innovation, H matrix, and R for position-only update.

        Parameters
        ----------
        meas : PoseMeasurement
            Measurement (only position used).

        Returns
        -------
        tuple
            (innovation [3], H [3×15], R [3×3]).
        """
        innovation = meas.position_w - self._state.position

        h = np.zeros((3, _STATE_DIM), dtype=np.float64)
        h[0:3, _P] = np.eye(3)

        r = np.eye(3) * meas.position_noise_std**2

        return innovation, h, r

    # ------------------------------------------------------------------
    # Process model internals
    # ------------------------------------------------------------------

    def _transition_matrix(
        self, r_wb: NDArray[np.floating[Any]], accel_body: Vec3, dt: float
    ) -> NDArray[np.floating[Any]]:
        """Discrete error-state transition matrix Fx.

        Parameters
        ----------
        r_wb : Mat3
            Current body-to-world rotation matrix.
        accel_body : Vec3
            Bias-corrected accelerometer reading in body frame.
        dt : float
            Time step (s).

        Returns
        -------
        NDArray
            15×15 transition matrix.
        """
        f = np.eye(_STATE_DIM, dtype=np.float64)

        # δp ← δp + δv·dt
        f[_P, _V] = np.eye(3) * dt

        # δv ← δv - R·[a×]·δθ·dt - R·δba·dt
        f[_V, _T] = -r_wb @ skew_symmetric(accel_body) * dt
        f[_V, _BA] = -r_wb * dt

        # δθ ← δθ - δbg·dt
        f[_T, _BG] = -np.eye(3) * dt

        return f

    def _build_continuous_noise(self) -> NDArray[np.floating[Any]]:
        """Continuous-time process noise spectral density Qc.

        Returns
        -------
        NDArray
            15×15 diagonal continuous noise matrix.
        """
        qc = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)

        # Velocity noise from accelerometer
        qc[_V, _V] = np.eye(3) * self._cfg.accel_noise_std**2
        # Orientation noise from gyroscope
        qc[_T, _T] = np.eye(3) * self._cfg.gyro_noise_std**2
        # Bias random walks
        qc[_BA, _BA] = np.eye(3) * self._cfg.accel_bias_std**2
        qc[_BG, _BG] = np.eye(3) * self._cfg.gyro_bias_std**2

        return qc

    def _discrete_process_noise(
        self, r_wb: NDArray[np.floating[Any]], dt: float
    ) -> NDArray[np.floating[Any]]:
        """Discretised process noise Qd ≈ Fx · Qc · Fx^T · dt.

        For simplicity we use the first-order approximation Qd = Qc · dt,
        which is adequate when dt is small (200+ Hz).

        Parameters
        ----------
        r_wb : Mat3
            Current rotation matrix (used to rotate accel noise to world).
        dt : float
            Time step (s).

        Returns
        -------
        NDArray
            15×15 discrete process noise matrix.
        """
        qd = self._q_c.copy() * dt

        # Rotate accelerometer noise to world frame
        qd[_V, _V] = r_wb @ (np.eye(3) * self._cfg.accel_noise_std**2) @ r_wb.T * dt

        return qd

    def _initial_covariance(self) -> CovMat:
        """Initial error-state covariance P₀.

        Returns
        -------
        CovMat
            15×15 initial covariance matrix.
        """
        p = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)

        # Position: ±0.5 m
        p[_P, _P] = np.eye(3) * 0.5**2
        # Velocity: ±0.1 m/s (starting from known velocity)
        p[_V, _V] = np.eye(3) * 0.1**2
        # Orientation: ±5° ≈ 0.087 rad
        p[_T, _T] = np.eye(3) * 0.087**2
        # Accel bias: ±0.5 m/s²
        p[_BA, _BA] = np.eye(3) * 0.5**2
        # Gyro bias: ±0.05 rad/s
        p[_BG, _BG] = np.eye(3) * 0.05**2

        return p

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Diagonal covariance caps (variance, not std-dev).
    # Prevents runaway growth during long IMU-only propagation.
    _COV_CAP = np.array(
        [
            10.0,
            10.0,
            10.0,  # position: ±3.2m
            25.0,
            25.0,
            25.0,  # velocity: ±5 m/s
            0.25,
            0.25,
            0.25,  # orientation: ±~29°
            0.25,
            0.25,
            0.25,  # accel bias: ±0.5 m/s²
            0.01,
            0.01,
            0.01,  # gyro bias: ±0.1 rad/s
        ]
    )

    def _cap_covariance(self) -> None:
        """Clamp covariance diagonal to prevent unbounded growth.

        Standard practice in real-world EKFs.  Without capping, orientation
        uncertainty couples into velocity via ``-R·[a×]·δθ·dt`` in the
        transition matrix, causing exponential growth during aggressive
        manoeuvres.
        """
        diag = np.diag(self._state.covariance)
        cap = self._COV_CAP
        if np.any(diag > cap):
            scale = np.minimum(1.0, cap / np.maximum(diag, 1e-12))
            # Scale rows and columns to preserve correlations proportionally
            s = np.sqrt(scale)
            self._state.covariance = s[:, np.newaxis] * self._state.covariance * s[np.newaxis, :]

    def _dt_from_last(self, timestamp_ns: int) -> float:
        """Compute dt in seconds from the current state timestamp.

        Parameters
        ----------
        timestamp_ns : int
            New measurement timestamp.

        Returns
        -------
        float
            Time difference in seconds. Returns 0 if non-positive.
        """
        dt_ns = timestamp_ns - self._state.timestamp_ns
        if dt_ns <= 0:
            return 0.0
        return dt_ns * 1e-9

    def _repropagate_from(self, imu_entries: list[_ImuBufferEntry]) -> None:
        """Re-propagate state using buffered IMU samples after a correction.

        After rolling back state and applying a vision correction at a past
        timestamp, this method replays the buffered IMU samples to bring the
        state back to the current time.

        Parameters
        ----------
        imu_entries : list[_ImuBufferEntry]
            IMU samples to replay, in chronological order.
        """
        for entry in imu_entries:
            dt = self._dt_from_last(entry.timestamp_ns)
            if dt <= 0.0:
                continue

            # Bias-corrected measurements
            accel = entry.accel_mps2 - self._state.accel_bias
            gyro = entry.gyro_rps - self._state.gyro_bias

            # Nominal state propagation (same as predict())
            r_wb = quat_to_rotation_matrix(self._state.orientation)
            accel_w = r_wb @ accel + GRAVITY_W

            self._state.position = (
                self._state.position + self._state.velocity * dt + 0.5 * accel_w * dt**2
            )
            self._state.velocity = self._state.velocity + accel_w * dt

            delta_angle = gyro * dt
            dq = small_angle_to_quat(delta_angle)
            self._state.orientation = quat_normalize(quat_multiply(self._state.orientation, dq))

            self._state.timestamp_ns = entry.timestamp_ns

            # Covariance propagation
            f_x = self._transition_matrix(r_wb, accel, dt)
            q_d = self._discrete_process_noise(r_wb, dt)
            self._state.covariance = f_x @ self._state.covariance @ f_x.T + q_d

        logger.debug(
            "Latency comp: re-propagated %d IMU samples to t=%d ns",
            len(imu_entries),
            self._state.timestamp_ns,
        )
