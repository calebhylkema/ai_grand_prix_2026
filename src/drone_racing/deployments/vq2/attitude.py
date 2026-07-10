"""State estimation for VQ2 (IMU + monocular PnP only — no ground-truth telemetry).

We have NO attitude/position/velocity from the sim, so we estimate them and, since
there is no ground truth to check against, we lean on **self-consistency** signals
the drone still gives us:

- Attitude (roll/pitch): the accelerometer points along gravity when |accel|≈g, so
  the angle between the estimated up-vector and the measured accel direction (during
  steady windows) is a real, ground-truth-free error metric. Small + non-drifting
  through maneuvers ⇒ good.
- Velocity: the gate is a STATIC world landmark, so its PnP relative motion equals the
  negative of the drone's motion. IMU-integrated velocity vs PnP-derived velocity is a
  consistency check; commanded hover ⇒ velocity should stay ~0 (drift metric).

Conventions: body FRD, world NED (gravity = +Z down). Quaternion is scalar-first
[w,x,y,z], rotation body→world (R @ v_body = v_world).
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body→world) for scalar-first quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) radians from body→world quaternion (ZYX)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    pitch = np.arcsin(max(-1.0, min(1.0, sp)))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


class AttitudeEstimator:
    """Mahony-style complementary filter on SO(3).

    Gyro propagates the quaternion (continuous through maneuvers); the accelerometer
    nudges roll/pitch toward gravity, weighted by how close |accel| is to g so that
    thrust/motion accelerations (|accel|≠g) don't corrupt it.
    """

    def __init__(self, kp: float = 1.5) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.kp = kp
        self._initialised = False
        self._vis_mute = 0  # updates left with accel muted (vision roll is streaming)
        self.in_flight = False  # set True at takeoff: accel correction OFF (see update)
        # rolling validation stats
        self._grav_err_sum = 0.0
        self._grav_err_n = 0

    def initialise(self, accel: np.ndarray) -> None:
        """Set the initial quaternion from the at-rest accel (roll/pitch), yaw=0."""
        ax, ay, az = accel
        roll = np.arctan2(-ay, -az)
        pitch = np.arctan2(ax, np.hypot(ay, az))
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        # yaw=0: q = qz(0)*qy(pitch)*qx(roll)
        self.q = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
        self.q /= np.linalg.norm(self.q)
        self._initialised = True

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        if not self._initialised:
            self.initialise(accel)
            return
        amag = float(np.linalg.norm(accel))
        omega = np.array(gyro, dtype=float)
        # Accel correction ONLY on the ground (in_flight=False). Every gate we tried
        # (|a|≈g band, gyro-quiet, vis-mute) still let it lie in some flight regime,
        # because on a quad the accel measures body-z thrust — direction "level" — at
        # ANY attitude: it erased a real 41° bank (run-3 crash, 2026-07-03) and in
        # steady cruise (gyro quiet, |a|=g·sec(18°)≈1.05g, inside every band) it built
        # a +12° nose-up pitch bias (run-E, 2026-07-04: filter p+8 while the horizon
        # row in the trace frames showed real p≈−4 → vz integration read "climb" while
        # the drone skimmed the floor into an obstacle). In flight: gyro integration
        # (drift-free over a run) + the vision-absolute roll from gate PnP.
        if amag > 1e-3 and not self.in_flight:
            a_hat = np.asarray(accel, float) / amag
            r = quat_to_matrix(self.q)
            up_body = r.T @ np.array([0.0, 0.0, -1.0])  # predicted gravity-up in body
            err = np.cross(a_hat, up_body)  # rotation that aligns prediction to measurement
            w_acc = max(0.0, 1.0 - abs(amag - G) / 2.5)  # 1 at |a|=g, 0 by 2.5 m/s^2 off
            w_acc *= max(0.0, 1.0 - float(np.linalg.norm(omega)) / 0.6)  # 0 by 0.6 rad/s
            if self._vis_mute > 0:  # an ABSOLUTE roll reference is streaming: don't
                self._vis_mute -= 1  # let the bank-blind accel fight it to a biased
                w_acc *= 0.2  # equilibrium (test: 40° vision vs accel stalled at 34.6°)
            omega = omega + self.kp * w_acc * err
            # validation: instantaneous gravity-alignment error (deg), steady windows only
            if w_acc > 0.5:
                ang = np.degrees(
                    np.arcsin(  # type: ignore[call-overload]  # numpy-stub friction w/ min()
                        min(1.0, np.linalg.norm(np.cross(a_hat, up_body)))
                    )
                )
                self._grav_err_sum += ang
                self._grav_err_n += 1
        # integrate quaternion: q_dot = 0.5 * q ⊗ [0, omega]
        qdot = 0.5 * quat_mul(self.q, np.array([0.0, *omega]))
        self.q = self.q + qdot * dt
        self.q /= np.linalg.norm(self.q)

    def correct_roll(self, roll_meas: float, gain: float = 0.5, max_step: float = 0.02) -> None:
        """Absolute roll correction from vision (world-upright gate PnP), slew-limited.

        2026-07-03: the accel correction is structurally blind to bank in flight — a
        quad's accelerometer reads body-z thrust ("level" at ANY bank), so it actively
        erases real bank from the gyro integral (run-3 crash: real 41° bank, estimate
        +5°). A visible gate's in-plane rotation is a drift-free roll reference.

        ``max_step`` caps the per-call rotation: run-4 showed that slamming a large
        error into the estimate at once (gain 0.5, ~50 Hz) step-drives the attitude
        P-loop against ~1-frame vision latency → ±38° limit cycle. 0.02 rad/call at
        ≤50 Hz is a ≤57°/s slew the P-loop tracks smoothly; a hidden 40° bank is fully
        walked in within ~0.7 s of continuous sightings.
        """
        roll, _, _ = self.euler
        err = math.atan2(math.sin(roll_meas - roll), math.cos(roll_meas - roll))
        a = float(np.clip(gain * err, -max_step, max_step))  # slew-limited body-x rotation
        dq = np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])
        self.q = quat_mul(self.q, dq)  # body-frame rotation: right-multiply
        self.q /= np.linalg.norm(self.q)
        self._vis_mute = 25  # ~0.25-0.5 s of IMU updates: mute the accel pull-to-level

    @property
    def euler(self) -> tuple[float, float, float]:
        return quat_to_euler(self.q)

    @property
    def R(self) -> np.ndarray:  # noqa: N802
        return quat_to_matrix(self.q)

    @property
    def mean_grav_err_deg(self) -> float:
        """Mean gravity-alignment error over steady windows — the roll/pitch validation."""
        return self._grav_err_sum / self._grav_err_n if self._grav_err_n else float("nan")


class VelocityEstimator:
    """World-frame velocity by integrating gravity-compensated accel, softly anchored
    by a zero-velocity or PnP-derived reference to bound drift.
    """

    def __init__(self) -> None:
        self.v = np.zeros(3)

    def predict(self, accel_body: np.ndarray, R: np.ndarray, dt: float) -> None:  # noqa: N803
        a = np.asarray(accel_body, float)
        # Reject collision/impact spikes (normal flight |a|<~20; impacts are 100+ m/s^2),
        # which would otherwise blow up the velocity integration.
        if abs(float(np.linalg.norm(a)) - G) > 20.0:
            return
        a_world = R @ a + np.array([0.0, 0.0, G])  # specific force + gravity
        self.v = self.v + a_world * dt

    def anchor(self, v_meas_world: np.ndarray, gain: float) -> None:
        self.v = self.v + gain * (np.asarray(v_meas_world, float) - self.v)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))
