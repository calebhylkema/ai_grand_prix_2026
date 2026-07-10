"""Coordinate-frame transforms and quaternion helpers.

Convention
----------
- Quaternions are scalar-last: [x, y, z, w]  (matches ``scipy.spatial.transform``)
- World frame (W): right-handed, Z-up
- Body frame (B): FRD (Front-Right-Down)
- Camera frame (C): OpenCV (X-right, Y-down, Z-forward)

All angles in radians, distances in metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from drone_racing.utils.types import Mat3, Mat4, Vec3, Vec4

# Gravity in world frame (Z-up, g ≈ 9.81 m/s² downward)
GRAVITY_W: Vec3 = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# ---------------------------------------------------------------------------
# NED ↔ ENU (Z-up) conversions
# ---------------------------------------------------------------------------
# MAVLink uses NED (North-East-Down).  Our internal world frame is ENU-like
# (X-East, Y-North, Z-Up) — or more precisely, a local Z-up Cartesian frame.
# The sim spec says "local Cartesian" without specifying NED vs ENU, but
# SET_POSITION_TARGET_LOCAL_NED explicitly uses NED, so we need conversions
# at the MAVSDK boundary.
#
# NED → ENU:  x_enu =  y_ned,  y_enu =  x_ned,  z_enu = -z_ned
# ENU → NED:  x_ned =  y_enu,  y_ned =  x_enu,  z_ned = -z_enu
# ---------------------------------------------------------------------------


def ned_to_enu(v_ned: Vec3) -> Vec3:
    """Convert a 3-vector from NED to ENU (Z-up) frame.

    Parameters
    ----------
    v_ned : Vec3
        Vector in NED frame [north, east, down].

    Returns
    -------
    Vec3
        Vector in ENU frame [east, north, up].
    """
    return np.array([v_ned[1], v_ned[0], -v_ned[2]], dtype=np.float64)


def enu_to_ned(v_enu: Vec3) -> Vec3:
    """Convert a 3-vector from ENU (Z-up) to NED frame.

    Parameters
    ----------
    v_enu : Vec3
        Vector in ENU frame [east, north, up].

    Returns
    -------
    Vec3
        Vector in NED frame [north, east, down].
    """
    return np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=np.float64)


def quat_ned_to_enu(q_ned: Vec4) -> Vec4:
    """Convert a quaternion from NED body convention to ENU body convention.

    NED body: FRD (Front-Right-Down).
    ENU body: FLU (Front-Left-Up) — but we use FRD internally, so this
    accounts for the world-frame rotation only.

    The rotation between NED and ENU world frames is 90° about Z then
    180° about the new X, but for our pipeline the simplest correct
    approach is: rotate the attitude by the NED→ENU frame transform.

    Parameters
    ----------
    q_ned : Vec4
        Quaternion [x, y, z, w] in NED convention.

    Returns
    -------
    Vec4
        Quaternion [x, y, z, w] in ENU convention.
    """
    # NED→ENU frame rotation: swap x↔y, negate z
    # R_enu_ned = [[0,1,0],[1,0,0],[0,0,-1]]
    r_enu_ned = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    q_frame = Rotation.from_matrix(r_enu_ned).as_quat()
    r_body = Rotation.from_quat(q_ned)
    r_enu = Rotation.from_quat(q_frame) * r_body
    return r_enu.as_quat().astype(np.float64)


def quat_enu_to_ned(q_enu: Vec4) -> Vec4:
    """Convert a quaternion from ENU body convention to NED body convention.

    Parameters
    ----------
    q_enu : Vec4
        Quaternion [x, y, z, w] in ENU convention.

    Returns
    -------
    Vec4
        Quaternion [x, y, z, w] in NED convention.
    """
    # ENU→NED is the same rotation matrix (it's its own inverse)
    r_ned_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    q_frame = Rotation.from_matrix(r_ned_enu).as_quat()
    r_body = Rotation.from_quat(q_enu)
    r_ned = Rotation.from_quat(q_frame) * r_body
    return r_ned.as_quat().astype(np.float64)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def quat_identity() -> Vec4:
    """Return the identity quaternion [0, 0, 0, 1]."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def quat_normalize(q: Vec4) -> Vec4:
    """Normalise a quaternion to unit length.

    Parameters
    ----------
    q : Vec4
        Quaternion [x, y, z, w].

    Returns
    -------
    Vec4
        Unit quaternion.
    """
    n = np.linalg.norm(q)
    if n < 1e-12:
        return quat_identity()
    return q / n


def quat_multiply(q1: Vec4, q2: Vec4) -> Vec4:
    """Hamilton product q1 ⊗ q2 (scalar-last convention).

    Parameters
    ----------
    q1, q2 : Vec4
        Quaternions [x, y, z, w].

    Returns
    -------
    Vec4
        Product quaternion (not normalised).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: Vec4) -> Vec4:
    """Return the conjugate (inverse for unit quaternions).

    Parameters
    ----------
    q : Vec4
        Quaternion [x, y, z, w].

    Returns
    -------
    Vec4
        Conjugate [-x, -y, -z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_rotate_vector(q: Vec4, v: Vec3) -> Vec3:
    """Rotate vector *v* by unit quaternion *q*.

    Equivalent to ``R(q) @ v`` but avoids building the full matrix.

    Parameters
    ----------
    q : Vec4
        Unit quaternion [x, y, z, w].
    v : Vec3
        3-vector.

    Returns
    -------
    Vec3
        Rotated vector.
    """
    return Rotation.from_quat(q).apply(v)  # type: ignore[no-any-return]


def quat_to_rotation_matrix(q: Vec4) -> Mat3:
    """Convert quaternion to 3×3 rotation matrix.

    Parameters
    ----------
    q : Vec4
        Quaternion [x, y, z, w].

    Returns
    -------
    Mat3
        3×3 rotation matrix.
    """
    return Rotation.from_quat(q).as_matrix()  # type: ignore[no-any-return]


def rotation_matrix_to_quat(r: Mat3) -> Vec4:
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w].

    Parameters
    ----------
    r : Mat3
        3×3 rotation matrix.

    Returns
    -------
    Vec4
        Quaternion [x, y, z, w].
    """
    return Rotation.from_matrix(r).as_quat()  # type: ignore[no-any-return]


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Vec4:
    """Euler angles (ZYX intrinsic) to quaternion.

    Parameters
    ----------
    roll, pitch, yaw : float
        Angles in radians.

    Returns
    -------
    Vec4
        Quaternion [x, y, z, w].
    """
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()  # type: ignore[no-any-return]


def quat_to_euler(q: Vec4) -> tuple[float, float, float]:
    """Quaternion to Euler angles (roll, pitch, yaw).

    Parameters
    ----------
    q : Vec4
        Quaternion [x, y, z, w].

    Returns
    -------
    tuple[float, float, float]
        (roll, pitch, yaw) in radians.
    """
    yaw, pitch, roll = Rotation.from_quat(q).as_euler("ZYX")
    return float(roll), float(pitch), float(yaw)


# ---------------------------------------------------------------------------
# Error-state (small-angle) parameterisation  — used by ESKF
# ---------------------------------------------------------------------------


def small_angle_to_quat(delta_theta: Vec3) -> Vec4:
    """Convert a small-angle rotation vector to a quaternion.

    For the ESKF error-state, the 3-DOF orientation error δθ is mapped to
    a quaternion via the first-order approximation:

        δq ≈ [δθ/2, 1]  (then normalised)

    Parameters
    ----------
    delta_theta : Vec3
        Small rotation vector (rad).

    Returns
    -------
    Vec4
        Corresponding quaternion [x, y, z, w].
    """
    return quat_normalize(np.array([*(delta_theta / 2.0), 1.0], dtype=np.float64))


def skew_symmetric(v: Vec3) -> Mat3:
    """Build the 3×3 skew-symmetric (cross-product) matrix of *v*.

    Parameters
    ----------
    v : Vec3
        3-vector.

    Returns
    -------
    Mat3
        [v]× such that [v]× @ u == np.cross(v, u).
    """
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Homogeneous transforms
# ---------------------------------------------------------------------------


def pose_to_matrix(position: Vec3, orientation: Vec4) -> Mat4:
    """Build a 4×4 homogeneous transform from position and quaternion.

    Parameters
    ----------
    position : Vec3
        Translation [x, y, z].
    orientation : Vec4
        Rotation quaternion [x, y, z, w].

    Returns
    -------
    Mat4
        4×4 SE(3) matrix.
    """
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_to_rotation_matrix(orientation)
    t[:3, 3] = position
    return t


def matrix_to_pose(t: Mat4) -> tuple[Vec3, Vec4]:
    """Extract position and quaternion from a 4×4 homogeneous transform.

    Parameters
    ----------
    t : Mat4
        4×4 SE(3) matrix.

    Returns
    -------
    tuple[Vec3, Vec4]
        (position, quaternion [x, y, z, w]).
    """
    position: Vec3 = t[:3, 3].copy()
    orientation = rotation_matrix_to_quat(t[:3, :3])
    return position, orientation


def invert_transform(t: Mat4) -> Mat4:
    """Invert a 4×4 homogeneous transform efficiently.

    Parameters
    ----------
    t : Mat4
        4×4 SE(3) matrix.

    Returns
    -------
    Mat4
        Inverse transform T⁻¹.
    """
    r = t[:3, :3]
    p = t[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


# ---------------------------------------------------------------------------
# Camera ↔ body transform
# ---------------------------------------------------------------------------


def camera_to_body_rotation(tilt_deg: float) -> Mat3:
    """Compute the rotation from OpenCV camera frame to FRD body frame.

    The FPV camera is tilted forward by *tilt_deg* degrees.  The mapping is:

    1. Camera Z-forward → Body X-forward
    2. Camera X-right   → Body Y-right
    3. Camera Y-down    → Body Z-down

    Then apply the forward tilt about the body Y axis.

    Parameters
    ----------
    tilt_deg : float
        Forward tilt of the camera in degrees.

    Returns
    -------
    Mat3
        3×3 rotation matrix R_BC (camera-to-body).
    """
    # Base rotation: camera convention → FRD body
    # cam X → body Y, cam Y → body Z, cam Z → body X
    r_base = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    # Camera tilt: rotation about body-Y axis (pitch up = positive tilt)
    tilt_rad = np.deg2rad(tilt_deg)
    r_tilt = Rotation.from_euler("y", -tilt_rad).as_matrix()
    return r_tilt @ r_base  # type: ignore[no-any-return]


def body_to_camera_rotation(tilt_deg: float) -> Mat3:
    """Inverse of :func:`camera_to_body_rotation`.

    Parameters
    ----------
    tilt_deg : float
        Forward tilt of the camera in degrees.

    Returns
    -------
    Mat3
        3×3 rotation matrix R_CB (body-to-camera).
    """
    return camera_to_body_rotation(tilt_deg).T
