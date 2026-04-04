"""Tests for drone_racing.utils.transforms — quaternion math and frame helpers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from drone_racing.utils.transforms import (
    camera_to_body_rotation,
    euler_to_quat,
    invert_transform,
    matrix_to_pose,
    pose_to_matrix,
    quat_conjugate,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    skew_symmetric,
    small_angle_to_quat,
)


class TestQuaternionBasics:
    """Identity, normalisation, conjugate."""

    def test_identity_is_unit(self) -> None:
        q = quat_identity()
        assert_allclose(np.linalg.norm(q), 1.0)
        assert_allclose(q, [0, 0, 0, 1])

    def test_normalize_unit(self) -> None:
        q = np.array([1.0, 2.0, 3.0, 4.0])
        qn = quat_normalize(q)
        assert_allclose(np.linalg.norm(qn), 1.0)

    def test_normalize_zero_returns_identity(self) -> None:
        q = np.zeros(4)
        assert_allclose(quat_normalize(q), quat_identity())

    def test_conjugate(self) -> None:
        q = np.array([0.1, 0.2, 0.3, 0.9])
        qc = quat_conjugate(q)
        assert_allclose(qc, [-0.1, -0.2, -0.3, 0.9])


class TestQuaternionMultiply:
    """Hamilton product properties."""

    def test_identity_is_neutral(self) -> None:
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_multiply(q, quat_identity())
        assert_allclose(result, q, atol=1e-14)

    def test_inverse_gives_identity(self) -> None:
        q = quat_normalize(np.array([0.5, -0.3, 0.7, 0.4]))
        result = quat_multiply(q, quat_conjugate(q))
        assert_allclose(result, quat_identity(), atol=1e-14)

    def test_matches_scipy(self) -> None:
        """Verify our multiply matches scipy's composition."""
        q1 = Rotation.random(random_state=42).as_quat()
        q2 = Rotation.random(random_state=43).as_quat()
        expected = (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()
        result = quat_multiply(q1, q2)
        # Quaternion double cover: q and -q represent the same rotation
        if np.dot(result, expected) < 0:
            result = -result
        assert_allclose(result, expected, atol=1e-12)


class TestQuaternionRotation:
    """Rotate vectors with quaternions."""

    def test_identity_no_rotation(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_vector(quat_identity(), v)
        assert_allclose(result, v, atol=1e-14)

    def test_90deg_z_rotation(self) -> None:
        """Rotate [1,0,0] by 90° about Z → [0,1,0]."""
        q = Rotation.from_euler("z", 90, degrees=True).as_quat()
        result = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        assert_allclose(result, [0, 1, 0], atol=1e-14)


class TestQuaternionConversions:
    """Quaternion ↔ matrix ↔ euler round-trips."""

    def test_quat_to_matrix_identity(self) -> None:
        r = quat_to_rotation_matrix(quat_identity())
        assert_allclose(r, np.eye(3), atol=1e-14)

    def test_matrix_roundtrip(self) -> None:
        q = Rotation.random(random_state=44).as_quat()
        r = quat_to_rotation_matrix(q)
        q_back = rotation_matrix_to_quat(r)
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        assert_allclose(q_back, q, atol=1e-12)

    def test_euler_roundtrip(self) -> None:
        roll, pitch, yaw = 0.1, -0.2, 0.3
        q = euler_to_quat(roll, pitch, yaw)
        r2, p2, y2 = quat_to_euler(q)
        assert_allclose([r2, p2, y2], [roll, pitch, yaw], atol=1e-12)


class TestSmallAngle:
    """Error-state small-angle parameterisation."""

    def test_zero_gives_identity(self) -> None:
        q = small_angle_to_quat(np.zeros(3))
        assert_allclose(q, quat_identity(), atol=1e-14)

    def test_small_rotation_accurate(self) -> None:
        """For small angles, δq ≈ [δθ/2, 1] should match exact rotation."""
        delta = np.array([0.001, -0.002, 0.003])
        q_approx = small_angle_to_quat(delta)
        q_exact = Rotation.from_rotvec(delta).as_quat()
        # At 0.003 rad the approximation error should be < 1e-6
        if np.dot(q_approx, q_exact) < 0:
            q_exact = -q_exact
        assert_allclose(q_approx, q_exact, atol=1e-5)


class TestSkewSymmetric:
    """Skew-symmetric matrix properties."""

    def test_cross_product_equivalence(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        assert_allclose(skew_symmetric(v) @ u, np.cross(v, u), atol=1e-14)

    def test_antisymmetric(self) -> None:
        v = np.array([1.0, -2.0, 0.5])
        s = skew_symmetric(v)
        assert_allclose(s + s.T, np.zeros((3, 3)), atol=1e-14)


class TestHomogeneousTransforms:
    """SE(3) pose conversions and inversion."""

    def test_identity_transform(self) -> None:
        t = pose_to_matrix(np.zeros(3), quat_identity())
        assert_allclose(t, np.eye(4), atol=1e-14)

    def test_roundtrip(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        ori = Rotation.random(random_state=45).as_quat()
        t = pose_to_matrix(pos, ori)
        pos2, ori2 = matrix_to_pose(t)
        assert_allclose(pos2, pos, atol=1e-12)
        if np.dot(ori, ori2) < 0:
            ori2 = -ori2
        assert_allclose(ori2, ori, atol=1e-12)

    def test_invert(self) -> None:
        pos = np.array([1.0, -2.0, 0.5])
        ori = Rotation.random(random_state=46).as_quat()
        t = pose_to_matrix(pos, ori)
        t_inv = invert_transform(t)
        product = t @ t_inv
        assert_allclose(product, np.eye(4), atol=1e-12)


class TestCameraBodyRotation:
    """Camera ↔ body frame rotation."""

    def test_zero_tilt_is_orthogonal(self) -> None:
        r = camera_to_body_rotation(0.0)
        assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert_allclose(np.linalg.det(r), 1.0, atol=1e-14)

    def test_nonzero_tilt_is_orthogonal(self) -> None:
        r = camera_to_body_rotation(15.0)
        assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert_allclose(np.linalg.det(r), 1.0, atol=1e-14)
