"""Tests for drone_racing.utils.transforms — quaternion math and frame helpers."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from drone_racing.utils.transforms import (
    body_to_camera_rotation,
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

    def test_zero_tilt_frame_mapping(self) -> None:
        """At zero tilt: cam Z→body X, cam X→body Y, cam Y→body Z."""
        r_bc = camera_to_body_rotation(0.0)
        cam_x = np.array([1.0, 0.0, 0.0])
        cam_y = np.array([0.0, 1.0, 0.0])
        cam_z = np.array([0.0, 0.0, 1.0])
        # R_BC maps camera vectors to body frame
        assert_allclose(r_bc @ cam_z, [1, 0, 0], atol=1e-14)  # cam Z → body X (forward)
        assert_allclose(r_bc @ cam_x, [0, 1, 0], atol=1e-14)  # cam X → body Y (right)
        assert_allclose(r_bc @ cam_y, [0, 0, 1], atol=1e-14)  # cam Y → body Z (down)

    def test_body_to_camera_is_inverse(self) -> None:
        """R_CB @ R_BC == I for any tilt."""
        for tilt in [0.0, 10.0, 25.0, -5.0]:
            r_bc = camera_to_body_rotation(tilt)
            r_cb = body_to_camera_rotation(tilt)
            assert_allclose(r_cb @ r_bc, np.eye(3), atol=1e-14)

    def test_tilt_rotates_forward_down(self) -> None:
        """Positive tilt: camera looks down, so cam Z maps below body X."""
        r_bc = camera_to_body_rotation(20.0)
        cam_forward = r_bc @ np.array([0.0, 0.0, 1.0])
        # Body X component should be largest (still mostly forward)
        assert cam_forward[0] > 0.9
        # Body Z component should be positive (tilted downward in FRD)
        assert cam_forward[2] > 0.0


class TestTransformChain:
    """Full PnP transform chain: T_WB = T_WG @ T_GC^{-1} @ T_CB.

    This tests the exact chain used by the PnP solver to compute the
    drone's world pose from a camera-to-gate measurement.
    """

    def test_known_poses_recover_body(self) -> None:
        """Given known drone and gate poses, verify the chain recovers them."""
        # Known drone pose: at [5, 3, 2], facing +X (yaw=0)
        drone_pos = np.array([5.0, 3.0, 2.0])
        drone_ori = Rotation.from_euler("z", 0.0).as_quat()  # identity yaw
        t_wb = pose_to_matrix(drone_pos, drone_ori)

        # Known gate pose: at [10, 3, 2], facing -X (drone approaches from +X side)
        gate_pos = np.array([10.0, 3.0, 2.0])
        gate_ori = Rotation.from_euler("z", np.pi).as_quat()
        t_wg = pose_to_matrix(gate_pos, gate_ori)

        # Camera extrinsic (zero tilt for simplicity)
        r_bc = camera_to_body_rotation(0.0)
        t_cb = np.eye(4, dtype=np.float64)
        t_cb[:3, :3] = r_bc.T  # R_CB = R_BC^T

        # Simulate what PnP would measure: T_CG (camera-from-gate)
        # T_CW = T_CB @ T_BW = T_CB @ inv(T_WB)
        t_cw = t_cb @ invert_transform(t_wb)
        # T_CG = T_CW @ T_WG
        t_cg = t_cw @ t_wg

        # Now run the PnP chain: T_WB_recovered = T_WG @ inv(T_CG) @ T_CB
        t_gc = invert_transform(t_cg)
        t_wc = t_wg @ t_gc
        t_wb_recovered = t_wc @ t_cb

        pos_recovered, ori_recovered = matrix_to_pose(t_wb_recovered)
        assert_allclose(pos_recovered, drone_pos, atol=1e-10)
        if np.dot(ori_recovered, drone_ori) < 0:
            ori_recovered = -ori_recovered
        assert_allclose(ori_recovered, drone_ori, atol=1e-10)

    def test_chain_with_tilted_camera(self) -> None:
        """Transform chain works with non-zero camera tilt."""
        drone_pos = np.array([0.0, 0.0, 3.0])
        drone_ori = Rotation.from_euler("z", np.pi / 4).as_quat()  # 45° yaw
        t_wb = pose_to_matrix(drone_pos, drone_ori)

        gate_pos = np.array([8.0, 8.0, 3.0])
        gate_ori = Rotation.from_euler("z", -3 * np.pi / 4).as_quat()
        t_wg = pose_to_matrix(gate_pos, gate_ori)

        tilt_deg = 15.0
        r_bc = camera_to_body_rotation(tilt_deg)
        t_cb = np.eye(4, dtype=np.float64)
        t_cb[:3, :3] = r_bc.T

        # Forward: compute T_CG
        t_cw = t_cb @ invert_transform(t_wb)
        t_cg = t_cw @ t_wg

        # Reverse: recover T_WB
        t_wb_recovered = t_wg @ invert_transform(t_cg) @ t_cb

        pos_recovered, ori_recovered = matrix_to_pose(t_wb_recovered)
        assert_allclose(pos_recovered, drone_pos, atol=1e-10)
        if np.dot(ori_recovered, drone_ori) < 0:
            ori_recovered = -ori_recovered
        assert_allclose(ori_recovered, drone_ori, atol=1e-10)

    def test_chain_different_altitudes(self) -> None:
        """Chain works when drone and gate are at different heights."""
        drone_pos = np.array([2.0, 1.0, 1.5])
        drone_ori = Rotation.from_euler("xyz", [0.1, -0.05, 0.3]).as_quat()
        t_wb = pose_to_matrix(drone_pos, drone_ori)

        gate_pos = np.array([10.0, 5.0, 3.0])
        gate_ori = Rotation.from_euler("xyz", [0.0, 0.0, -0.7]).as_quat()
        t_wg = pose_to_matrix(gate_pos, gate_ori)

        r_bc = camera_to_body_rotation(10.0)
        t_cb = np.eye(4, dtype=np.float64)
        t_cb[:3, :3] = r_bc.T

        t_cg = t_cb @ invert_transform(t_wb) @ t_wg
        t_wb_recovered = t_wg @ invert_transform(t_cg) @ t_cb

        pos_recovered, ori_recovered = matrix_to_pose(t_wb_recovered)
        assert_allclose(pos_recovered, drone_pos, atol=1e-10)
        if np.dot(ori_recovered, drone_ori) < 0:
            ori_recovered = -ori_recovered
        assert_allclose(ori_recovered, drone_ori, atol=1e-10)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

# Strategy: non-zero float vectors suitable for quaternion components
_quat_component = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_nonzero_quat = arrays(np.float64, 4, elements=_quat_component).filter(
    lambda q: np.linalg.norm(q) > 0.01
)


class TestQuaternionProperties:
    """Property-based tests for quaternion operations."""

    @given(q=_nonzero_quat)
    @settings(max_examples=100)
    def test_normalize_is_unit(self, q: np.ndarray) -> None:
        """Normalized quaternion always has unit norm."""
        qn = quat_normalize(q)
        assert abs(np.linalg.norm(qn) - 1.0) < 1e-10

    @given(q=_nonzero_quat)
    @settings(max_examples=100)
    def test_normalize_idempotent(self, q: np.ndarray) -> None:
        """Normalizing twice gives the same result."""
        qn1 = quat_normalize(q)
        qn2 = quat_normalize(qn1)
        assert_allclose(qn1, qn2, atol=1e-14)

    @given(q1=_nonzero_quat, q2=_nonzero_quat, q3=_nonzero_quat)
    @settings(max_examples=50)
    def test_multiply_associative(self, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> None:
        """Quaternion multiply is associative: (q1*q2)*q3 == q1*(q2*q3)."""
        lhs = quat_multiply(quat_multiply(q1, q2), q3)
        rhs = quat_multiply(q1, quat_multiply(q2, q3))
        assert_allclose(lhs, rhs, atol=1e-8)

    @given(q=_nonzero_quat)
    @settings(max_examples=100)
    def test_conjugate_is_inverse_for_unit(self, q: np.ndarray) -> None:
        """For unit quaternions, conjugate is the inverse."""
        qu = quat_normalize(q)
        product = quat_multiply(qu, quat_conjugate(qu))
        assert_allclose(product, quat_identity(), atol=1e-10)

    @given(q=_nonzero_quat)
    @settings(max_examples=100)
    def test_rotation_matrix_is_orthogonal(self, q: np.ndarray) -> None:
        """Rotation matrix from any unit quaternion is orthogonal with det=+1."""
        qu = quat_normalize(q)
        r = quat_to_rotation_matrix(qu)
        assert_allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(r), 1.0, atol=1e-10)

    @given(
        q=_nonzero_quat,
        v=arrays(
            np.float64, 3, elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
        ),
    )
    @settings(max_examples=100)
    def test_rotation_preserves_norm(self, q: np.ndarray, v: np.ndarray) -> None:
        """Rotating a vector preserves its magnitude."""
        qu = quat_normalize(q)
        rotated = quat_rotate_vector(qu, v)
        assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v), atol=1e-8)
