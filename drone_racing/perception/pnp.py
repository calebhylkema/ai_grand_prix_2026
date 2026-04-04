"""PnP pose solver — converts 2D gate corners to a 6-DOF pose measurement.

Takes detected gate corners (2D pixels) and the known 3D gate model,
solves PnP to get the camera-to-gate transform, then chains through
the known gate world pose and camera extrinsic to produce a
world-frame drone pose for the ESKF.

Transform chain::

    T_WB = T_WG @ T_GC^{-1} @ T_CB

Where:
    - T_WG: gate pose in world frame (from gate map)
    - T_GC: gate-to-camera transform (from PnP — what OpenCV gives us)
    - T_CB: camera-to-body extrinsic (from calibration / config)
    - T_WB: drone body pose in world frame (what the ESKF needs)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.utils.transforms import (
    camera_to_body_rotation,
    invert_transform,
    matrix_to_pose,
    pose_to_matrix,
)
from drone_racing.utils.types import (
    GateDetection,
    GateWaypoint,
    PoseMeasurement,
)

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig, EstimationConfig

logger = logging.getLogger(__name__)


class PnPSolver:
    """Solve Perspective-n-Point to convert gate detections into pose measurements.

    Parameters
    ----------
    camera_config : CameraConfig
        Camera intrinsics and tilt angle.
    estimation_config : EstimationConfig
        Noise parameters for the output measurements.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        estimation_config: EstimationConfig,
    ) -> None:
        self._camera_k = camera_config.intrinsic_matrix
        self._camera_dist = camera_config.distortion_coeffs

        # Camera-to-body rotation (fixed extrinsic)
        self._r_bc = camera_to_body_rotation(camera_config.tilt_deg)

        # Build the full 4×4 camera-to-body transform (translation assumed zero)
        self._t_cb = np.eye(4, dtype=np.float64)
        self._t_cb[:3, :3] = self._r_bc.T  # R_CB = R_BC^T (body-to-camera)
        # Actually we want T_CB: transforms points from camera to body
        # T_CB has R_BC (camera-to-body) and zero translation
        self._t_cb[:3, :3] = self._r_bc

        self._est_cfg = estimation_config

    def solve(
        self,
        detection: GateDetection,
        gate: GateWaypoint,
        gate_corners_3d_local: np.ndarray,
    ) -> PoseMeasurement | None:
        """Solve PnP for a single gate detection.

        Parameters
        ----------
        detection : GateDetection
            Detected 2D corners from the perception module.
        gate : GateWaypoint
            Known gate pose in world frame.
        gate_corners_3d_local : np.ndarray
            (N, 3) gate corner positions in the gate's local frame.
            Must have the same number of points and same ordering as
            ``detection.corners_2d``.

        Returns
        -------
        PoseMeasurement | None
            World-frame pose measurement for the ESKF, or None if PnP fails.
        """
        corners_2d = detection.corners_2d
        n_corners = len(corners_2d)

        if n_corners < 4:
            logger.debug("PnP: need ≥4 corners, got %d — skipping", n_corners)
            return None

        # Use only the first N corners that match the 3D model
        n_model = len(gate_corners_3d_local)
        n_use = min(n_corners, n_model)
        if n_use < 4:
            logger.debug("PnP: not enough matching corners (%d) — skipping", n_use)
            return None

        obj_pts = gate_corners_3d_local[:n_use].astype(np.float64)
        img_pts = corners_2d[:n_use].astype(np.float64)

        # --- Solve PnP ---
        # solvePnP gives the rotation and translation of the object (gate)
        # in the camera frame: T_CG (camera-from-gate)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self._camera_k,
            self._camera_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.debug("PnP: solvePnP failed for gate %s", gate.gate_id)
            return None

        # Convert rvec to rotation matrix
        r_cg, _ = cv2.Rodrigues(rvec)
        t_cg_vec = tvec.flatten()

        # Build T_CG (camera-from-gate = gate pose in camera frame)
        t_cg = np.eye(4, dtype=np.float64)
        t_cg[:3, :3] = r_cg
        t_cg[:3, 3] = t_cg_vec

        # --- Chain transforms to get drone world pose ---
        # T_WG: gate pose in world frame
        t_wg = pose_to_matrix(gate.position_w, gate.orientation_w)

        # T_GC = T_CG^{-1}  (gate-from-camera)
        t_gc = invert_transform(t_cg)

        # T_WC = T_WG @ T_GC  (world-from-camera = camera pose in world)
        t_wc = t_wg @ t_gc

        # T_WB = T_WC @ T_CB  (world-from-body = body pose in world)
        t_wb = t_wc @ self._t_cb

        # Extract position and orientation
        position_w, orientation_w = matrix_to_pose(t_wb)

        # --- Noise scaling based on distance ---
        distance = float(np.linalg.norm(t_cg_vec))
        # Scale noise with distance (farther = noisier)
        distance_factor = max(1.0, distance / 3.0)  # normalised to ~3m
        pos_noise = self._est_cfg.pnp_position_noise_std * distance_factor
        ori_noise = self._est_cfg.pnp_orientation_noise_std * distance_factor

        return PoseMeasurement(
            timestamp_ns=detection.timestamp_ns,
            position_w=position_w,
            orientation_w=orientation_w,
            position_noise_std=pos_noise,
            orientation_noise_std=ori_noise,
            confidence=detection.confidence_level,
            gate_id=gate.gate_id,
        )

    def solve_multiple(
        self,
        detections: list[GateDetection],
        gates: list[GateWaypoint],
        gate_corners_3d_local: np.ndarray,
    ) -> list[PoseMeasurement]:
        """Solve PnP for multiple detections against a gate map.

        Each detection is matched to the nearest gate by ID (if set) or
        skipped if no match. All successful solutions are returned.

        Parameters
        ----------
        detections : list[GateDetection]
            Detected gates in the current frame.
        gates : list[GateWaypoint]
            Known gate positions (the gate map).
        gate_corners_3d_local : np.ndarray
            (N, 3) gate corners in the gate's local frame.

        Returns
        -------
        list[PoseMeasurement]
            Successful pose measurements (may be empty).
        """
        # Build gate lookup by ID
        gate_by_id: dict[int, GateWaypoint] = {g.gate_id: g for g in gates}

        measurements: list[PoseMeasurement] = []
        for det in detections:
            if det.gate_id is None:
                # No ID assigned — skip (tracker hasn't matched this detection yet)
                logger.debug("PnP: detection has no gate_id — skipping")
                continue

            gate = gate_by_id.get(det.gate_id)
            if gate is None:
                logger.debug("PnP: no gate with id=%d in map — skipping", det.gate_id)
                continue

            meas = self.solve(det, gate, gate_corners_3d_local)
            if meas is not None:
                measurements.append(meas)

        logger.debug(
            "PnP: %d/%d detections produced measurements",
            len(measurements),
            len(detections),
        )
        return measurements
