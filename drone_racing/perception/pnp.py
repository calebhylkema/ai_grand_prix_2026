"""PnP pose solver — converts 2D gate corners to a 6-DOF pose measurement.

Takes detected gate corners (2D pixels) and the known 3D gate model,
solves PnP to get the camera-to-gate transform, then chains through
the known gate world pose and camera extrinsic to produce a
world-frame drone pose for the ESKF.

Transform chain::

    T_WB = T_WG @ T_GC^{-1} @ T_CB

Where:
    - T_WG: gate pose in world frame (from gate map)
    - T_CG: camera-from-gate transform (from PnP — what OpenCV gives us)
    - T_CB: body-to-camera extrinsic (from calibration / config)
    - T_WB: drone body pose in world frame (what the ESKF needs)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.utils.transforms import (
    body_to_camera_rotation,
    camera_to_body_rotation,
    invert_transform,
    matrix_to_pose,
    pose_to_matrix,
)
from drone_racing.utils.types import (
    GateDetection,
    GateWaypoint,
    PoseMeasurement,
    State,
)

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig, EstimationConfig

logger = logging.getLogger(__name__)


class PnPSolver:
    """Solve Perspective-n-Point to convert gate detections into pose measurements.

    Uses the drone's approximate state (from ESKF) to resolve the PnP
    ambiguity for coplanar points by: (1) matching detected 2D corners to
    projected 3D corners, and (2) providing an initial guess to solvePnP.

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
        # Body-to-camera rotation
        self._r_cb = body_to_camera_rotation(camera_config.tilt_deg)

        # Build the full 4×4 body-to-camera transform (translation assumed zero)
        # T_CB maps body→camera, so its rotation is R_CB = R_BC^T
        self._t_cb = np.eye(4, dtype=np.float64)
        self._t_cb[:3, :3] = self._r_bc.T

        self._est_cfg = estimation_config

    def _compute_approximate_t_cg(
        self,
        drone_state: State,
        gate: GateWaypoint,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute approximate camera-from-gate transform using drone state.

        Parameters
        ----------
        drone_state : State
            Current ESKF state estimate.
        gate : GateWaypoint
            Gate in world frame.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (rvec, tvec) for solvePnP initial guess.
        """
        # T_WB: body pose in world
        r_wb = Rotation.from_quat(drone_state.orientation).as_matrix()

        # R_CW = R_CB @ R_BW = R_CB @ R_WB^T
        r_cw = self._r_cb @ r_wb.T
        t_cw = -r_cw @ drone_state.position

        # T_WG: gate pose in world
        r_wg = Rotation.from_quat(gate.orientation_w).as_matrix()

        # R_CG = R_CW @ R_WG
        r_cg = r_cw @ r_wg
        t_cg = r_cw @ gate.position_w + t_cw

        rvec, _ = cv2.Rodrigues(r_cg)
        tvec = t_cg.reshape(3, 1)
        return rvec, tvec

    def _match_corners(
        self,
        gate_corners_3d_local: np.ndarray,
        detected_corners_2d: np.ndarray,
        rvec_guess: np.ndarray,
        tvec_guess: np.ndarray,
    ) -> np.ndarray:
        """Reorder 3D model corners to match detected 2D corner ordering.

        Projects the 3D corners using the approximate pose, then matches
        each detected 2D corner to the nearest projected corner.

        Parameters
        ----------
        gate_corners_3d_local : np.ndarray
            (N, 3) gate corners in gate local frame.
        detected_corners_2d : np.ndarray
            (N, 2) detected corner pixels from the detector.
        rvec_guess : np.ndarray
            Approximate rotation vector (camera-from-gate).
        tvec_guess : np.ndarray
            Approximate translation vector.

        Returns
        -------
        np.ndarray
            (N, 3) reordered 3D corners matching the 2D detection order.
        """
        # Project 3D corners using the approximate pose
        projected, _ = cv2.projectPoints(
            gate_corners_3d_local,
            rvec_guess,
            tvec_guess,
            self._camera_k,
            self._camera_dist,
        )
        projected_2d = projected.reshape(-1, 2)

        n = len(detected_corners_2d)
        reordered = np.empty_like(gate_corners_3d_local[:n])

        # For each detected corner, find nearest projected corner
        used: set[int] = set()
        for i in range(n):
            dists = np.linalg.norm(projected_2d - detected_corners_2d[i], axis=1)
            # Mask already-used indices
            for u in used:
                dists[u] = float("inf")
            best_j = int(np.argmin(dists))
            used.add(best_j)
            reordered[i] = gate_corners_3d_local[best_j]

        return reordered

    def solve(
        self,
        detection: GateDetection,
        gate: GateWaypoint,
        gate_corners_3d_local: np.ndarray,
        drone_state: State | None = None,
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
        drone_state : State | None
            Approximate drone state for corner matching and initial guess.
            If None, falls back to direct solvePnP (may be ambiguous).

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

        n_model = len(gate_corners_3d_local)
        n_use = min(n_corners, n_model)
        if n_use < 4:
            logger.debug("PnP: not enough matching corners (%d) — skipping", n_use)
            return None

        img_pts = corners_2d[:n_use].astype(np.float64)

        # --- Corner matching + PnP with initial guess ---
        if drone_state is not None:
            rvec_guess, tvec_guess = self._compute_approximate_t_cg(drone_state, gate)

            # Reorder 3D corners to match detected 2D ordering
            obj_pts = self._match_corners(
                gate_corners_3d_local[:n_use],
                img_pts,
                rvec_guess,
                tvec_guess,
            )

            # Solve PnP with initial guess to resolve coplanar ambiguity
            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                self._camera_k,
                self._camera_dist,
                rvec=rvec_guess.copy(),
                tvec=tvec_guess.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            obj_pts = gate_corners_3d_local[:n_use].astype(np.float64)
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

        # Compute reprojection error for diagnostics
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, self._camera_k, self._camera_dist)
        reproj_err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_pts, axis=1)))
        logger.debug("PnP: gate %s reprojection error = %.2f px", gate.gate_id, reproj_err)

        # Reject solutions with high reprojection error
        if reproj_err > 10.0:
            logger.debug(
                "PnP: rejecting gate %s — reprojection error %.1f px too high",
                gate.gate_id,
                reproj_err,
            )
            return None

        # Convert rvec to rotation matrix
        r_cg, _ = cv2.Rodrigues(rvec)
        t_cg_vec = tvec.flatten()

        # Reject if gate is too far away (PnP unreliable at distance due to
        # small angular errors amplifying to large position errors)
        distance = float(np.linalg.norm(t_cg_vec))
        if distance > 12.0:
            logger.debug(
                "PnP: rejecting gate %s — distance %.1fm too far",
                gate.gate_id,
                distance,
            )
            return None

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

        # Sanity check: PnP position should be close to current ESKF estimate
        if drone_state is not None:
            pos_diff = float(np.linalg.norm(position_w - drone_state.position))
            if pos_diff > 5.0:
                logger.debug(
                    "PnP: rejecting gate %s — position %.1fm from ESKF estimate",
                    gate.gate_id,
                    pos_diff,
                )
                return None

        # --- Noise scaling based on distance ---
        # Scale noise with distance (farther = noisier)
        distance_factor = max(1.0, distance / 3.0)  # normalised to ~3m
        pos_noise = self._est_cfg.pnp_position_noise_std * distance_factor
        ori_noise = self._est_cfg.pnp_orientation_noise_std * distance_factor

        # Cap confidence at MEDIUM when using only 4 model corners.
        # Full 6-DOF corrections require reliable orientation data, which
        # needs more than 4 coplanar points (PnP orientation is ambiguous).
        from drone_racing.utils.types import DetectionConfidence

        confidence = detection.confidence_level
        if n_use <= 4 and confidence == DetectionConfidence.HIGH:
            confidence = DetectionConfidence.MEDIUM

        return PoseMeasurement(
            timestamp_ns=detection.timestamp_ns,
            position_w=position_w,
            orientation_w=orientation_w,
            position_noise_std=pos_noise,
            orientation_noise_std=ori_noise,
            confidence=confidence,
            gate_id=gate.gate_id,
        )

    def solve_multiple(
        self,
        detections: list[GateDetection],
        gates: list[GateWaypoint],
        gate_corners_3d_local: np.ndarray,
        drone_state: State | None = None,
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
        drone_state : State | None
            Approximate drone state for corner matching and initial guess.

        Returns
        -------
        list[PoseMeasurement]
            Successful pose measurements (may be empty).
        """
        gate_by_id: dict[int, GateWaypoint] = {g.gate_id: g for g in gates}

        measurements: list[PoseMeasurement] = []
        for det in detections:
            if det.gate_id is None:
                logger.debug("PnP: detection has no gate_id — skipping")
                continue

            gate = gate_by_id.get(det.gate_id)
            if gate is None:
                logger.debug("PnP: no gate with id=%d in map — skipping", det.gate_id)
                continue

            meas = self.solve(det, gate, gate_corners_3d_local, drone_state)
            if meas is not None:
                measurements.append(meas)

        logger.debug(
            "PnP: %d/%d detections produced measurements",
            len(measurements),
            len(detections),
        )
        return measurements
