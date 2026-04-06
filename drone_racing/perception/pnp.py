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
        max_reprojection_px: float = 10.0,
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
        self._max_reproj_px = max_reprojection_px

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

    def _solve_pnp_ippe_best(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        drone_state: State,
        gate: GateWaypoint,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Solve PnP with IPPE and pick the solution closest to the ESKF state.

        IPPE (Infinitesimal Plane-based Pose Estimation) is designed for
        coplanar points and returns up to 2 solutions.  We pick the one
        whose implied drone position is closest to the current ESKF estimate.

        Parameters
        ----------
        obj_pts : np.ndarray
            (N, 3) 3D object points in gate-local frame.
        img_pts : np.ndarray
            (N, 2) detected 2D image points.
        drone_state : State
            Current ESKF state estimate for disambiguation.
        gate : GateWaypoint
            Gate pose in world frame.

        Returns
        -------
        tuple[bool, np.ndarray, np.ndarray]
            (success, rvec, tvec) best solution.
        """
        try:
            n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
                obj_pts,
                img_pts,
                self._camera_k,
                self._camera_dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            return False, np.zeros(3), np.zeros(3)

        if n_solutions == 0:
            return False, np.zeros(3), np.zeros(3)

        # Score each solution by distance from ESKF estimate
        best_rvec = rvecs[0]
        best_tvec = tvecs[0]
        best_dist = float("inf")

        try:
            t_wg = pose_to_matrix(gate.position_w, gate.orientation_w)

            for i in range(n_solutions):
                rv = rvecs[i]
                tv = tvecs[i]
                r_cg, _ = cv2.Rodrigues(rv)
                t_cg = np.eye(4, dtype=np.float64)
                t_cg[:3, :3] = r_cg
                t_cg[:3, 3] = tv.flatten()
                t_gc = invert_transform(t_cg)
                t_wc = t_wg @ t_gc
                t_wb = t_wc @ self._t_cb
                pos_w = t_wb[:3, 3]
                dist = float(np.linalg.norm(pos_w - drone_state.position))
                if dist < best_dist:
                    best_dist = dist
                    best_rvec = rv
                    best_tvec = tv
        except (np.linalg.LinAlgError, ValueError):
            return False, np.zeros(3), np.zeros(3)

        return True, best_rvec, best_tvec

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

        # --- Corner matching + PnP ---
        if drone_state is not None:
            rvec_guess, tvec_guess = self._compute_approximate_t_cg(drone_state, gate)

            # Match 2D→3D corner ordering. For 8 corners, match outer and
            # inner rings separately to prevent cross-assignment.
            if n_use <= 4:
                obj_pts = self._match_corners(
                    gate_corners_3d_local[:n_use],
                    img_pts,
                    rvec_guess,
                    tvec_guess,
                )
            else:
                # Match outer 4 and inner 4 independently
                outer_3d = self._match_corners(
                    gate_corners_3d_local[:4], img_pts[:4], rvec_guess, tvec_guess
                )
                inner_3d = self._match_corners(
                    gate_corners_3d_local[4:n_use],
                    img_pts[4:n_use],
                    rvec_guess,
                    tvec_guess,
                )
                obj_pts = np.vstack([outer_3d, inner_3d])

            if n_use >= 6:
                # 6+ corners: use IPPE (coplanar solver) which returns both
                # solutions, then pick the one closest to our ESKF estimate.
                success, rvec, tvec = self._solve_pnp_ippe_best(obj_pts, img_pts, drone_state, gate)
            else:
                # 4 corners: ITERATIVE with extrinsic guess
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
        if reproj_err > self._max_reproj_px:
            logger.debug(
                "PnP: rejecting gate %s — reprojection error %.1f px too high",
                gate.gate_id,
                reproj_err,
            )
            return None

        # Convert rvec to rotation matrix
        r_cg, _ = cv2.Rodrigues(rvec)
        t_cg_vec = tvec.flatten()

        # Reject if gate is too close (PnP unreliable with 4 coplanar
        # points at extreme viewing angles) or too far away (small angular
        # errors amplify to large position errors)
        distance = float(np.linalg.norm(t_cg_vec))
        if distance < 2.0:
            logger.debug(
                "PnP: rejecting gate %s — distance %.1fm too close",
                gate.gate_id,
                distance,
            )
            return None
        if distance > 18.0:
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
        try:
            position_w, orientation_w = matrix_to_pose(t_wb)
        except np.linalg.LinAlgError:
            logger.debug("PnP: degenerate rotation for gate %s — skipping", gate.gate_id)
            return None

        # Sanity check: PnP position should be close to current ESKF estimate.
        # Use a generous threshold — the filter itself handles outliers via chi².
        if drone_state is not None:
            pos_diff = float(np.linalg.norm(position_w - drone_state.position))
            if pos_diff > 15.0:
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

        # Cap confidence at MEDIUM when the detection has only 4 corners.
        # Full 6-DOF corrections require reliable orientation data.  With
        # 8 detected corners (even if PnP only uses 4 model corners),
        # the detection is reliable enough for HIGH confidence.
        from drone_racing.utils.types import DetectionConfidence

        confidence = detection.confidence_level
        if n_corners <= 4 and confidence == DetectionConfidence.HIGH:
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

        For each detection:
        - If it has a ``gate_id``, try that gate first.
        - Otherwise (or if the assigned gate fails), try **all** gates and
          pick the solution with the lowest reprojection error.

        This makes the PnP robust to incorrect gate-ID assignments from
        the tracker, which can happen when the ESKF state drifts.

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
        used_gate_ids: set[int] = set()

        measurements: list[PoseMeasurement] = []
        for det in detections:
            meas: PoseMeasurement | None = None

            # Try the assigned gate_id first (fast path)
            if det.gate_id is not None:
                gate = gate_by_id.get(det.gate_id)
                if gate is not None:
                    meas = self.solve(det, gate, gate_corners_3d_local, drone_state)

            # Fallback: try all gates, pick lowest reproj error
            if meas is None:
                meas = self._solve_best_gate(
                    det, gates, gate_corners_3d_local, drone_state, used_gate_ids
                )

            if meas is not None:
                measurements.append(meas)
                if meas.gate_id is not None:
                    used_gate_ids.add(meas.gate_id)

        logger.debug(
            "PnP: %d/%d detections produced measurements",
            len(measurements),
            len(detections),
        )
        return measurements

    def _solve_best_gate(
        self,
        detection: GateDetection,
        gates: list[GateWaypoint],
        gate_corners_3d_local: np.ndarray,
        drone_state: State | None,
        exclude_ids: set[int],
    ) -> PoseMeasurement | None:
        """Try PnP against all gates and return the best match.

        Parameters
        ----------
        detection : GateDetection
            Detected 2D corners.
        gates : list[GateWaypoint]
            All known gates.
        gate_corners_3d_local : np.ndarray
            (N, 3) gate corners in gate local frame.
        drone_state : State | None
            Approximate drone state.
        exclude_ids : set[int]
            Gate IDs already matched to other detections in this frame.

        Returns
        -------
        PoseMeasurement | None
            Best matching measurement, or None.
        """
        best_meas: PoseMeasurement | None = None
        best_reproj = float("inf")

        for gate in gates:
            if gate.gate_id in exclude_ids:
                continue

            # Quick distance check to avoid solving PnP for far-away gates
            if drone_state is not None:
                dist_to_gate = float(np.linalg.norm(gate.position_w - drone_state.position))
                if dist_to_gate > 25.0:
                    continue

            try:
                meas = self.solve(detection, gate, gate_corners_3d_local, drone_state)
            except cv2.error:
                continue
            if meas is None:
                continue

            # Use position noise as a proxy for quality — lower noise = closer
            # gate = more reliable match (noise scales with distance).
            reproj_proxy = meas.position_noise_std
            if reproj_proxy < best_reproj:
                best_reproj = reproj_proxy
                best_meas = meas

        if best_meas is not None:
            logger.debug(
                "PnP: best-gate match → gate %s (noise=%.2f)",
                best_meas.gate_id,
                best_reproj,
            )

        return best_meas
