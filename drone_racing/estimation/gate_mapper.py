"""Gate-map estimator — builds a world-frame gate map from PnP observations.

The ESKF needs a known set of gate world poses to turn PnP camera-from-gate
measurements into drone pose corrections.  During VQ1 the course is not
provided — we have to *discover* each gate's world pose the first few times
we see it, then hand the consolidated map to the racing-line planner.

Flow
----
1. Detector → tracker produces a :class:`GateDetection` tagged with a stable
   track ``gate_id``.
2. :meth:`GateMapper.observe` runs PnP on the detection against the known
   *gate-local* corner model (not the world model — that's what we're
   estimating), giving :math:`T_{CG}`.
3. We chain through the current drone state to recover the gate pose in
   world frame:

       T_WG = T_WB @ T_BC @ T_CG

   where :math:`T_{WB}` comes from the ESKF and :math:`T_{BC}` is a fixed
   camera extrinsic.
4. Observations are aggregated per track ID with inverse-distance weights.
   Once a track has enough consistent observations, :meth:`waypoints`
   emits a list of :class:`GateWaypoint`.

This module is intentionally decoupled from the ESKF: it *consumes* the
filter's state but does not feed back.  The correction path (PnP → ESKF)
only runs after a gate has been promoted to the consolidated map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.utils.transforms import (
    camera_to_body_rotation,
    matrix_to_pose,
    pose_to_matrix,
    quat_normalize,
)
from drone_racing.utils.types import GateWaypoint

if TYPE_CHECKING:
    from drone_racing.utils.types import (
        CameraConfig,
        GateDetection,
        State,
        Vec3,
        Vec4,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GateMapperConfig:
    """Tuneable parameters for the gate-map estimator.

    Parameters
    ----------
    min_observations : int
        Number of consistent PnP observations required before a gate is
        promoted to the consolidated map.
    max_reproj_px : float
        Maximum mean reprojection error (pixels) for an observation to
        count.  Noisy fits are dropped.
    min_distance : float
        Minimum drone–gate distance (m) for a valid observation.  PnP is
        unreliable with tight foreshortening.
    max_distance : float
        Maximum drone–gate distance (m).  Far observations amplify angular
        error into large world-frame position error.
    position_consistency_m : float
        A new observation whose implied gate position is more than this
        many metres from the running mean is treated as an outlier and
        rejected (unless no observations exist yet).
    distance_norm_m : float
        Reference distance for the inverse-distance weighting.
        Observations at this distance get weight 1.0.
    """

    min_observations: int = 3
    max_reproj_px: float = 10.0
    min_distance: float = 1.5
    max_distance: float = 18.0
    position_consistency_m: float = 1.5
    distance_norm_m: float = 3.0


# ---------------------------------------------------------------------------
# Internal per-track aggregate
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GateAggregate:
    """Accumulated PnP observations for a single tracked gate."""

    gate_id: int
    """Tracker-assigned gate ID (becomes the map gate ID)."""

    weight_sum: float = 0.0
    """Sum of observation weights."""

    position_accum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    """Weighted sum of observed gate positions (world frame)."""

    quat_accum: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    """Weighted sum of observed orientation quaternions (hemisphere-aligned).

    Averaging quaternions by component and renormalising is a valid
    first-order approximation when the rotations are close together, which
    is the case for repeated observations of the same gate.
    """

    corners_accum: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    """Weighted sum of world-frame corner positions, shape (N, 3)."""

    n_observations: int = 0
    """Count of valid observations accepted."""

    last_reproj_px: float = 0.0
    """Most recent reprojection error, for diagnostics."""


def _weighted_mean_position(agg: _GateAggregate) -> Vec3:
    return agg.position_accum / agg.weight_sum


def _weighted_mean_quat(agg: _GateAggregate) -> Vec4:
    q = agg.quat_accum / agg.weight_sum
    return quat_normalize(q)


def _weighted_mean_corners(agg: _GateAggregate) -> np.ndarray:
    return agg.corners_accum / agg.weight_sum


# ---------------------------------------------------------------------------
# Pure PnP helper: solve T_CG against a *gate-local* model
# ---------------------------------------------------------------------------


def estimate_gate_pose_world(
    detection: GateDetection,
    drone_state: State,
    gate_corners_3d_local: np.ndarray,
    camera_config: CameraConfig,
    max_reproj_px: float = 10.0,
) -> tuple[Vec3, Vec4, np.ndarray, float] | None:
    """Estimate a single gate's world pose from one detection.

    Unlike :class:`drone_racing.perception.pnp.PnPSolver`, this routine
    *does not* assume the gate's world pose is known — that is what we are
    trying to recover.  The drone's world pose (from the ESKF) is taken as
    ground truth for this purpose.

    Parameters
    ----------
    detection : GateDetection
        Tracked gate detection with corner pixels.
    drone_state : State
        Current ESKF state estimate (provides :math:`T_{WB}`).
    gate_corners_3d_local : np.ndarray
        ``(N, 3)`` gate corner positions in the gate's local frame.
    camera_config : CameraConfig
        Camera intrinsics and tilt.
    max_reproj_px : float
        Reject fits whose mean reprojection error exceeds this.

    Returns
    -------
    tuple[Vec3, Vec4, np.ndarray, float] | None
        ``(position_w, orientation_w, corners_w, reproj_err_px)`` or
        ``None`` on failure.  ``corners_w`` has the same shape as
        ``gate_corners_3d_local`` and is the gate-local model transformed
        into the world frame.
    """
    corners_2d = detection.corners_2d
    n_corners = int(corners_2d.shape[0])
    n_model = int(gate_corners_3d_local.shape[0])
    n_use = min(n_corners, n_model)
    if n_use < 4:
        logger.debug("GateMapper: need ≥4 corners, got %d", n_use)
        return None

    obj_pts = np.ascontiguousarray(gate_corners_3d_local[:n_use], dtype=np.float64)
    img_pts = np.ascontiguousarray(corners_2d[:n_use], dtype=np.float64)

    k = camera_config.intrinsic_matrix
    dist = camera_config.distortion_coeffs

    try:
        if n_use >= 4 and np.allclose(gate_corners_3d_local[:n_use, 2], 0.0):
            # Coplanar: IPPE is the preferred solver; pick the solution
            # with lower reprojection error.
            n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                obj_pts, img_pts, k, dist, flags=cv2.SOLVEPNP_IPPE
            )
            if n_sol == 0:
                return None
            errs = np.asarray(reproj_errs, dtype=np.float64).reshape(-1)
            best = int(np.argmin(errs))
            rvec = rvecs[best]
            tvec = tvecs[best]
            success = True
        else:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, k, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
    except cv2.error:
        logger.debug("GateMapper: cv2 error during PnP — degenerate geometry")
        return None

    if not success:
        return None

    # Reprojection error check
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist)
    reproj_err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_pts, axis=1)))
    if reproj_err > max_reproj_px:
        logger.debug("GateMapper: reproj %.2f px > %.2f", reproj_err, max_reproj_px)
        return None

    # Build T_CG
    r_cg, _ = cv2.Rodrigues(rvec)
    t_cg = np.eye(4, dtype=np.float64)
    t_cg[:3, :3] = r_cg
    t_cg[:3, 3] = tvec.flatten()

    # T_BC: camera-to-body transform (R_BC, zero translation)
    r_bc = camera_to_body_rotation(camera_config.tilt_deg)
    t_bc = np.eye(4, dtype=np.float64)
    t_bc[:3, :3] = r_bc

    # T_WB: body-to-world transform from ESKF
    t_wb = pose_to_matrix(drone_state.position, drone_state.orientation)

    # T_WG = T_WB @ T_BC @ T_CG
    t_wg = t_wb @ t_bc @ t_cg
    try:
        position_w, orientation_w = matrix_to_pose(t_wg)
    except (np.linalg.LinAlgError, ValueError):
        logger.debug("GateMapper: degenerate rotation from PnP — skipping")
        return None

    # Transform the local corner model into world frame.
    r_wg = t_wg[:3, :3]
    corners_w = (r_wg @ gate_corners_3d_local.T).T + position_w

    return position_w, orientation_w, corners_w, reproj_err


# ---------------------------------------------------------------------------
# Gate mapper
# ---------------------------------------------------------------------------


class GateMapper:
    """Accumulate PnP observations into a consolidated world-frame gate map.

    Parameters
    ----------
    camera_config : CameraConfig
        Camera intrinsics and tilt.
    gate_corners_3d_local : np.ndarray
        ``(N, 3)`` canonical gate corners in the gate's local frame.  Same
        layout as :attr:`GateConfig.corners_3d_array`.
    config : GateMapperConfig | None
        Tuneable parameters; defaults if omitted.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        gate_corners_3d_local: np.ndarray,
        config: GateMapperConfig | None = None,
    ) -> None:
        self._camera_cfg = camera_config
        self._model_corners = np.ascontiguousarray(gate_corners_3d_local, dtype=np.float64)
        self._cfg = config or GateMapperConfig()
        self._aggregates: dict[int, _GateAggregate] = {}
        self._next_order: int = 0
        self._order_assignments: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> GateMapperConfig:
        """Configuration in use (read-only)."""
        return self._cfg

    @property
    def tracked_gate_ids(self) -> list[int]:
        """IDs of all gates with at least one accepted observation."""
        return list(self._aggregates.keys())

    def observation_count(self, gate_id: int) -> int:
        """Number of accepted observations for *gate_id* (0 if unseen)."""
        agg = self._aggregates.get(gate_id)
        return agg.n_observations if agg is not None else 0

    def reset(self) -> None:
        """Drop all accumulated observations."""
        self._aggregates.clear()
        self._order_assignments.clear()
        self._next_order = 0

    def observe(
        self,
        detection: GateDetection,
        drone_state: State,
    ) -> bool:
        """Ingest a single tracked detection.

        Parameters
        ----------
        detection : GateDetection
            Must have ``gate_id`` assigned by the tracker; otherwise
            ignored.
        drone_state : State
            Current ESKF state estimate.

        Returns
        -------
        bool
            True if the observation was accepted and fused; False if it
            was rejected (missing gate_id, bad PnP, distance/consistency
            gating).
        """
        if detection.gate_id is None:
            return False

        result = estimate_gate_pose_world(
            detection,
            drone_state,
            self._model_corners,
            self._camera_cfg,
            max_reproj_px=self._cfg.max_reproj_px,
        )
        if result is None:
            return False

        position_w, orientation_w, corners_w, reproj_err = result

        # Distance gating
        distance = float(np.linalg.norm(position_w - drone_state.position))
        if distance < self._cfg.min_distance or distance > self._cfg.max_distance:
            logger.debug(
                "GateMapper: gate %d distance %.1fm outside [%.1f, %.1f]",
                detection.gate_id,
                distance,
                self._cfg.min_distance,
                self._cfg.max_distance,
            )
            return False

        # Weight: inverse (squared) distance, normalised at distance_norm_m
        weight = (self._cfg.distance_norm_m / max(distance, 0.5)) ** 2
        # Also scale by detection confidence to suppress weak fits
        weight *= max(float(detection.confidence), 0.1)

        agg = self._aggregates.get(detection.gate_id)
        if agg is None:
            agg = _GateAggregate(
                gate_id=detection.gate_id,
                corners_accum=np.zeros_like(corners_w),
            )
            self._aggregates[detection.gate_id] = agg
        else:
            # Outlier rejection against the running mean
            running_pos = _weighted_mean_position(agg)
            delta = float(np.linalg.norm(position_w - running_pos))
            if delta > self._cfg.position_consistency_m:
                logger.debug(
                    "GateMapper: gate %d observation %.2fm off mean — rejecting",
                    detection.gate_id,
                    delta,
                )
                return False

            # Hemisphere-align quaternion to running mean before accumulating
            running_q = _weighted_mean_quat(agg)
            if float(np.dot(orientation_w, running_q)) < 0.0:
                orientation_w = -orientation_w

        agg.weight_sum += weight
        agg.position_accum += weight * position_w
        agg.quat_accum += weight * orientation_w
        agg.corners_accum += weight * corners_w
        agg.n_observations += 1
        agg.last_reproj_px = reproj_err
        return True

    def observe_many(
        self,
        detections: list[GateDetection],
        drone_state: State,
    ) -> int:
        """Ingest a batch of detections; return the number accepted."""
        accepted = 0
        for det in detections:
            if self.observe(det, drone_state):
                accepted += 1
        return accepted

    def waypoints(self) -> list[GateWaypoint]:
        """Return consolidated :class:`GateWaypoint` list for promoted gates.

        Gates with fewer than :attr:`GateMapperConfig.min_observations`
        accepted observations are excluded.

        The ``order`` field is assigned in the order gates were first
        promoted — callers that need a geometric ordering (lap sequence)
        should re-sort using the racing-line planner once the map
        stabilises.
        """
        out: list[GateWaypoint] = []
        for gate_id, agg in self._aggregates.items():
            if agg.n_observations < self._cfg.min_observations:
                continue
            if agg.weight_sum <= 0.0:
                continue
            if gate_id not in self._order_assignments:
                self._order_assignments[gate_id] = self._next_order
                self._next_order += 1
            out.append(
                GateWaypoint(
                    gate_id=gate_id,
                    position_w=_weighted_mean_position(agg),
                    orientation_w=_weighted_mean_quat(agg),
                    corners_3d_w=_weighted_mean_corners(agg),
                    order=self._order_assignments[gate_id],
                )
            )
        out.sort(key=lambda g: g.order)
        return out
