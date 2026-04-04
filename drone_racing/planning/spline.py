"""B-spline racing line through gate waypoints.

Layer 1 of the two-layer planning architecture:

1. Fit a cubic B-spline through gate centres with approach/exit offsets.
2. Compute arc-length parameterisation for uniform sampling.
3. Generate a curvature-based speed profile (slow in turns, fast on straights).

The output is a :class:`RacingLine` that downstream modules (MPCC, PID tracker)
consume to get position, tangent, and speed at any progress value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from drone_racing.utils.types import PlannerConfig, Vec3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Racing line data
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RacingLine:
    """Precomputed racing line — the reference path for the controller.

    All arrays are indexed by a uniform progress parameter ``s ∈ [0, 1]``
    mapped to ``num_points`` samples.
    """

    positions: np.ndarray
    """(N, 3) waypoint positions along the spline, world frame (m)."""

    tangents: np.ndarray
    """(N, 3) unit tangent vectors at each waypoint."""

    curvatures: np.ndarray
    """(N,) curvature magnitude at each waypoint (1/m)."""

    speeds: np.ndarray
    """(N,) speed profile at each waypoint (m/s)."""

    arc_lengths: np.ndarray
    """(N,) cumulative arc length from start (m)."""

    total_length: float
    """Total path length (m)."""

    def at_progress(self, s: float) -> tuple[Vec3, Vec3, float]:
        """Interpolate position, tangent, and speed at normalised progress *s*.

        Parameters
        ----------
        s : float
            Progress along the racing line, 0.0 = start, 1.0 = finish.

        Returns
        -------
        tuple[Vec3, Vec3, float]
            (position, tangent, speed) at progress *s*.
        """
        s = float(np.clip(s, 0.0, 1.0))
        n = len(self.positions)
        idx_f = s * (n - 1)
        idx_lo = int(np.floor(idx_f))
        idx_hi = min(idx_lo + 1, n - 1)
        alpha = idx_f - idx_lo

        pos = (1.0 - alpha) * self.positions[idx_lo] + alpha * self.positions[idx_hi]
        tan = (1.0 - alpha) * self.tangents[idx_lo] + alpha * self.tangents[idx_hi]
        tan_norm = float(np.linalg.norm(tan))
        if tan_norm > 1e-12:
            tan = tan / tan_norm
        spd = (1.0 - alpha) * self.speeds[idx_lo] + alpha * self.speeds[idx_hi]

        return pos, tan, float(spd)

    def closest_progress(self, position: Vec3) -> float:
        """Find the normalised progress *s* of the closest point on the line.

        Uses brute-force over the discretised points — fast enough for the
        number of samples we use (few hundred).

        Parameters
        ----------
        position : Vec3
            Query position in world frame (m).

        Returns
        -------
        float
            Progress s ∈ [0, 1].
        """
        diffs = self.positions - position[np.newaxis, :]
        dists_sq = np.sum(diffs**2, axis=1)
        idx = int(np.argmin(dists_sq))
        return idx / (len(self.positions) - 1)


# ---------------------------------------------------------------------------
# Spline builder
# ---------------------------------------------------------------------------


def build_racing_line(
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    config: PlannerConfig,
    num_points: int = 200,
    closed: bool = True,
) -> RacingLine:
    """Fit a cubic spline through gate centres and compute a speed profile.

    Parameters
    ----------
    gate_positions : np.ndarray
        (G, 3) gate centre positions in world frame, ordered by passage.
    gate_orientations : np.ndarray
        (G, 4) gate orientation quaternions [x, y, z, w].  The gate normal
        (local Z) defines the approach direction.
    config : PlannerConfig
        Planner parameters (smoothing, offsets, max velocity).
    num_points : int
        Number of evenly-spaced samples along the output racing line.
    closed : bool
        If True, close the loop (last gate → first gate).

    Returns
    -------
    RacingLine
        Precomputed racing line with positions, tangents, curvatures, speeds.
    """
    n_gates = len(gate_positions)
    if n_gates < 2:
        msg = f"Need ≥2 gates to build a racing line, got {n_gates}"
        raise ValueError(msg)

    # Build waypoints: for each gate, add approach and exit points
    # offset along the gate normal (local Z in gate frame)
    waypoints = _build_waypoints(gate_positions, gate_orientations, config, closed)

    # Fit cubic spline
    positions, tangents_raw, curvatures = _fit_spline(waypoints, num_points, closed)

    # Normalise tangents
    tangent_norms = np.linalg.norm(tangents_raw, axis=1, keepdims=True)
    tangent_norms = np.maximum(tangent_norms, 1e-12)
    tangents = tangents_raw / tangent_norms

    # Arc lengths
    diffs = np.diff(positions, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.zeros(num_points)
    arc_lengths[1:] = np.cumsum(seg_lengths)
    total_length = float(arc_lengths[-1])

    # Speed profile based on curvature
    speeds = _curvature_speed_profile(curvatures, config)

    logger.info(
        "Racing line: %d gates, %.1f m total, %d samples",
        n_gates,
        total_length,
        num_points,
    )

    return RacingLine(
        positions=positions,
        tangents=tangents,
        curvatures=curvatures,
        speeds=speeds,
        arc_lengths=arc_lengths,
        total_length=total_length,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_waypoints(
    gate_positions: np.ndarray,
    gate_orientations: np.ndarray,
    config: PlannerConfig,
    closed: bool,
) -> np.ndarray:
    """Insert approach/exit waypoints around each gate centre.

    For each gate we add:
        1. An approach point: gate_centre - offset * gate_normal
        2. The gate centre itself
        3. An exit point:     gate_centre + offset * gate_normal

    Parameters
    ----------
    gate_positions : np.ndarray
        (G, 3) gate centres.
    gate_orientations : np.ndarray
        (G, 4) quaternions.
    config : PlannerConfig
        Contains ``gate_approach_offset_m``.
    closed : bool
        Whether to close the loop.

    Returns
    -------
    np.ndarray
        (W, 3) ordered waypoints.
    """
    from scipy.spatial.transform import Rotation

    offset = config.gate_approach_offset_m
    waypoints = []

    for i in range(len(gate_positions)):
        pos = gate_positions[i]
        # Gate normal is the local Z-axis of the gate orientation
        r = Rotation.from_quat(gate_orientations[i])
        normal = r.apply(np.array([0.0, 0.0, 1.0]))

        waypoints.append(pos - offset * normal)  # approach
        waypoints.append(pos)  # gate centre
        waypoints.append(pos + offset * normal)  # exit

    if closed:
        # Repeat the first approach point to close the loop
        pos_first = gate_positions[0]
        r_first = Rotation.from_quat(gate_orientations[0])
        normal_first = r_first.apply(np.array([0.0, 0.0, 1.0]))
        waypoints.append(pos_first - offset * normal_first)

    return np.array(waypoints, dtype=np.float64)


def _fit_spline(
    waypoints: np.ndarray,
    num_points: int,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a cubic spline through waypoints and sample positions + derivatives.

    Parameters
    ----------
    waypoints : np.ndarray
        (W, 3) ordered waypoints.
    num_points : int
        Number of output samples.
    closed : bool
        If True, use periodic boundary conditions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (positions (N,3), tangents (N,3), curvatures (N,)).
    """
    n_wp = len(waypoints)
    # Parameter: cumulative chord length
    diffs = np.diff(waypoints, axis=0)
    chord_lengths = np.linalg.norm(diffs, axis=1)
    t_wp = np.zeros(n_wp)
    t_wp[1:] = np.cumsum(chord_lengths)

    # Normalise to [0, 1]
    if t_wp[-1] > 1e-12:
        t_wp = t_wp / t_wp[-1]

    bc_type = "periodic" if closed and n_wp >= 4 else "not-a-knot"

    # Fit each axis
    cs_x = CubicSpline(t_wp, waypoints[:, 0], bc_type=bc_type)
    cs_y = CubicSpline(t_wp, waypoints[:, 1], bc_type=bc_type)
    cs_z = CubicSpline(t_wp, waypoints[:, 2], bc_type=bc_type)

    # Sample
    t_eval = np.linspace(0.0, 1.0, num_points)

    positions = np.column_stack([cs_x(t_eval), cs_y(t_eval), cs_z(t_eval)])

    # First derivative (tangent)
    dx = cs_x(t_eval, 1)
    dy = cs_y(t_eval, 1)
    dz = cs_z(t_eval, 1)
    tangents = np.column_stack([dx, dy, dz])

    # Second derivative
    ddx = cs_x(t_eval, 2)
    ddy = cs_y(t_eval, 2)
    ddz = cs_z(t_eval, 2)

    # Curvature κ = |r' × r''| / |r'|³
    cross = np.column_stack(
        [
            dy * ddz - dz * ddy,
            dz * ddx - dx * ddz,
            dx * ddy - dy * ddx,
        ]
    )
    cross_mag = np.linalg.norm(cross, axis=1)
    tangent_mag = np.linalg.norm(tangents, axis=1)
    tangent_mag_cubed = tangent_mag**3
    curvatures = np.where(tangent_mag_cubed > 1e-12, cross_mag / tangent_mag_cubed, 0.0)

    return positions, tangents, curvatures


def _curvature_speed_profile(
    curvatures: np.ndarray,
    config: PlannerConfig,
) -> np.ndarray:
    """Compute speed limits based on curvature.

    Uses the relation: v = min(v_max, sqrt(a_lat_max / κ))
    where a_lat_max is the maximum lateral acceleration the drone can sustain.

    Parameters
    ----------
    curvatures : np.ndarray
        (N,) curvature values (1/m).
    config : PlannerConfig
        Contains ``max_velocity_mps``.

    Returns
    -------
    np.ndarray
        (N,) speed profile (m/s).
    """
    # Lateral acceleration budget: ~1.5 g for aggressive racing
    a_lat_max = 1.5 * 9.81  # m/s²

    # Avoid division by zero for straight segments
    safe_curvature = np.maximum(curvatures, 1e-6)

    # v = sqrt(a_lat / κ), clamped to max velocity
    curvature_speed = np.sqrt(a_lat_max / safe_curvature)
    speeds = np.minimum(curvature_speed, config.max_velocity_mps)

    # Smooth the speed profile to avoid discontinuities
    # Simple moving average (window ~5% of points)
    window = max(3, len(speeds) // 20)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.pad(speeds, (window // 2, window // 2), mode="edge")
    speeds_smooth = np.convolve(padded, kernel, mode="valid")

    # Re-clamp after smoothing
    speeds_smooth = np.clip(speeds_smooth, 1.0, config.max_velocity_mps)

    return speeds_smooth[: len(curvatures)]
