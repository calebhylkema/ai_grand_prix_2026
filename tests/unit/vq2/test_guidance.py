"""Offline validation of the geometric guidance controller (no sim, no ground-truth
needed). Simulate a rigid-body quadrotor driven by guidance.geometric_control from
several start offsets and confirm it converges to the gate (position -> ~0) and
arrives with low speed, without blowing up or over-tilting. Validates the control
MATH before trusting it on the real sim (same approach that de-risked the ESKF).

Conventions: world NED, body FRD (thrust along body -z), gravity +Z down.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.deployments.vq2.attitude import quat_mul, quat_to_matrix
from drone_racing.deployments.vq2.eskf import quat_from_rotvec
from drone_racing.deployments.vq2.guidance import G, geometric_control


def simulate(p0: np.ndarray, v0: np.ndarray, secs: float = 8.0, hover: float = 0.30) -> dict:
    dt = 0.01
    p = np.asarray(p0, float).copy()
    v = np.asarray(v0, float).copy()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    # start with a small random tilt so it must correct attitude too
    q = quat_mul(q, quat_from_rotvec(np.array([0.1, -0.15, 0.0])))
    max_tilt_seen = 0.0
    for _ in range(int(secs / dt)):
        r = quat_to_matrix(q)
        p_rel = -p  # gate at world origin
        coll, omega, _ = geometric_control(p_rel, v, r, hover_thrust=hover)
        thrust_acc = (coll / hover) * G
        a = r @ np.array([0.0, 0.0, -thrust_acc]) + np.array([0.0, 0.0, G])
        v = v + a * dt
        p = p + v * dt
        q = quat_mul(q, quat_from_rotvec(omega * dt))
        q = q / np.linalg.norm(q)
        tilt = np.arccos(max(-1.0, min(1.0, float(r[2, 2]))))  # angle of body-up from world-up
        max_tilt_seen = max(max_tilt_seen, tilt)
        if np.linalg.norm(p) > 100 or not np.all(np.isfinite(p)):
            return {"diverged": True, "pos_err": 1e9, "speed": 1e9, "max_tilt": max_tilt_seen}
    return {
        "diverged": False,
        "pos_err": float(np.linalg.norm(p)),
        "speed": float(np.linalg.norm(v)),
        "max_tilt": float(np.degrees(max_tilt_seen)),
    }


@pytest.mark.parametrize(
    ("p0", "desc"),
    [
        ((-8.0, 0.0, 0.0), "straight, 8m back"),
        ((-6.0, 3.0, -1.0), "turn: 6m back, 3m right, 1m up"),
        ((-5.0, -2.5, 2.0), "turn: 5m back, 2.5m left, 2m down"),
        ((-10.0, 4.0, 1.5), "far turn: 10m back, 4m right"),
    ],
)
def test_geometric_control_converges(p0: tuple[float, float, float], desc: str) -> None:
    """From each start offset: converge to the gate, arrive slow, never over-tilt."""
    r = simulate(np.array(p0), np.zeros(3))
    assert not r["diverged"], desc
    assert r["pos_err"] < 0.5, f"{desc}: pos_err={r['pos_err']:.2f}m"
    assert r["speed"] < 1.5, f"{desc}: speed={r['speed']:.2f}m/s"
    assert r["max_tilt"] < 45, f"{desc}: max_tilt={r['max_tilt']:.0f}deg"
