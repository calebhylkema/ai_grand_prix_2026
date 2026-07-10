#!/usr/bin/env python3
"""Sweep MPCC solver parameters to find the best latency/quality tradeoff.

The pipeline benchmark identified MPCC as the latency bottleneck (p95 ~98ms
vs 30ms target).  This script tries several horizon lengths and IPOPT
settings and reports solve time + success rate.

Usage::

    uv run aigp-benchmark mpcc
    uv run aigp-benchmark mpcc --iterations 50

Each variant is run for ``iterations`` warm solves against a shifting state
trajectory (to simulate real operation with warm-starting).
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from drone_racing.planning.mpcc import MPCCConfig, MPCCSolver
from drone_racing.planning.spline import RacingLine, build_racing_line
from drone_racing.utils.config import load_config
from drone_racing.utils.types import GateWaypoint, PlannerConfig

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    name: str
    horizon: int
    dt: float
    solver_opts: dict[str, Any]


def _build_circuit_and_line() -> RacingLine:
    config = load_config()
    corners_local = config.gate.outer_corners_3d_array
    gate_defs = [
        (np.array([15.0, 0.0, 3.0]), 0.0),
        (np.array([15.0, 15.0, 3.0]), np.pi / 2),
        (np.array([0.0, 15.0, 3.0]), np.pi),
        (np.array([0.0, 0.0, 3.0]), -np.pi / 2),
    ]
    gates: list[GateWaypoint] = []
    for i, (pos, yaw) in enumerate(gate_defs):
        ori = Rotation.from_euler("yz", [np.pi / 2, yaw]).as_quat()
        r_wg = Rotation.from_quat(ori)
        corners_w = (r_wg.apply(corners_local) + pos).astype(np.float64)
        gates.append(
            GateWaypoint(
                gate_id=i,
                position_w=pos,
                orientation_w=ori.astype(np.float64),
                corners_3d_w=corners_w,
                order=i,
            )
        )
    gate_positions = np.array([g.position_w for g in gates])
    gate_orientations = np.array([g.orientation_w for g in gates])
    racing_line = build_racing_line(
        gate_positions, gate_orientations, PlannerConfig(), num_points=200, closed=True
    )
    return racing_line


def _make_variants() -> list[Variant]:
    """Define the sweep configurations.

    Baseline matches drone_racing/planning/mpcc.py defaults.
    """
    base_opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 50,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 1e-2,
        "ipopt.acceptable_iter": 3,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.mu_init": 1e-3,
        "print_time": 0,
    }

    def with_opts(**overrides: Any) -> dict[str, Any]:
        d = dict(base_opts)
        d.update(overrides)
        return d

    return [
        Variant("baseline N=20", horizon=20, dt=0.05, solver_opts=with_opts()),
        Variant("short N=15", horizon=15, dt=0.05, solver_opts=with_opts()),
        Variant("short N=12", horizon=12, dt=0.05, solver_opts=with_opts()),
        Variant("short N=10", horizon=10, dt=0.05, solver_opts=with_opts()),
        Variant(
            "N=15 max_iter=30",
            horizon=15,
            dt=0.05,
            solver_opts=with_opts(**{"ipopt.max_iter": 30}),
        ),
        Variant(
            "N=15 max_iter=20 loose",
            horizon=15,
            dt=0.05,
            solver_opts=with_opts(
                **{
                    "ipopt.max_iter": 20,
                    "ipopt.tol": 5e-3,
                    "ipopt.acceptable_tol": 5e-2,
                    "ipopt.acceptable_iter": 2,
                }
            ),
        ),
        Variant(
            "N=10 max_iter=20 loose",
            horizon=10,
            dt=0.05,
            solver_opts=with_opts(
                **{
                    "ipopt.max_iter": 20,
                    "ipopt.tol": 5e-3,
                    "ipopt.acceptable_tol": 5e-2,
                    "ipopt.acceptable_iter": 2,
                }
            ),
        ),
        Variant(
            "N=20 max_iter=20 loose",
            horizon=20,
            dt=0.05,
            solver_opts=with_opts(
                **{
                    "ipopt.max_iter": 20,
                    "ipopt.tol": 5e-3,
                    "ipopt.acceptable_tol": 5e-2,
                    "ipopt.acceptable_iter": 2,
                }
            ),
        ),
        Variant(
            "N=15 dt=0.08 (same lookahead)",
            horizon=15,
            dt=0.08,
            solver_opts=with_opts(**{"ipopt.max_iter": 30}),
        ),
    ]


def _make_state(x: float = 5.0, v: float = 5.0) -> np.ndarray:
    """Build a 9-D MPCC state vector [px, py, pz, vx, vy, vz, phi, theta, psi]."""
    return np.array([x, 0.0, 3.0, v, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def run_variant(racing_line: RacingLine, variant: Variant, n_iter: int) -> dict[str, Any]:
    """Run one variant and collect timing + success stats."""
    cfg = MPCCConfig(horizon=variant.horizon, dt=variant.dt, solver_opts=variant.solver_opts)
    solver = MPCCSolver(racing_line, config=cfg, closed=True)

    # Warm the solver (first solve is always slow — cold NLP)
    state = _make_state(x=5.0, v=5.0)
    progress = racing_line.closest_progress(state[:3])
    for _ in range(3):
        solver.solve(state, progress)

    # Timed solves — shift state forward to exercise warm-starting realistically
    times = []
    successes = 0
    x = 5.0
    for _ in range(n_iter):
        state = _make_state(x=x, v=5.0)
        progress = racing_line.closest_progress(state[:3])
        sol = solver.solve(state, progress)
        times.append(sol.solve_time_ms)
        if sol.success:
            successes += 1
        # Advance state ~0.05s * 5m/s = 0.25m per call
        x += 0.25
        if x > 14.0:
            x = 5.0

    return {
        "times": times,
        "success_rate": successes / n_iter,
        "n_vars": solver._n_w,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MPCC solver parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=30, help="Solves per variant")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Building racing line...")
    racing_line = _build_circuit_and_line()

    print(f"\nRunning {args.iterations} iterations per variant...\n")

    results = []
    for variant in _make_variants():
        print(f"  {variant.name}...", flush=True)
        r = run_variant(racing_line, variant, args.iterations)
        results.append((variant, r))

    print("\n" + "=" * 105)
    print("MPCC Parameter Sweep Results")
    print("=" * 105)
    header = f"{'Variant':35s}  {'Nvars':>6s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}  {'success':>8s}"
    print(header)
    print("-" * 105)
    for variant, r in results:
        times = r["times"]
        p50 = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        mx = max(times)
        print(
            f"{variant.name:35s}  {r['n_vars']:>6d}  "
            f"{p50:>7.1f}ms  {p95:>7.1f}ms  {p99:>7.1f}ms  {mx:>7.1f}ms  "
            f"{r['success_rate'] * 100:>6.0f}%"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
