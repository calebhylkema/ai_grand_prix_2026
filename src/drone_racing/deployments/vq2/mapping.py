"""Course-map miner: reconstruct gate geometry from saved fly_trace frames.

The strategic pivot (2026-07-09, user-approved): stop flying the course from
instantaneous detections and reactive branches — the course is FIXED and attempts are
unlimited, so build a map once and fly the map. The AX-batch forensics showed the
reactive controller latching three DIFFERENT gates (PnP z 27.2 -> 21.2 -> 17.1 m in
0.3 s, an impossible 35 m/s "closure") because nothing in the stack knows where the
gates actually are.

Method: any frame with TWO PnP-valid gates gives their relative position directly
(no odometry, no velocity estimate needed):

    d_cam  = pose_j - pose_i                      # camera frame
    d_body = CAM_TO_BODY @ d_cam                  # FRD body frame
    d_lvl  = Ry(pitch) @ Rx(roll) @ d_body        # gravity-leveled (yaw-free up/fwd)

The leveled vertical component (up = -z in FRD/NED) and horizontal distance are
yaw-invariant; the horizontal direction is expressed in the drone's heading frame,
which is consistent across runs on the same course segment (every run flies pad ->
gate 0 on nearly the same heading). Single-gate sightings are also dumped (with the
active_gate context from the trace filename) for identity/cluster analysis.

Usage:
    uv run python -m drone_racing.deployments.vq2.mapping [trace_dir ...]
    uv run python -m drone_racing.deployments.vq2.mapping --estimate-pair [trace_dir ...]
        (default: every prev_* under .local/datasets/vq2/fly_trace)

Writes .local/datasets/vq2/map/observations.csv and prints the co-visible pair table.
``--estimate-pair`` is opt-in because mixing different physical gate pairs would
create a plausible-looking but meaningless transform.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drone_racing.deployments.vq2 import perception as gates
from drone_racing.utils.workspace import Workspace

_WS = Workspace.locate()
TRACE_ROOT = _WS.datasets / "vq2" / "fly_trace"
OUT_DIR = _WS.datasets / "vq2" / "map"

# trace filename telemetry: NNN_tX.XX_r{roll}p{pitch}_vz{vz}_gz{gz}_g{active_gate}_...
_NAME_RE = re.compile(
    r"^(\d+)_t([\d.]+)_r([+-]?\d+)p([+-]?\d+)_vz([+-]?[\d.]+)_gz([+-]?[\d.]+)_g(-?\d+)_"
)


@dataclass(frozen=True, slots=True)
class PairDeltaEstimate:
    """Robust run-independent estimate of a co-visible gate-pair transform.

    ``delta_ned_m`` follows the map frame: forward, right, down.  Repeated frames
    from one run are collapsed before cross-run statistics so they do not inflate
    the independence count.
    """

    delta_ned_m: tuple[float, float, float]
    position_std_m: tuple[float, float, float]
    source_runs: tuple[str, ...]
    independent_runs: int
    observation_count: int
    flight_evidence_ready: bool
    readiness_reason: str


def _finite_number(value: object) -> float:
    out = float(value)  # type: ignore[arg-type]
    if not math.isfinite(out):
        raise ValueError(f"non-finite map observation value {value!r}")
    return out


def estimate_pair_delta(
    pairs: list[dict[str, object]],
    *,
    perception_revision: str,
    min_independent_runs: int = 3,
) -> PairDeltaEstimate:
    """Estimate one near-to-far gate transform without frame-count leakage.

    Parameters
    ----------
    pairs : list[dict[str, object]]
        Rows from :func:`pair_table` for a single physical gate pair.
    perception_revision : str
        Detector/PnP revision that produced the rows. Obsolete or unknown revisions
        are never marked flight-ready.
    min_independent_runs : int
        Minimum distinct runs required for evidence readiness.

    Returns
    -------
    PairDeltaEstimate
        Median transform and conservative cross-run dispersion.
    """
    if not pairs:
        raise ValueError("cannot estimate a gate-pair transform from zero observations")
    by_run: dict[str, list[tuple[float, float, float]]] = {}
    for pair in pairs:
        run = str(pair["run"])
        # mapping.py stores up-positive; the course-map frame is NED (down-positive).
        delta = (
            _finite_number(pair["d_fwd"]),
            _finite_number(pair["d_right"]),
            -_finite_number(pair["d_up"]),
        )
        by_run.setdefault(run, []).append(delta)

    # A run is one independent experiment regardless of how many adjacent frames
    # contain the same pair. Collapse it before estimating cross-run uncertainty.
    run_values = np.asarray(
        [np.median(np.asarray(values, dtype=np.float64), axis=0) for values in by_run.values()]
    )
    center = np.median(run_values, axis=0)
    if len(run_values) > 1:
        # Sample spread is deliberately conservative for today's very small N. Once
        # larger corpora exist, it also guards against overconfident tiny MAD values.
        sample_std = np.std(run_values, axis=0, ddof=1)
        mad_std = 1.4826 * np.median(np.abs(run_values - center), axis=0)
        spread = np.maximum(sample_std, mad_std)
    else:
        spread = np.full(3, np.inf)

    revision = perception_revision.lower()
    revision_ok = not any(tag in revision for tag in ("legacy", "inverted", "unknown"))
    count_ok = len(run_values) >= min_independent_runs
    ready = bool(revision_ok and count_ok and np.all(np.isfinite(spread)))
    reasons: list[str] = []
    if not count_ok:
        reasons.append(f"need {min_independent_runs} independent runs, have {len(run_values)}")
    if not revision_ok:
        reasons.append(f"obsolete/unknown perception revision {perception_revision!r}")
    if not np.all(np.isfinite(spread)):
        reasons.append("cross-run uncertainty is not finite")

    return PairDeltaEstimate(
        delta_ned_m=tuple(float(x) for x in center),  # type: ignore[arg-type]
        position_std_m=tuple(float(x) for x in spread),  # type: ignore[arg-type]
        source_runs=tuple(sorted(by_run)),
        independent_runs=len(run_values),
        observation_count=len(pairs),
        flight_evidence_ready=ready,
        readiness_reason="ready" if ready else "; ".join(reasons),
    )


def _level_rot(roll: float, pitch: float) -> np.ndarray:
    """Body->leveled-world rotation (ZYX with yaw=0), radians. Matches estimator ZYX."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx


def scan(dirs: list[Path]) -> list[dict[str, object]]:
    """All PnP-valid gate observations (one row per gate per frame), leveled."""
    rows: list[dict[str, object]] = []
    for d in dirs:
        for f in sorted(d.glob("*.png")):
            m = _NAME_RE.match(f.name)
            if m is None:
                continue
            t_rel = float(m.group(2))
            roll = math.radians(float(m.group(3)))
            pitch = math.radians(float(m.group(4)))
            active_gate = int(m.group(7))
            img = cv2.imread(str(f))
            if img is None:
                continue
            r_lvl = _level_rot(roll, pitch)
            for g in gates.detect(img):
                if g.pose_cam is None:
                    continue
                p_lvl = r_lvl @ (gates.CAM_TO_BODY @ np.array(g.pose_cam))
                rows.append(
                    {
                        "run": d.name,
                        "frame": f.name.split("_", 1)[0],
                        "t": t_rel,
                        "active_gate": active_gate,
                        "area": g.area,
                        "cy": round(g.cy, 1),
                        "z_cam": round(g.pose_cam[2], 2),
                        "bearing_deg": round(math.degrees(g.bearing), 1),
                        "fwd": round(float(p_lvl[0]), 2),
                        "right": round(float(p_lvl[1]), 2),
                        "up": round(float(-p_lvl[2]), 2),  # FRD: up = -z
                    }
                )
    return rows


def pair_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Co-visible pair deltas (near gate -> far gate), the yaw-free map constraints."""
    pairs: list[dict[str, object]] = []
    by_frame: dict[tuple[object, object], list[dict[str, object]]] = {}
    for r in rows:
        by_frame.setdefault((r["run"], r["frame"]), []).append(r)
    for (run, frame), obs in by_frame.items():
        if len(obs) < 2:
            continue
        obs = sorted(obs, key=lambda o: float(o["z_cam"]))  # type: ignore[arg-type]
        for i in range(len(obs) - 1):
            a, b = obs[i], obs[i + 1]
            dfwd = float(b["fwd"]) - float(a["fwd"])  # type: ignore[arg-type]
            drgt = float(b["right"]) - float(a["right"])  # type: ignore[arg-type]
            dup = float(b["up"]) - float(a["up"])  # type: ignore[arg-type]
            pairs.append(
                {
                    "run": run,
                    "frame": frame,
                    "t": a["t"],
                    "active_gate": a["active_gate"],
                    "z_near": a["z_cam"],
                    "z_far": b["z_cam"],
                    "d_fwd": round(dfwd, 2),
                    "d_right": round(drgt, 2),
                    "d_up": round(dup, 2),
                    "d_horiz": round(math.hypot(dfwd, drgt), 2),
                }
            )
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Mine gate geometry from saved fly_trace frames.")
    ap.add_argument(
        "trace_dirs",
        nargs="*",
        type=Path,
        help=f"trace archive dirs (default: every prev_* under {TRACE_ROOT})",
    )
    ap.add_argument(
        "--estimate-pair",
        action="store_true",
        help=(
            "estimate one physical near/far pair from the selected dirs; caller must "
            "first verify all co-visible rows describe the same ordered gate pair"
        ),
    )
    args = ap.parse_args(argv)
    dirs = args.trace_dirs if args.trace_dirs else sorted(TRACE_ROOT.glob("prev_*"))
    rows = scan(dirs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUT_DIR / "observations.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} posed-gate observations from {len(dirs)} runs")
    print("\n--- single sightings (identity clues: t + active_gate + z) ---")
    for r in rows:
        print(
            f"  {r['run']}/{r['frame']} t{r['t']:<5} g{r['active_gate']} "
            f"a{r['area']:<6.0f} z{r['z_cam']:<6} b{r['bearing_deg']:+6.1f}deg "
            f"fwd{r['fwd']:+7.2f} right{r['right']:+7.2f} up{r['up']:+6.2f}"
        )
    pairs = pair_table(rows)
    print(f"\n--- co-visible pairs ({len(pairs)}): near gate -> far gate ---")
    for p in pairs:
        print(
            f"  {p['run']}/{p['frame']} t{p['t']:<5} g{p['active_gate']} "
            f"z {p['z_near']}->{p['z_far']}: dfwd {p['d_fwd']:+6.2f} "
            f"dright {p['d_right']:+6.2f} dup {p['d_up']:+6.2f} (horiz {p['d_horiz']})"
        )
    if pairs and args.estimate_pair:
        estimate = estimate_pair_delta(
            pairs,
            perception_revision="legacy-pre-structured-unknown-tilt",
        )
        print("\n--- pair estimate (offline evidence only; forward/right/down NED) ---")
        print(
            f"  delta={estimate.delta_ned_m} std={estimate.position_std_m} "
            f"runs={estimate.independent_runs} observations={estimate.observation_count}"
        )
        print(
            f"  flight_evidence_ready={estimate.flight_evidence_ready}: {estimate.readiness_reason}"
        )


if __name__ == "__main__":
    main()
