#!/usr/bin/env python3
"""Visualize a flight log from the mock simulator.

Loads a .npz flight log and produces:
  1. 3D trajectory plot (ESKF estimate vs ground truth, gate positions)
  2. XY top-down view with gate markers and passage events
  3. Position error over time (ESKF vs ground truth)
  4. Control commands over time (throttle, rates)
  5. Altitude and speed profiles

Usage::

    uv run python scripts/visualize_flight.py                      # latest log
    uv run python scripts/visualize_flight.py logs/flight_xxx.npz  # specific log
    uv run python scripts/visualize_flight.py --run                 # run sim then visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_flight_log(path: Path) -> dict[str, np.ndarray]:
    """Load a .npz flight log.

    Parameters
    ----------
    path : Path
        Path to the .npz file.

    Returns
    -------
    dict[str, np.ndarray]
        All arrays from the log.
    """
    data = dict(np.load(str(path), allow_pickle=True))
    return data


def find_latest_log() -> Path:
    """Find the most recently modified flight log.

    Returns
    -------
    Path
        Path to the latest .npz log file.
    """
    log_dir = Path("logs")
    logs = sorted(log_dir.glob("flight_*.npz"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("No flight logs found in logs/. Run the sim first:")
        print("  uv run python scripts/run_sim.py")
        sys.exit(1)
    return logs[-1]


def parse_state_data(timestamps_ns: np.ndarray, data: np.ndarray) -> dict[str, np.ndarray]:
    """Parse state data array into named components.

    State vector: [pos(3), vel(3), ori(4), accel_bias(3), gyro_bias(3)] = 16

    Parameters
    ----------
    timestamps_ns : np.ndarray
        (N,) timestamps.
    data : np.ndarray
        (N, 16) state vectors.

    Returns
    -------
    dict[str, np.ndarray]
        Named arrays.
    """
    t = (timestamps_ns - timestamps_ns[0]) * 1e-9  # seconds from start
    return {
        "t": t,
        "pos": data[:, 0:3],
        "vel": data[:, 3:6],
        "ori": data[:, 6:10],
    }


def parse_gt_data(timestamps_ns: np.ndarray, data: np.ndarray) -> dict[str, np.ndarray]:
    """Parse ground truth data array.

    GT vector: [pos(3), vel(3), ori(4)] = 10

    Parameters
    ----------
    timestamps_ns : np.ndarray
        (N,) timestamps.
    data : np.ndarray
        (N, 10) state vectors.

    Returns
    -------
    dict[str, np.ndarray]
        Named arrays.
    """
    t = (timestamps_ns - timestamps_ns[0]) * 1e-9
    return {
        "t": t,
        "pos": data[:, 0:3],
        "vel": data[:, 3:6],
        "ori": data[:, 6:10],
    }


def parse_command_data(timestamps_ns: np.ndarray, data: np.ndarray) -> dict[str, np.ndarray]:
    """Parse command data array.

    Command vector: [throttle, roll_rate, pitch_rate, yaw_rate] = 4

    Parameters
    ----------
    timestamps_ns : np.ndarray
        (N,) timestamps.
    data : np.ndarray
        (N, 4) command vectors.

    Returns
    -------
    dict[str, np.ndarray]
        Named arrays.
    """
    t = (timestamps_ns - timestamps_ns[0]) * 1e-9
    return {
        "t": t,
        "throttle": data[:, 0],
        "roll_rate": data[:, 1],
        "pitch_rate": data[:, 2],
        "yaw_rate": data[:, 3],
    }


def get_gate_positions() -> np.ndarray:
    """Get the standard 4-gate circuit positions.

    Returns
    -------
    np.ndarray
        (4, 3) gate positions.
    """
    return np.array(
        [
            [15.0, 0.0, 3.0],
            [15.0, 15.0, 3.0],
            [0.0, 15.0, 3.0],
            [0.0, 0.0, 3.0],
        ]
    )


def speed_coloured_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    cmap: str = "viridis",
    linewidth: float = 2.0,
) -> LineCollection:
    """Draw a line coloured by speed.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    x, y : np.ndarray
        (N,) coordinates.
    speed : np.ndarray
        (N,) speed values for colouring.
    cmap : str
        Colormap name.
    linewidth : float
        Line width.

    Returns
    -------
    LineCollection
        The drawn collection.
    """
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)
    return lc


def visualize(log_path: Path) -> None:
    """Create visualization plots from a flight log.

    Parameters
    ----------
    log_path : Path
        Path to the .npz flight log.
    """
    print(f"Loading {log_path}...")
    data = load_flight_log(log_path)

    # Parse data
    has_states = len(data.get("state_timestamps_ns", [])) > 0
    has_gt = len(data.get("ground_truth_timestamps_ns", [])) > 0
    has_cmds = len(data.get("command_timestamps_ns", [])) > 0
    has_gates = "gate_passage_timestamps_ns" in data

    if not has_states:
        print("No state data in log — nothing to visualize.")
        return

    est = parse_state_data(data["state_timestamps_ns"], data["state_data"])

    gt = None
    if has_gt:
        gt = parse_gt_data(data["ground_truth_timestamps_ns"], data["ground_truth_data"])

    cmds = None
    if has_cmds:
        cmds = parse_command_data(data["command_timestamps_ns"], data["command_data"])

    gate_positions = get_gate_positions()

    # Compute derived quantities
    est_speed = np.linalg.norm(est["vel"], axis=1)
    gt_speed = np.linalg.norm(gt["vel"], axis=1) if gt else None

    # --- Figure layout ---
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Flight: {log_path.name}", fontsize=14, fontweight="bold")

    # 1. Top-down XY view (large, left)
    ax_xy = fig.add_subplot(2, 3, (1, 4))
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal")

    # Ground truth trajectory
    if gt is not None:
        ax_xy.plot(
            gt["pos"][:, 0], gt["pos"][:, 1], "k-", alpha=0.3, linewidth=1, label="Ground truth"
        )

    # ESKF estimate coloured by speed
    lc = speed_coloured_line(ax_xy, est["pos"][:, 0], est["pos"][:, 1], est_speed)
    fig.colorbar(lc, ax=ax_xy, label="Speed (m/s)", shrink=0.6)

    # Start/end markers
    ax_xy.plot(est["pos"][0, 0], est["pos"][0, 1], "go", markersize=10, label="Start", zorder=5)
    ax_xy.plot(est["pos"][-1, 0], est["pos"][-1, 1], "rs", markersize=10, label="End", zorder=5)

    # Gate positions
    for i, gp in enumerate(gate_positions):
        ax_xy.plot(gp[0], gp[1], "D", color="orange", markersize=12, zorder=5)
        ax_xy.annotate(
            f"G{i}",
            (gp[0], gp[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
            fontweight="bold",
            color="orange",
        )

    # Gate passage events
    if has_gates:
        gp_ts = data["gate_passage_timestamps_ns"]
        t0 = data["state_timestamps_ns"][0]
        for ts in gp_ts:
            # Find closest state index
            idx = np.argmin(np.abs(data["state_timestamps_ns"] - ts))
            pos = data["state_data"][idx, 0:3]
            ax_xy.plot(pos[0], pos[1], "*", color="red", markersize=15, zorder=6)

    ax_xy.autoscale_view()
    ax_xy.legend(loc="upper left", fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    # 2. 3D trajectory
    ax_3d = fig.add_subplot(2, 3, 2, projection="3d")
    ax_3d.set_title("3D Trajectory")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")

    ax_3d.plot(
        est["pos"][:, 0], est["pos"][:, 1], est["pos"][:, 2], "b-", linewidth=1.5, label="ESKF"
    )
    if gt is not None:
        ax_3d.plot(
            gt["pos"][:, 0],
            gt["pos"][:, 1],
            gt["pos"][:, 2],
            "k--",
            alpha=0.4,
            linewidth=1,
            label="Ground truth",
        )

    for i, gp in enumerate(gate_positions):
        ax_3d.scatter(*gp, color="orange", s=100, marker="D", zorder=5)
        ax_3d.text(gp[0], gp[1], gp[2] + 0.5, f"G{i}", fontsize=9, color="orange")

    ax_3d.legend(fontsize=8)

    # 3. Position error over time
    ax_err = fig.add_subplot(2, 3, 3)
    ax_err.set_title("Position Error (ESKF vs Ground Truth)")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Error (m)")

    if gt is not None:
        # Interpolate GT to ESKF timestamps for comparison
        from scipy.interpolate import interp1d

        gt_interp = np.zeros_like(est["pos"])
        for dim in range(3):
            f = interp1d(
                gt["t"],
                gt["pos"][:, dim],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            gt_interp[:, dim] = f(est["t"])

        pos_err = np.linalg.norm(est["pos"] - gt_interp, axis=1)
        ax_err.plot(est["t"], pos_err, "r-", linewidth=1)
        ax_err.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="1m")
        ax_err.set_ylim(bottom=0)
        ax_err.legend(fontsize=8)

        # Per-axis error
        labels = ["X", "Y", "Z"]
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        for dim in range(3):
            err = est["pos"][:, dim] - gt_interp[:, dim]
            ax_err.plot(
                est["t"],
                err,
                color=colors[dim],
                alpha=0.4,
                linewidth=0.8,
                label=f"{labels[dim]} err",
            )
        ax_err.legend(fontsize=7)
    else:
        ax_err.text(
            0.5,
            0.5,
            "No ground truth",
            transform=ax_err.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    ax_err.grid(True, alpha=0.3)

    # 4. Altitude and speed
    ax_alt = fig.add_subplot(2, 3, 5)
    ax_alt.set_title("Altitude & Speed")
    ax_alt.set_xlabel("Time (s)")

    ax_alt.plot(est["t"], est["pos"][:, 2], "b-", linewidth=1.5, label="Altitude (m)")
    if gt is not None:
        ax_alt.plot(gt["t"], gt["pos"][:, 2], "b--", alpha=0.4, linewidth=1, label="GT altitude")
    ax_alt.set_ylabel("Altitude (m)", color="blue")
    ax_alt.tick_params(axis="y", labelcolor="blue")

    ax_spd = ax_alt.twinx()
    ax_spd.plot(est["t"], est_speed, "r-", linewidth=1, alpha=0.7, label="Speed (m/s)")
    if gt_speed is not None:
        ax_spd.plot(gt["t"], gt_speed, "r--", alpha=0.3, linewidth=1)
    ax_spd.set_ylabel("Speed (m/s)", color="red")
    ax_spd.tick_params(axis="y", labelcolor="red")

    # Gate passage time markers
    if has_gates:
        t0 = data["state_timestamps_ns"][0]
        for ts in data["gate_passage_timestamps_ns"]:
            t_s = (ts - t0) * 1e-9
            ax_alt.axvline(t_s, color="orange", alpha=0.5, linestyle=":", linewidth=1)

    ax_alt.grid(True, alpha=0.3)

    # 5. Control commands
    ax_cmd = fig.add_subplot(2, 3, 6)
    ax_cmd.set_title("Control Commands")
    ax_cmd.set_xlabel("Time (s)")

    if cmds is not None:
        ax_cmd.plot(cmds["t"], cmds["throttle"], label="Throttle", linewidth=1)
        ax_cmd.plot(cmds["t"], cmds["roll_rate"], label="Roll rate", linewidth=0.8, alpha=0.7)
        ax_cmd.plot(cmds["t"], cmds["pitch_rate"], label="Pitch rate", linewidth=0.8, alpha=0.7)
        ax_cmd.plot(cmds["t"], cmds["yaw_rate"], label="Yaw rate", linewidth=0.8, alpha=0.7)
        ax_cmd.legend(fontsize=7)
    else:
        ax_cmd.text(
            0.5,
            0.5,
            "No command data",
            transform=ax_cmd.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    ax_cmd.grid(True, alpha=0.3)

    # --- Stats text ---
    duration = est["t"][-1]
    max_speed = float(np.max(est_speed))
    mean_speed = float(np.mean(est_speed))
    gates_passed = len(data.get("gate_passage_ids", []))

    stats_text = (
        f"Duration: {duration:.1f}s | "
        f"Gates: {gates_passed} | "
        f"Speed: {mean_speed:.1f} avg / {max_speed:.1f} max m/s"
    )
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.show()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize flight log")
    parser.add_argument(
        "log_path", nargs="?", default=None, help="Path to .npz flight log (default: latest)"
    )
    parser.add_argument("--run", action="store_true", help="Run simulation first, then visualize")
    parser.add_argument(
        "--max-time", type=float, default=60.0, help="Max sim time if --run (seconds)"
    )
    args = parser.parse_args()

    if args.run:
        from drone_racing.utils.config import load_config
        from scripts.run_sim import run_pipeline

        config = load_config()
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        run_pipeline(config, max_time=args.max_time, log_path="logs/viz_latest.npz")
        log_path = Path("logs/viz_latest.npz")
    elif args.log_path:
        log_path = Path(args.log_path)
    else:
        log_path = find_latest_log()

    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    visualize(log_path)


if __name__ == "__main__":
    main()
