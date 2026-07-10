"""Flight telemetry logger — records pipeline data for replay and analysis.

Stores timestamped records of state estimates, commands, detections, and
ground truth into a structured numpy archive (.npz) for post-flight analysis.

Usage::

    logger = FlightLogger()
    # ... during flight loop ...
    logger.log_state(estimated_state)
    logger.log_command(command)
    logger.log_ground_truth(gt_state)
    # ... after flight ...
    logger.save("logs/flight_001.npz")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from drone_racing.utils.types import DroneCommand, State

logger = logging.getLogger(__name__)


@dataclass
class _TimeseriesBuffer:
    """Growable buffer for a named timeseries."""

    timestamps_ns: list[int] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)

    def append(self, ts: int, values: np.ndarray) -> None:
        """Append a sample."""
        self.timestamps_ns.append(ts)
        self.data.append(values)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (timestamps (N,), data (N, D)).
        """
        if len(self.timestamps_ns) == 0:
            return np.array([], dtype=np.int64), np.array([])
        return np.array(self.timestamps_ns, dtype=np.int64), np.array(self.data)


class FlightLogger:
    """Records flight data for post-flight analysis.

    Parameters
    ----------
    max_samples : int
        Maximum number of samples per channel before oldest are dropped.
        Set to 0 for unlimited.
    """

    def __init__(self, max_samples: int = 0) -> None:
        self._max = max_samples
        self._states = _TimeseriesBuffer()
        self._commands = _TimeseriesBuffer()
        self._ground_truth = _TimeseriesBuffer()
        self._gate_passages: list[dict[str, object]] = []
        self._metadata: dict[str, object] = {}

    def set_metadata(self, key: str, value: object) -> None:
        """Store arbitrary metadata (config values, run ID, etc.).

        Parameters
        ----------
        key : str
            Metadata key.
        value : object
            Metadata value (must be serialisable by numpy).
        """
        self._metadata[key] = value

    def log_state(self, state: State) -> None:
        """Record an ESKF state estimate.

        Parameters
        ----------
        state : State
            Current state estimate.
        """
        vec = np.concatenate(
            [
                state.position,
                state.velocity,
                state.orientation,
                state.accel_bias,
                state.gyro_bias,
            ]
        )
        self._states.append(state.timestamp_ns, vec)

    def log_command(self, cmd: DroneCommand) -> None:
        """Record a control command.

        Parameters
        ----------
        cmd : DroneCommand
            Command sent to the flight controller.
        """
        vec = np.array([cmd.throttle, cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate])
        self._commands.append(cmd.timestamp_ns, vec)

    def log_ground_truth(self, state: State) -> None:
        """Record a ground-truth state (from simulator).

        Parameters
        ----------
        state : State
            Ground-truth state.
        """
        vec = np.concatenate(
            [
                state.position,
                state.velocity,
                state.orientation,
            ]
        )
        self._ground_truth.append(state.timestamp_ns, vec)

    def log_gate_passage(self, timestamp_ns: int, gate_id: int, total_passed: int) -> None:
        """Record a gate passage event.

        Parameters
        ----------
        timestamp_ns : int
            When the gate was passed.
        gate_id : int
            Which gate was passed.
        total_passed : int
            Running total of gates passed.
        """
        self._gate_passages.append(
            {
                "timestamp_ns": timestamp_ns,
                "gate_id": gate_id,
                "total_passed": total_passed,
            }
        )

    def save(self, path: str | Path) -> Path:
        """Save all recorded data to a compressed numpy archive.

        Parameters
        ----------
        path : str | Path
            Output file path (should end in .npz).

        Returns
        -------
        Path
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        state_ts, state_data = self._states.to_arrays()
        cmd_ts, cmd_data = self._commands.to_arrays()
        gt_ts, gt_data = self._ground_truth.to_arrays()

        save_dict: dict[str, Any] = {
            "state_timestamps_ns": state_ts,
            "state_data": state_data,
            "command_timestamps_ns": cmd_ts,
            "command_data": cmd_data,
            "ground_truth_timestamps_ns": gt_ts,
            "ground_truth_data": gt_data,
        }

        # Gate passages
        if self._gate_passages:
            gp = self._gate_passages
            save_dict["gate_passage_timestamps_ns"] = np.array(
                [g["timestamp_ns"] for g in gp], dtype=np.int64
            )
            save_dict["gate_passage_ids"] = np.array([g["gate_id"] for g in gp], dtype=np.int32)

        np.savez_compressed(str(out), **save_dict)

        n_states = len(state_ts)
        n_cmds = len(cmd_ts)
        logger.info("Flight log saved: %s (%d states, %d cmds)", out, n_states, n_cmds)
        return out.resolve()

    @property
    def num_state_samples(self) -> int:
        """Number of state estimates recorded."""
        return len(self._states.timestamps_ns)

    @property
    def num_command_samples(self) -> int:
        """Number of commands recorded."""
        return len(self._commands.timestamps_ns)
