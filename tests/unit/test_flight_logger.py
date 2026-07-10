"""Tests for drone_racing.logging.flight_logger — flight data recording.

Test strategy:
    1. Log states, commands, and ground truth → save → reload → verify.
    2. Empty logger produces valid (empty) arrays.
    3. Gate passage events are serialised correctly.
    4. Metadata is stored (as best-effort, not round-tripped via npz).
    5. Parent directories are created automatically.
    6. Property accessors return correct counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from drone_racing.logging.flight_logger import FlightLogger
from drone_racing.utils.types import DroneCommand, State

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(ts: int, pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> State:
    return State(
        timestamp_ns=ts,
        position=np.array(pos),
        velocity=np.zeros(3),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15) * 1e-2,
    )


def _make_command(ts: int, throttle: float = 0.5) -> DroneCommand:
    return DroneCommand(
        timestamp_ns=ts,
        throttle=throttle,
        roll_rate=0.1,
        pitch_rate=-0.1,
        yaw_rate=0.05,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlightLoggerBasic:
    """Basic logging and property access."""

    def test_empty_logger_counts(self) -> None:
        logger = FlightLogger()
        assert logger.num_state_samples == 0
        assert logger.num_command_samples == 0

    def test_log_state_increments_count(self) -> None:
        logger = FlightLogger()
        logger.log_state(_make_state(1000))
        logger.log_state(_make_state(2000))
        assert logger.num_state_samples == 2

    def test_log_command_increments_count(self) -> None:
        logger = FlightLogger()
        logger.log_command(_make_command(1000))
        assert logger.num_command_samples == 1


class TestFlightLoggerSaveLoad:
    """Save to .npz and verify contents."""

    def test_save_and_reload_states(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_state(_make_state(1_000_000, (1.0, 2.0, 3.0)))
        logger.log_state(_make_state(2_000_000, (4.0, 5.0, 6.0)))

        out = logger.save(tmp_path / "flight.npz")
        assert out.exists()

        data = np.load(str(out))
        assert "state_timestamps_ns" in data
        assert "state_data" in data
        ts = data["state_timestamps_ns"]
        assert len(ts) == 2
        assert ts[0] == 1_000_000
        assert ts[1] == 2_000_000

        # state_data columns: pos(3) + vel(3) + ori(4) + accel_bias(3) + gyro_bias(3) = 16
        sd = data["state_data"]
        assert sd.shape == (2, 16)
        np.testing.assert_allclose(sd[0, :3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(sd[1, :3], [4.0, 5.0, 6.0])

    def test_save_and_reload_commands(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_command(_make_command(500, throttle=0.7))

        out = logger.save(tmp_path / "flight.npz")
        data = np.load(str(out))
        assert "command_timestamps_ns" in data
        cd = data["command_data"]
        assert cd.shape == (1, 4)
        assert cd[0, 0] == pytest.approx(0.7)  # throttle

    def test_save_and_reload_ground_truth(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_ground_truth(_make_state(100, (10.0, 20.0, 30.0)))

        out = logger.save(tmp_path / "flight.npz")
        data = np.load(str(out))
        gt = data["ground_truth_data"]
        # ground truth: pos(3) + vel(3) + ori(4) = 10
        assert gt.shape == (1, 10)
        np.testing.assert_allclose(gt[0, :3], [10.0, 20.0, 30.0])

    def test_save_gate_passages(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_gate_passage(1_000_000, gate_id=0, total_passed=1)
        logger.log_gate_passage(5_000_000, gate_id=1, total_passed=2)

        out = logger.save(tmp_path / "flight.npz")
        data = np.load(str(out))
        assert "gate_passage_timestamps_ns" in data
        assert "gate_passage_ids" in data
        gp_ts = data["gate_passage_timestamps_ns"]
        gp_ids = data["gate_passage_ids"]
        assert len(gp_ts) == 2
        assert gp_ids[0] == 0
        assert gp_ids[1] == 1

    def test_save_empty_logger(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        out = logger.save(tmp_path / "empty.npz")
        assert out.exists()
        data = np.load(str(out))
        assert len(data["state_timestamps_ns"]) == 0
        assert len(data["command_timestamps_ns"]) == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_state(_make_state(100))
        deep_path = tmp_path / "a" / "b" / "c" / "flight.npz"
        out = logger.save(deep_path)
        assert out.exists()

    def test_no_gate_passages_key_absent(self, tmp_path: Path) -> None:
        logger = FlightLogger()
        logger.log_state(_make_state(100))
        out = logger.save(tmp_path / "flight.npz")
        data = np.load(str(out))
        assert "gate_passage_timestamps_ns" not in data
