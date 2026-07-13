"""Tests for the dual-clock run recorder and the arrival-order ESKF replay."""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.deployments.vq2.attitude import G
from drone_racing.deployments.vq2.recorder import (
    RunRecorder,
    camera_clock_valid,
    camera_latency_s,
    capture_time_monotonic_ns,
    load_run,
    replay_arrival_order,
)

_FIX_EPOCH_NS = 1_783_724_000_000_000_000  # a camera epoch far from the µs IMU boot clock


def test_record_save_load_roundtrip(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_imu(1000, 5_000_000, np.array([0.1, 0.2, -9.8]), np.array([0.01, 0.0, -0.02]))
    rec.add_fix(_FIX_EPOCH_NS, 5_500_000, np.array([0.0, 0.0, 6.0]), 0, np.eye(3) * 0.04)
    assert rec.n_imu == 1 and rec.n_fix == 1
    data = load_run(rec.save(tmp_path / "run.npz"))
    assert data["imu_sensor_t"][0] == 1000 and data["imu_arrival_t"][0] == 5_000_000
    assert data["fix_sensor_t"][0] == _FIX_EPOCH_NS and data["fix_arrival_t"][0] == 5_500_000
    assert data["session_id"] == rec.session_id


def _moving_run(rec: RunRecorder, fix_at_imu: int, n: int = 20) -> None:
    """A forward-accelerating run; one gate fix arrives interleaved after ``fix_at_imu``
    IMU samples. Sensor clocks use radically different epochs (µs boot vs ns camera)."""
    dt_us, arr_step = 7000, 10_000_000  # 7 ms sensor dt, 10 ms arrival spacing
    for k in range(n):
        rec.add_imu(k * dt_us, k * arr_step, np.array([3.0, 0.0, -G]), np.zeros(3))
    # fix arrival lands strictly between two IMU arrivals → must fuse there, not at the end
    rec.add_fix(
        _FIX_EPOCH_NS,
        fix_at_imu * arr_step + arr_step // 2,
        np.array([0.0, 0.0, 6.0]),
        active_gate=0,
    )


def test_replay_fuses_fix_midway_by_arrival_not_at_end(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=10, n=20)  # fix arrives ~halfway through the IMU stream
    res = replay_arrival_order(load_run(rec.save(tmp_path / "run.npz")))
    assert res.reanchors == 1 and res.n_dup == 0 and res.n_drop == 0
    v_at_fix = abs(res.vel_at_gate_change[0][0])  # forward velocity when the fix fused
    v_final = abs(res.vel[-1, 0])
    # merged by ARRIVAL, the fix fuses at ~half the accel ramp; a (buggy) sensor-clock
    # merge would place the far-epoch fix at the very end (v_at_fix ≈ v_final).
    assert v_at_fix < 0.6 * v_final, f"fix fused at end, not midway: {v_at_fix} vs {v_final}"


def test_replay_rejects_missing_arrival_clock(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    del data["imu_arrival_t"]
    with pytest.raises(ValueError, match="arrival-clock"):
        replay_arrival_order(data)


def test_replay_rejects_nonmonotonic_arrival(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    data["imu_arrival_t"][3] = data["imu_arrival_t"][3] - 10**9  # jump backwards
    with pytest.raises(ValueError, match="non-monotonic"):
        replay_arrival_order(data)


def test_camera_latency_and_capture_time_mapping() -> None:
    cam = 1_000_000_000_000_000_000  # camera capture stamp (Unix-epoch ns)
    latency_ns = 40_000_000  # 40 ms transport+decode
    wall = cam + latency_ns  # wall clock at receipt (same epoch as the camera stamp)
    mono = 500_000_000_000  # monotonic clock at receipt (unrelated uptime origin)
    assert abs(camera_latency_s(cam, wall) - 0.040) < 1e-9
    assert camera_clock_valid(cam, wall)
    # capture time on the monotonic clock = receipt monotonic minus the measured latency
    assert capture_time_monotonic_ns(cam, wall, mono) == mono - latency_ns


def test_camera_clock_invalid_when_not_wall_epoch() -> None:
    # if sim_time_ns were NOT wall-epoch (e.g. µs-since-boot), the implied latency is absurd
    assert not camera_clock_valid(cam_sim_ns=40_000_000, arrival_wall_ns=1_780_000_000_000_000_000)


def test_delayed_frame_capture_vs_arrival_residual() -> None:
    """A frame captured at t_cap but arriving 60 ms later: fusing at ARRIVAL misplaces it by
    the 60 ms of rotation; capture-time recovers the true capture instant."""
    cam = 2_000_000_000_000_000_000
    latency_ns = 60_000_000
    wall = cam + latency_ns
    arrival_mono = 900_000_000_000
    cap_mono = capture_time_monotonic_ns(cam, wall, arrival_mono)
    assert arrival_mono - cap_mono == latency_ns  # arrival residual == the latency
    assert abs(camera_latency_s(cam, wall) - 0.060) < 1e-9


def test_replay_rejects_missing_schema_version(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    del data["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        replay_arrival_order(data)


def test_replay_rejects_wrong_schema_version(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    data["schema_version"] = np.int64(999)
    with pytest.raises(ValueError, match="schema"):
        replay_arrival_order(data)


def test_replay_rejects_inconsistent_array_lengths(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    data["imu"] = data["imu"][:-2]  # truncate one array → length mismatch
    with pytest.raises(ValueError, match="inconsistent lengths"):
        replay_arrival_order(data)


def test_replay_counts_duplicate_and_dropped_sensor_dt(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_imu(0, 0, np.array([0.0, 0.0, -G]), np.zeros(3))
    rec.add_imu(10_000, 10_000_000, np.array([0.0, 0.0, -G]), np.zeros(3))
    rec.add_imu(10_000, 20_000_000, np.array([0.0, 0.0, -G]), np.zeros(3))  # dup sensor ts
    rec.add_imu(510_000, 30_000_000, np.array([0.0, 0.0, -G]), np.zeros(3))  # 0.5 s sensor gap
    res = replay_arrival_order(load_run(rec.save(tmp_path / "run.npz")))
    assert res.n_dup >= 1 and res.n_drop >= 1
