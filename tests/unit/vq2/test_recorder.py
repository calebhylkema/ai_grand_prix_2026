"""Tests for the dual-clock run recorder and the arrival-order ESKF replay."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from drone_racing.deployments.vq2.attitude import G, quat_mul, quat_to_matrix
from drone_racing.deployments.vq2.eskf import ESKF, quat_from_rotvec
from drone_racing.deployments.vq2.recorder import (
    SCHEMA_VERSION,
    RunRecorder,
    camera_clock_valid,
    camera_latency_s,
    capture_time_monotonic_ns,
    event_timeline,
    load_run,
    replay_arrival_order,
    replay_authority_decisions,
    replay_guidance_decisions,
    replay_handoff_identity,
    replay_visual_decisions,
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


def test_replay_applies_gyro_sign_convention(tmp_path) -> None:
    """The sim reports body rates inverted vs our FRD frame, so replay must integrate the
    sign-corrected gyro (recorder.GYRO_SIGN) — not the raw stream. A constant +yaw rate as
    reported must produce a NEGATIVE integrated yaw once the intake sign is applied. Guards
    the fix that collapsed run 20260713 NIS from 49 (1/25 fused) to ~1 (25/25) at gate 0."""
    from drone_racing.deployments.vq2.recorder import GYRO_SIGN

    assert GYRO_SIGN[2] < 0.0
    rec = RunRecorder()
    dt_us, wz, n = 7000, 2.0, 30  # constant +2 rad/s yaw as reported by the sim
    for k in range(n):  # accel = rest gravity → level init, then pure yaw
        rec.add_imu(k * dt_us, k * 10_000_000, np.array([0.0, 0.0, -G]), np.array([0.0, 0.0, wz]))
    res = replay_arrival_order(load_run(rec.save(tmp_path / "gyro.npz")))
    yaw = float(res.euler[-1, 2])
    expected = wz * GYRO_SIGN[2] * (n - 1) * dt_us * 1e-6  # ≈ -0.41 rad
    assert yaw < -0.2, f"gyro sign not applied in replay: yaw={yaw}"
    assert abs(yaw - expected) < 0.05, f"yaw {yaw} != sign-corrected integral {expected}"


def test_gyro_sign_matches_static_landmark_kinematics(tmp_path) -> None:
    """A fixed gate independently disambiguates physical FRD rotation from the sim's
    response-signed gyro. The correct sign keeps the predicted gate body vector aligned;
    the raw sign rotates it the wrong way and the unchanged NIS gate rejects it."""
    pitch = math.radians(-20.0)
    acc0 = np.array([G * math.sin(pitch), 0.0, -G * math.cos(pitch)])
    truth = ESKF()
    truth.initialise_attitude(acc0)
    q_true = truth.q.copy()
    p_drone_rel_gate = np.array([-20.0, 0.0, 0.0])
    raw_gyro = np.array([0.0, -1.0, 0.0])
    physical_gyro = -raw_gyro
    dt = 0.02
    rec = RunRecorder()
    rec.add_imu(0, 0, acc0, raw_gyro, True)
    rec.add_imu(20_000, 20_000_000, acc0, raw_gyro, True)
    rotation = quat_to_matrix(q_true)
    rec.add_fix(
        1,
        21_000_000,
        -rotation.T @ p_drone_rel_gate,
        0,
        np.eye(3) * 1e-4,
    )
    for k in range(2, 32):
        q_true = quat_mul(q_true, quat_from_rotvec(physical_gyro * dt))
        q_true /= np.linalg.norm(q_true)
        rotation = quat_to_matrix(q_true)
        accel = rotation.T @ np.array([0.0, 0.0, -G])
        arrival = k * 20_000_000
        rec.add_imu(k * 20_000, arrival, accel, raw_gyro, True)
        rec.add_fix(
            k,
            arrival + 1_000_000,
            -rotation.T @ p_drone_rel_gate,
            0,
            np.eye(3) * 1e-4,
        )
    data = load_run(rec.save(tmp_path / "landmark.npz"))
    corrected = replay_arrival_order(data, gyro_sign=(-1.0, -1.0, -1.0))
    raw = replay_arrival_order(data, gyro_sign=(1.0, 1.0, 1.0))
    assert corrected.n_fix_accepted == len(corrected.nis)
    assert np.median(corrected.nis) < 0.01
    assert raw.n_fix_accepted < len(raw.nis) // 4
    assert np.median(raw.nis) > 100.0


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


# --- Phase-4A: synchronized control-loop stream ------------------------------------


def _ctrl_row(mono_ns: int, phase: str, gate: int, **over: object) -> dict[str, object]:
    """A control-telemetry row like mission._record_control builds (defaults overridable)."""
    row: dict[str, object] = {
        "mono_ns": mono_ns,
        "phase": phase,
        "active_gate": gate,
        "tgt_track": -1,
        "tgt_mode": "none",
        "tgt_fresh_s": float("nan"),
        "tgt_bearing": float("nan"),
        "tgt_slant": float("nan"),
        "req": (0.3, 0.0, 0.0, 0.0),
        "des": (0.3, 0.0, 0.0, 0.0),
        "cal": (0.3, 0.0, 0.0, 0.0),
        "sat": (0.0, 0.0, 0.0, 0.0),
        "att": (0.0, 0.0, 0.0),
        "vz": 0.0,
        "sh_anchor": -1,
        "sh_init": 0,
        "sh_fresh_s": float("nan"),
        "sh_p": (0.0, 0.0, 0.0),
        "sh_v": (0.0, 0.0, 0.0),
        "sh_euler": (0.0, 0.0, 0.0),
        "sh_cov": (0.0,) * 9,
        "failsafe": "",
    }
    row.update(over)
    return row


def test_schema_version_is_v9() -> None:
    assert SCHEMA_VERSION == 9


def test_phase4c_v7_recording_remains_replayable(tmp_path) -> None:
    rec = RunRecorder()
    _moving_run(rec, fix_at_imu=5, n=10)
    data = load_run(rec.save(tmp_path / "run.npz"))
    data["schema_version"] = np.asarray(7, np.int64)
    assert replay_arrival_order(data).n_predict > 0


def test_shadow_policy_stream_roundtrip(tmp_path) -> None:
    """Phase-4B: the gate-relative shadow decision (sg_*) is stored beside each command."""
    rec = RunRecorder()
    rec.add_control(
        **_ctrl_row(
            200,
            "THREAD",
            1,
            sg_status="transition_acquire",
            sg_mode="transition",
            sg_track=7,
            sg_anchor=0,
            sg_decage=0.01,
            sg_set=(0.31, 0.4, -0.1, 0.2),  # hypothetical desired ACTUAL (thr,r,p,y rates)
            sg_acc=(2.0, 1.0, -0.5),
            sg_heading=0.35,
            sg_closure=9.8,
            sg_dist=float("nan"),  # transition trusts direction, not range
            sg_verr=(1.0, 0.5, 0.0),
            sg_fov=0.6,
            sg_sat="tilt|body_rate",
            st_valid=1,
            st_track=7,
            st_race=1,
            st_hits=4,
            st_mode="position_only",
            st_bearing=0.61,
            st_slant=31.0,
            st_disp="assoc_refused",
            st_oob=1,
            st_wrong=0,
            st_refused=0,
            loop_scheduled_ns=190,
            loop_released_ns=195,
            loop_lateness_ms=0.005,
            loop_missed=0,
            loop_exec_ms=2.1,
            auth_requested=1,
            auth_source="transition",
            auth_reason="transition_acquire",
            auth_entered=1,
            auth_exited=0,
            auth_age=0.01,
            auth_elapsed=0.2,
            auth_legacy=(0.30, 0.0, 0.0, 0.0),
            auth_transition=(0.31, 0.4, -0.1, 0.2),
            auth_selected=(0.305, 0.2, -0.05, 0.1),
            auth_slew=1,
        )
    )
    # a fail-closed row: only the reason is set; the setpoint columns stay NaN
    rec.add_control(**_ctrl_row(210, "SCAN", 1, sg_status="anchor_mismatch", sg_mode="none"))
    d = load_run(rec.save(tmp_path / "run.npz"))
    assert d["ctrl_sg_status"][0] == "transition_acquire"
    assert d["ctrl_sg_mode"][0] == "transition" and int(d["ctrl_sg_track"][0]) == 7
    assert np.allclose(d["ctrl_sg_set"][0], [0.31, 0.4, -0.1, 0.2])
    assert np.allclose(d["ctrl_sg_acc"][0], [2.0, 1.0, -0.5])
    assert d["ctrl_sg_closure"][0] == 9.8 and np.isnan(d["ctrl_sg_dist"][0])
    assert d["ctrl_sg_sat"][0] == "tilt|body_rate"
    # structured target stored separately from the legacy detector fields
    assert int(d["ctrl_st_track"][0]) == 7 and int(d["ctrl_st_race"][0]) == 1
    assert d["ctrl_st_mode"][0] == "position_only" and int(d["ctrl_st_oob"][0]) == 1
    assert d["ctrl_auth_source"][0] == "transition"
    assert np.allclose(d["ctrl_auth_selected"][0], [0.305, 0.2, -0.05, 0.1])
    assert int(d["ctrl_auth_slew"][0]) == 1
    assert d["ctrl_loop_exec_ms"][0] == pytest.approx(2.1)
    assert d["ctrl_sg_status"][1] == "anchor_mismatch"
    assert np.all(np.isnan(d["ctrl_sg_set"][1]))  # no setpoint when fail-closed


def test_schema_v8_provisional_next_target_roundtrip(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_control(
        **_ctrl_row(
            200,
            "G0",
            0,
            nt_valid=1,
            nt_track=12,
            nt_race=1,
            nt_hits=5,
            nt_mode="position_only",
            nt_bearing=0.55,
            nt_slant=31.0,
            nt_seq=9,
            nt_frame=77,
            nt_sensor_ns=1_000,
            nt_arrival_ns=2_000,
            nt_active=0,
            nt_los=(0.85, 0.52, 0.0),
            nt_conf=0.92,
        )
    )
    rec.add_frame(
        pnp_roll=0.2,
        up_body=(0.0, -0.2, -0.98),
        next_track=12,
        next_streak=5,
        next_misses=0,
        next_bearing=0.55,
        next_slant=31.0,
        next_conf=0.92,
        next_mode="position_only",
    )
    data = load_run(rec.save(tmp_path / "next.npz"))
    assert int(data["schema_version"]) == 9
    assert int(data["ctrl_nt_track"][0]) == 12 and int(data["ctrl_nt_hits"][0]) == 5
    assert np.allclose(data["ctrl_nt_los"][0], [0.85, 0.52, 0.0])
    assert int(data["frm_next_track"][0]) == 12
    assert np.allclose(data["frm_up_body"][0], [0.0, -0.2, -0.98])


def test_handoff_identity_replay_matches_provisional_promotion(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_control(**_ctrl_row(100, "G0", 0, nt_valid=1, nt_track=7, nt_hits=7))
    rec.add_control(**_ctrl_row(200, "G0", 0, nt_valid=1, nt_track=7, nt_hits=8))
    rec.add_control(**_ctrl_row(300, "BRAKE", 1, st_valid=1, st_track=7, st_hits=9))
    rec.add_control(**_ctrl_row(400, "BRAKE", 1, st_valid=1, st_track=7, st_hits=10))
    result = replay_handoff_identity(load_run(rec.save(tmp_path / "handoff.npz")))
    assert result.transition_row == 2 and result.provisional_rows == 2
    assert result.provisional_track_id == 7 and result.active_track_id == 7
    assert result.provisional_track_hits == 8 and result.active_track_hits == 9
    assert result.identity_match and result.confirmation_preserved


def test_handoff_identity_replay_detects_confirmation_reset(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_control(**_ctrl_row(100, "G0", 0, nt_valid=1, nt_track=7, nt_hits=8))
    rec.add_control(**_ctrl_row(200, "BRAKE", 1, st_valid=1, st_track=7, st_hits=1))
    result = replay_handoff_identity(load_run(rec.save(tmp_path / "reset.npz")))
    assert result.identity_match
    assert not result.confirmation_preserved


def test_handoff_identity_replay_rejects_pre_v8(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_control(**_ctrl_row(100, "G0", 0))
    data = load_run(rec.save(tmp_path / "old.npz"))
    data["schema_version"] = np.asarray(7, np.int64)
    with pytest.raises(ValueError, match="schema v8"):
        replay_handoff_identity(data)


def test_guidance_replay_uses_exact_worker_snapshot_identity(tmp_path) -> None:
    from drone_racing.deployments.vq2 import authority, mission
    from drone_racing.deployments.vq2.gate_relative import StructuredTarget, default_policy

    now_ns = 2_000_000_000
    bearing = math.radians(35.0)
    target = StructuredTarget(
        valid=True,
        frame_id=12,
        frame_sensor_ns=123,
        arrival_ns=now_ns - 5_000_000,
        active_gate_at_receipt=1,
        track_id=7,
        race_gate_id=1,
        track_hits=4,
        output_mode="position_only",
        bearing_rad=bearing,
        los_body_unit=np.array([math.cos(bearing), math.sin(bearing), 0.0]),
        slant_range_m=31.0,
        has_metric_pose=True,
        confidence=0.9,
        cov_trace=0.3,
        expected_turn_sign=1.0,
        disposition="assoc_refused",
        out_of_band=True,
        sequence=5,
    )
    p = np.zeros(3)
    v = np.array([12.0, 0.0, 0.0])
    euler = np.zeros(3)
    cov = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    decision = default_policy().step(
        now_ns=now_ns,
        active_gate=1,
        anchor_gate=0,
        initialised=True,
        position_drone_rel_gate_ned_m=p,
        velocity_ned_mps=v,
        euler_rpy_rad=euler,
        cov_diag=cov,
        fix_age_s=0.05,
        target=target,
        transition_age_s=0.3,
    )
    decision = dataclasses.replace(
        decision,
        decision_sequence=11,
        state_sequence=4,
        state_mono_ns=now_ns - 1_000_000,
        target_sequence=5,
        input_mono_ns=now_ns,
        initialised=True,
        fix_age_s=0.05,
        transition_age_s=0.3,
    )
    row = _ctrl_row(
        now_ns + 1_000_000,
        "BRAKE",
        1,
        sh_anchor=0,
        sh_init=1,
        sh_p=p,
        sh_v=v,
        sh_euler=euler,
        sh_cov=cov,
        sh_seq=4,
        sh_pub_ns=now_ns - 1_000_000,
        loop_released_ns=now_ns + 1_000_000,
    )
    mission._record_guidance(row, target, decision, now_ns + 1_000_000)
    selected = authority.TransitionAuthorityMux(True).select(
        now_ns=now_ns + 1_000_000,
        transition_start_ns=now_ns - 300_000_000,
        active_gate=1,
        anchor_gate=0,
        state_sequence=4,
        target=target,
        decision=decision,
        legacy=(0.3, 0.0, 0.0, 0.0),
    )
    row.update(
        auth_requested=1,
        auth_source=selected.source,
        auth_reason=selected.reason,
        auth_entered=int(selected.entered),
        auth_exited=int(selected.exited),
        auth_age=selected.decision_age_s,
        auth_elapsed=selected.authority_elapsed_s,
        auth_legacy=selected.legacy,
        auth_transition=selected.transition,
        auth_selected=selected.selected,
        auth_slew=int(selected.slew_limited),
    )
    rec = RunRecorder()
    rec.add_control(**row)
    data = load_run(rec.save(tmp_path / "guidance.npz"))
    replay = replay_guidance_decisions(data)
    assert replay.n_rows == 1 and replay.n_unique_decisions == 1
    assert replay.exact and replay.n_compared == 1 and not replay.mismatches
    authority_replay = replay_authority_decisions(data)
    assert authority_replay.exact and authority_replay.n_compared == 1


def test_control_stream_roundtrip_keeps_command_levels_separate(tmp_path) -> None:
    """req / des / cal are stored as three independent (thr,r,p,y) rows so the sim's rate
    inversion+scale can never be applied twice in replay."""
    rec = RunRecorder()
    rec.add_control(
        **_ctrl_row(
            100,
            "THREAD",
            1,
            req=(0.5, 9.9, 0.0, 5.0),  # requested (roll over MAX_RATE, yaw large)
            des=(0.5, 2.0, 0.0, 3.0),  # post-clamp desired ACTUAL rates
            cal=(0.5, -0.784, 0.0, -1.277),  # calibrated raw (des / RATE_GAIN)
            sat=(0.0, 1.0, 0.0, 1.0),  # roll+yaw clamped
            tgt_mode="metric",
            tgt_bearing=0.2,
            tgt_slant=6.1,
            sh_anchor=1,
            sh_v=(4.0, 1.0, -0.2),
        )
    )
    assert rec.n_ctrl == 1
    d = load_run(rec.save(tmp_path / "run.npz"))
    assert int(d["schema_version"]) == 9
    assert d["ctrl_phase"][0] == "THREAD" and int(d["ctrl_active_gate"][0]) == 1
    assert np.allclose(d["ctrl_req"][0], [0.5, 9.9, 0.0, 5.0])
    assert np.allclose(d["ctrl_des"][0], [0.5, 2.0, 0.0, 3.0])
    assert np.allclose(d["ctrl_cal"][0], [0.5, -0.784, 0.0, -1.277])
    assert np.allclose(d["ctrl_sat"][0], [0.0, 1.0, 0.0, 1.0])
    assert d["ctrl_tgt_mode"][0] == "metric"
    assert int(d["ctrl_sh_anchor"][0]) == 1 and np.allclose(d["ctrl_sh_v"][0], [4.0, 1.0, -0.2])
    # the three levels must actually differ (guards a double-applied calibration)
    assert not np.allclose(d["ctrl_des"][0], d["ctrl_cal"][0])


def test_event_timeline_interleaves_streams_by_monotonic_clock(tmp_path) -> None:
    """IMU (µs sensor epoch), fix (ns camera epoch) and control (monotonic) rows all carry
    the SAME monotonic receipt clock, so the merged timeline orders them by it regardless of
    their wildly different sensor epochs."""
    rec = RunRecorder()
    rec.add_imu(1_000, 100, np.zeros(3), np.zeros(3))  # mono 100
    rec.add_control(**_ctrl_row(150, "G0", 0))  # mono 150
    rec.add_frame(frame_id=0, arrival_ns=180)  # mono 180
    rec.add_imu(8_000, 200, np.zeros(3), np.zeros(3))  # mono 200
    rec.add_fix(_FIX_EPOCH_NS, 250, np.array([0.0, 0.0, 6.0]), 0)  # mono 250
    tl = event_timeline(load_run(rec.save(tmp_path / "run.npz")))
    assert [name for _t, name, _i in tl] == ["imu", "ctrl", "frame", "imu", "fix"]
    assert [t for t, _n, _i in tl] == [100, 150, 180, 200, 250]


def test_control_stream_absent_is_empty_not_error(tmp_path) -> None:
    """A run with no control rows still saves/loads and yields an empty ctrl timeline slice."""
    rec = RunRecorder()
    rec.add_imu(1_000, 100, np.zeros(3), np.zeros(3))
    d = load_run(rec.save(tmp_path / "run.npz"))
    assert d["ctrl_mono_ns"].shape[0] == 0
    assert all(name != "ctrl" for _t, name, _i in event_timeline(d))


# --------------------------------------------------------------------------- schema-v9 visual
def _st_row_fields(t, prefix: str) -> dict:
    """Serialize a StructuredTarget into ctrl_{st|nt}_* row fields (mirrors mission)."""
    return {
        f"{prefix}_valid": int(t.valid),
        f"{prefix}_frame": t.frame_id,
        f"{prefix}_sensor_ns": t.frame_sensor_ns,
        f"{prefix}_arrival_ns": t.arrival_ns,
        f"{prefix}_active": t.active_gate_at_receipt,
        f"{prefix}_track": t.track_id,
        f"{prefix}_race": t.race_gate_id,
        f"{prefix}_hits": t.track_hits,
        f"{prefix}_mode": t.output_mode,
        f"{prefix}_bearing": t.bearing_rad,
        f"{prefix}_los": tuple(float(x) for x in t.los_body_unit),
        f"{prefix}_slant": t.slant_range_m,
        f"{prefix}_metric": int(t.has_metric_pose),
        f"{prefix}_conf": t.confidence,
        f"{prefix}_cov": t.cov_trace,
        f"{prefix}_turn_sign": t.expected_turn_sign,
        f"{prefix}_disp": t.disposition,
        f"{prefix}_refused": int(t.association_refused),
        f"{prefix}_wrong": int(t.wrong_turn),
        f"{prefix}_oob": int(t.out_of_band),
        f"{prefix}_seq": t.sequence,
        f"{prefix}_cx": t.cx_px,
        f"{prefix}_cy": t.cy_px,
        f"{prefix}_area": t.area_px2,
        f"{prefix}_visfrac": t.visible_fraction,
        f"{prefix}_vroll": t.vision_roll_rad,
    }


def _visual_run_recording(tmp_path):
    """Drive the visual controller through a PRETURN→TURN sequence and record faithfully."""
    from drone_racing.deployments.vq2.gate_relative import (
        StructuredTarget,
        empty_structured_target,
    )
    from drone_racing.deployments.vq2.visual_handoff import (
        VisualHandoffController,
        VisualHandoffInput,
    )

    ms = 1_000_000
    now0 = 20_000_000_000

    def prov(now_ns, **over):
        base = dict(
            valid=True,
            frame_id=5,
            frame_sensor_ns=1,
            arrival_ns=now_ns,
            active_gate_at_receipt=0,
            track_id=3,
            race_gate_id=1,
            track_hits=6,
            output_mode="bearing_only",
            bearing_rad=math.radians(22.0),
            los_body_unit=np.array([1.0, 0.0, 0.0]),
            slant_range_m=float("nan"),
            has_metric_pose=False,
            confidence=0.9,
            cov_trace=float("nan"),
            expected_turn_sign=1.0,
            disposition="provisional",
            area_px2=4000.0,
            cx_px=420.0,
            cy_px=130.0,
            visible_fraction=0.5,
            sequence=1,
        )
        base.update(over)
        return StructuredTarget(**base)

    def gate0(now_ns, **over):
        base = dict(
            valid=True,
            frame_id=5,
            frame_sensor_ns=1,
            arrival_ns=now_ns,
            active_gate_at_receipt=0,
            track_id=0,
            race_gate_id=0,
            track_hits=20,
            output_mode="full_pose",
            bearing_rad=math.radians(2.0),
            los_body_unit=np.array([1.0, 0.0, 0.0]),
            slant_range_m=5.0,
            has_metric_pose=True,
            confidence=0.95,
            cov_trace=0.2,
            expected_turn_sign=0.0,
            disposition="anchored",
            area_px2=40000.0,
            cx_px=320.0,
            cy_px=130.0,
            visible_fraction=0.45,
            sequence=1,
        )
        base.update(over)
        return StructuredTarget(**base)

    inputs = []
    for k in range(4):  # PRETURN ticks (fresh provisional each loop)
        t = now0 + k * 16 * ms
        inputs.append(
            VisualHandoffInput(
                now_ns=t,
                training=True,
                flag_enabled=True,
                active_gate=0,
                anchor_gate=0,
                active_target=gate0(t),
                provisional_target=prov(t, sequence=k + 1),
                eskf_roll_rad=math.radians(6.0),
                eskf_roll_cov_rad2=0.001,
                eskf_att_cov_trace_rad2=0.01,
                state_age_s=0.02,
                legacy_command=(0.30, 0.0, -0.2, 0.0),
                control_roll_rad=0.0,
                control_pitch_rad=0.0,
                control_yaw_rad=0.0,
                body_gyro_rps=(0.0, 0.0, 0.0),
            )
        )
    for k in range(6):  # promotion + TURN ticks
        t = now0 + (5 + k) * 16 * ms
        act = prov(
            t,
            active_gate_at_receipt=1,
            race_gate_id=1,
            track_id=3,
            bearing_rad=math.radians(20.0 - k),
            sequence=100 + k,
        )
        inputs.append(
            VisualHandoffInput(
                now_ns=t,
                training=True,
                flag_enabled=True,
                active_gate=1,
                anchor_gate=0,
                active_target=act,
                provisional_target=empty_structured_target(t),
                eskf_roll_rad=math.radians(12.0),
                eskf_roll_cov_rad2=0.001,
                eskf_att_cov_trace_rad2=0.01,
                state_age_s=0.02,
                legacy_command=(0.34, 0.0, 1.4, 0.0),
                control_roll_rad=0.0,
                control_pitch_rad=0.0,
                control_yaw_rad=0.0,
                body_gyro_rps=(0.0, 0.0, 0.0),
            )
        )

    controller = VisualHandoffController()
    rec = RunRecorder()
    for inp in inputs:
        d = controller.step(inp)
        s = d.status
        cov = (0.0,) * 6 + (
            inp.eskf_roll_cov_rad2,
            inp.eskf_att_cov_trace_rad2 - inp.eskf_roll_cov_rad2,
            0.0,
        )
        row = _ctrl_row(
            inp.now_ns,
            "G0",
            inp.active_gate,
            loop_released_ns=inp.now_ns,
            sh_anchor=inp.anchor_gate,
            sh_euler=(inp.eskf_roll_rad, 0.0, 0.0),
            sh_cov=cov,
            sh_pub_ns=inp.now_ns - int(round(inp.state_age_s * 1e9)),
            failsafe=inp.abort_reason,
            att=(inp.control_roll_rad, inp.control_pitch_rad, inp.control_yaw_rad),
            vh_state=s.state.value,
            vh_source=s.source,
            vh_legacy=d.legacy,
            vh_selected=d.selected,
            vh_body_gyro=s.body_gyro_rps,
            **_st_row_fields(inp.active_target, "st"),
            **_st_row_fields(inp.provisional_target, "nt"),
        )
        rec.add_control(**row)
    return load_run(rec.save(tmp_path / "visual.npz"))


def test_visual_handoff_schema_v9_roundtrip(tmp_path) -> None:
    data = _visual_run_recording(tmp_path)
    assert int(data["schema_version"]) == 9
    assert "ctrl_vh_state" in data and "ctrl_st_cx" in data and "ctrl_nt_cx" in data
    # PRETURN reached, then TURN engaged (visual authority present)
    states = {str(s) for s in data["ctrl_vh_state"]}
    assert "preturn" in states and "turn" in states


def test_visual_handoff_exact_decision_replay(tmp_path) -> None:
    data = _visual_run_recording(tmp_path)
    result = replay_visual_decisions(data)
    assert result.exact, result.mismatches
    assert result.n_visual_rows > 0


def test_visual_replay_rejects_non_visual_recording(tmp_path) -> None:
    rec = RunRecorder()
    rec.add_control(**_ctrl_row(100, "G0", 0))  # no vh_* fields → not a visual run
    data = load_run(rec.save(tmp_path / "plain.npz"))
    with pytest.raises(ValueError, match="not a visual-handoff run"):
        replay_visual_decisions(data)


def test_visual_gate0_speed_brake_exact_replay(tmp_path) -> None:
    """A gate-0 exit-speed braking sequence replays byte-for-byte from recorded inputs."""
    import numpy as np

    from drone_racing.deployments.vq2.gate_relative import StructuredTarget, empty_structured_target
    from drone_racing.deployments.vq2.visual_handoff import (
        VisualHandoffController,
        VisualHandoffInput,
    )

    ms = 1_000_000
    t0 = 30_000_000_000

    def g0(t: int, slant: float) -> StructuredTarget:
        return StructuredTarget(
            valid=True,
            frame_id=int(t % 100000),
            frame_sensor_ns=0,
            arrival_ns=t,
            active_gate_at_receipt=0,
            track_id=1,
            race_gate_id=0,
            track_hits=10,
            output_mode="position_only",
            bearing_rad=math.radians(1.0),
            los_body_unit=np.array([1.0, 0.0, 0.0]),
            slant_range_m=slant,
            has_metric_pose=True,
            confidence=0.9,
            cov_trace=0.2,
            expected_turn_sign=0.0,
            disposition="accepted",
            area_px2=26000.0,
            cx_px=323.0,
            cy_px=120.0,
            visible_fraction=0.5,
        )

    controller = VisualHandoffController()
    rec = RunRecorder()
    speed = 10.0
    for k in range(8):
        t = t0 + k * 33 * ms
        act = g0(t, slant=8.0 - k * 0.33)
        inp = VisualHandoffInput(
            now_ns=t,
            training=True,
            flag_enabled=True,
            active_gate=0,
            anchor_gate=0,
            active_target=act,
            provisional_target=empty_structured_target(t),
            eskf_roll_rad=math.radians(3.0),
            eskf_roll_cov_rad2=0.001,
            eskf_att_cov_trace_rad2=0.01,
            state_age_s=0.02,
            legacy_command=(0.30, 0.0, -0.05, 0.0),
            eskf_speed_mps=speed,
            eskf_vel_cov_trace=0.5,
            eskf_fix_age_s=0.03,
            eskf_initialised=True,
        )
        d = controller.step(inp)
        s = d.status
        cov = (0.0,) * 6 + (0.001, 0.009, 0.0)
        row = _ctrl_row(
            t,
            "G0",
            0,
            loop_released_ns=t,
            sh_anchor=0,
            sh_init=1,
            sh_euler=(math.radians(3.0), 0.0, 0.0),
            sh_cov=cov,
            sh_pub_ns=t - 20 * ms,
            failsafe="",
            vh_state=s.state.value,
            vh_source=s.source,
            vh_legacy=d.legacy,
            vh_selected=d.selected,
            vh_spd_eskf=s.eskf_speed_mps,
            vh_spd_velcov=s.vel_cov_trace,
            vh_spd_fixage=s.gate0_fix_age_s,
            **_st_row_fields(act, "st"),
            **_st_row_fields(empty_structured_target(t), "nt"),
        )
        rec.add_control(**row)

    data = load_run(rec.save(tmp_path / "spd.npz"))
    # the run actually braked
    assert any(int(x) for x in data["ctrl_vh_spd_active"])
    result = replay_visual_decisions(data)
    assert result.exact, result.mismatches
