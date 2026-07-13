"""Task-6 shadow integration: Shared IMU/frame queues + the log-only ShadowWorker.

The worker owns all shadow state (tracker, PnP, ESKF, recorder) on its own thread; here
we drive its internal steps synchronously (no thread) for determinism. It is structurally
separate from Flight/control, so it cannot affect command generation.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2.attitude import G
from drone_racing.deployments.vq2.calibration import FRAME_Q_MAX, IMU_GAP_US, Shared
from drone_racing.deployments.vq2.recorder import RunRecorder, load_run, replay_arrival_order
from drone_racing.deployments.vq2.shadow import GATE1_SLANT_BAND, GATE1_TURN_SIGN, ShadowWorker

_A = (0.0, 0.0, -G, 0.0, 0.0, 0.0)  # a rest-pose IMU sample (accel, gyro)


class _SpyTracker:
    """Records the constraints passed to step(); returns no selection."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def step(self, cands, active_gate_index=None, expected_turn_sign=None, range_band=None):
        self.calls.append((active_gate_index, expected_turn_sign, range_band))
        return None


class _FixedTracker:
    """Always returns a fixed observation; counts retire_active() calls."""

    def __init__(self, obs) -> None:
        self.obs = obs
        self.retired = 0

    def step(self, *a, **k):
        return self.obs

    def retire_active(self) -> None:
        self.retired += 1


def _metric_obs(pose_cam, bearing: float, mode=gates.OutputMode.POSITION_ONLY):
    """A confirmed metric GateObservation with a given camera-frame pose (slant = ‖pose‖)."""
    return gates.GateObservation(
        cx=320.0,
        cy=180.0,
        area=15000.0,
        bbox=(280, 140, 80, 80),
        bearing=bearing,
        visibility=gates.Visibility.FULL,
        visible_fraction=1.0,
        outer_corners=None,
        inner_corners=None,
        confidence=0.9,
        pose_cam=pose_cam,
        pose_cov=np.eye(3) * 0.1,
        output_mode=mode,
    )


def _fuse_one(sh: Shared, active_gate: int, obs) -> ShadowWorker:
    """Drive one worker frame with a fixed selected observation."""
    sh.active_gate = active_gate
    w = ShadowWorker(sh)
    w.tracker = _FixedTracker(obs)  # type: ignore[assignment]
    # frame event: (frame_id, sensor_ns, arrival_ns, wall_ns, active_gate, image)
    w._process_frame((0, 1, 1, 1, active_gate, np.zeros((360, 640, 3), np.uint8)))
    return w


def _ring(cx: int = 320, cy: int = 180, half: int = 80) -> np.ndarray:
    img = np.zeros((360, 640, 3), np.uint8)
    inner = int(round(half * gates.GATE_INNER_M / gates.GATE_OUTER_M))
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 0, 255), -1)
    cv2.rectangle(img, (cx - inner, cy - inner), (cx + inner, cy + inner), (0, 0, 0), -1)
    return img


# --- Shared IMU FIFO health -----------------------------------------------------


def test_imu_fifo_counts_dup_backward_gap() -> None:
    sh = Shared()
    sh.push_imu(1000, 1, *_A)
    sh.push_imu(1000, 2, *_A)  # duplicate sensor timestamp
    sh.push_imu(900, 3, *_A)  # backward
    sh.push_imu(900 + IMU_GAP_US + 1, 4, *_A)  # large gap
    assert sh.imu_dup_ts == 1 and sh.imu_backward_ts == 1 and sh.imu_gaps == 1
    assert len(sh.imu_fifo) == 4 and sh.imu_fifo[0][:2] == (1000, 1)  # (sensor, arrival)


def test_imu_fifo_overflow_counts() -> None:
    sh = Shared()
    n = sh.imu_fifo.maxlen + 50
    for k in range(n):
        sh.push_imu(1000 * (k + 1), k, *_A)
    assert sh.imu_fifo_overflow == 50 and len(sh.imu_fifo) == sh.imu_fifo.maxlen


def test_frame_queue_drops_stale_on_overflow() -> None:
    sh = Shared()
    for k in range(FRAME_Q_MAX + 3):
        sh.push_frame(k, k, k, _ring())
    assert sh.frame_q_dropped == 3
    assert len(sh.frame_q) == FRAME_Q_MAX
    assert sh.frame_q[-1][0] == FRAME_Q_MAX + 2  # newest retained


# --- worker: per-sample IMU prediction -----------------------------------------


def test_worker_drain_imu_predicts_per_sample_and_records_both_clocks() -> None:
    sh = Shared()
    rec = RunRecorder()
    w = ShadowWorker(sh, rec)
    for k in range(6):
        sh.push_imu(10_000 * (k + 1), 5_000_000 * (k + 1), 0.1, 0.0, -G, 0.0, 0.0, 0.0)
    w._tick()
    assert len(sh.imu_fifo) == 0  # fully drained
    assert w.imu_predicts == 5  # 6 samples → 5 dt steps (first inits)
    assert rec.n_imu == 6  # every sample recorded with both clocks
    # a second drain with nothing queued does nothing (each sample processed once)
    w._tick()
    assert w.imu_predicts == 5


def test_worker_drain_imu_rejects_bad_dt() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    sh.push_imu(1000, 1, *_A)
    sh.push_imu(1000, 2, *_A)  # dt=0
    sh.push_imu(10_000_000, 3, *_A)  # ~10 s sensor gap → too large
    w._tick()
    assert w.imu_dt_rejects == 2


# --- worker: frame processing + fusion -----------------------------------------


def test_worker_processes_newest_frame_dropping_stale() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    for k in range(3):
        sh.push_frame(k, k, k, _ring())
    w._tick()  # one tick handles the newest, drops the 2 stale
    assert w.frames_processed == 1
    assert w.frames_stale_dropped == 2
    assert len(sh.frame_q) == 0


def _confirm(w: ShadowWorker, sh: Shared, n: int = 8) -> None:
    img = _ring()
    for k in range(n):
        sh.push_frame(k, k * 33_000_000, k * 33_000_000, img)
        w._tick()
        if w.reanchors > 0:
            break


def test_worker_reanchors_from_confirmed_metric_gate() -> None:
    sh = Shared()
    rec = RunRecorder()
    w = ShadowWorker(sh, rec)
    _confirm(w, sh)
    assert w.reanchors == 1 and not w.kf_needs_anchor
    assert rec.n_fix >= 1  # the reanchor fix is recorded (dual clocks)


def test_worker_never_fuses_bearing_only() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    img = _ring(cx=600, cy=180, half=80)  # partial at the right border → bearing_only
    for k in range(8):
        sh.push_frame(k, k, k, img)
        w._tick()
    assert w.reanchors == 0 and w.fixes == 0
    assert w.kf_needs_anchor  # never anchored from a bearing-only observation


def test_worker_records_fix_with_dual_clocks() -> None:
    sh = Shared()
    rec = RunRecorder()
    w = ShadowWorker(sh, rec)
    _confirm(w, sh)
    data_sensor = rec._fix_sensor[0]
    data_arrival = rec._fix_arrival[0]
    assert data_sensor >= 0 and data_arrival >= 0  # both clocks captured per fix


def test_worker_reports_no_eskf_health_anomalies() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    _confirm(w, sh, n=12)
    assert w.eskf_nonfinite == 0 and w.eskf_nonpsd == 0


# --- arrival-order interleave + frame active_gate snapshot ----------------------


def test_frame_between_imu_samples_processed_in_arrival_order() -> None:
    sh = Shared()
    sh.active_gate = 0
    w = ShadowWorker(sh)
    w.tracker = _SpyTracker()  # type: ignore[assignment]
    # IMU arrivals 10,20,40,50 ; frame arrival 30 → must sort IMU,IMU,FRAME,IMU,IMU
    for i, arr in enumerate((10, 20, 40, 50)):
        sh.push_imu(1000 * (i + 1), arr, *_A)
    sh.push_frame(0, 5, 30, _ring())
    w._tick()
    assert w.imu_predicts == 3  # 4 IMU → 3 dt steps (first inits)
    assert w.frames_processed == 1  # the frame was processed within the tick, not skipped


def test_frame_retains_gate_id_from_receipt_not_processing() -> None:
    sh = Shared()
    sh.active_gate = 0
    w = ShadowWorker(sh)
    spy = _SpyTracker()
    w.tracker = spy  # type: ignore[assignment]
    sh.push_frame(0, 1, 1, _ring())  # snapshots gate 0
    sh.active_gate = 1  # race advances BEFORE the worker processes the frame
    w._tick()
    assert spy.calls[-1][0] == 0  # tracker saw the gate present at receipt, not 1


def test_gate1_constraints_passed_to_tracker() -> None:
    sh = Shared()
    sh.active_gate = 1
    w = ShadowWorker(sh)
    spy = _SpyTracker()
    w.tracker = spy  # type: ignore[assignment]
    sh.push_frame(0, 1, 1, _ring())
    w._tick()
    assert spy.calls[-1] == (1, GATE1_TURN_SIGN, GATE1_SLANT_BAND)


def test_gate0_is_unconstrained() -> None:
    sh = Shared()
    sh.active_gate = 0
    w = ShadowWorker(sh)
    spy = _SpyTracker()
    w.tracker = spy  # type: ignore[assignment]
    sh.push_frame(0, 1, 1, _ring())
    w._tick()
    assert spy.calls[-1] == (0, None, None)


def test_stale_backlog_dropped_keeping_newest() -> None:
    sh = Shared()
    sh.active_gate = 0
    w = ShadowWorker(sh)
    spy = _SpyTracker()
    w.tracker = spy  # type: ignore[assignment]
    for k in range(4):
        sh.push_frame(k, k, 100 + k, _ring())
    w._tick()
    assert w.frames_processed == 1 and w.frames_stale_dropped == 3  # only newest processed


# --- attempt isolation ----------------------------------------------------------


def test_clear_shadow_queues_excludes_preattempt_events() -> None:
    sh = Shared()
    for k in range(5):  # pre-attempt junk queued during startup
        sh.push_imu(1000 * (k + 1), k, *_A)
        sh.push_frame(k, k, k, _ring())
    sh.clear_shadow_queues()  # explicit attempt boundary
    rec = RunRecorder()
    w = ShadowWorker(sh, rec)
    w._tick()  # nothing queued → nothing recorded
    assert rec.n_imu == 0 and rec.n_fix == 0 and w.frames_processed == 0
    assert sh.imu_fifo_overflow == 0 and sh.frame_q_dropped == 0  # counters reset too


def test_second_attempt_does_not_inherit_first_attempt_state() -> None:
    sh = Shared()
    w1 = ShadowWorker(sh)
    _confirm(w1, sh)
    assert w1.reanchors == 1
    # a fresh worker for attempt 2 starts with clean ESKF/tracker/counters
    sh.clear_shadow_queues()
    w2 = ShadowWorker(sh)
    assert w2.reanchors == 0 and w2.fixes == 0 and w2.kf_needs_anchor
    assert float(np.linalg.norm(w2.kf.p)) == 0.0 and len(w2.tracker.tracks) == 0


# --- lifecycle: completion event + safe snapshot --------------------------------


def test_wait_done_true_after_normal_termination() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    w.start()
    w.stop()
    assert w.wait_done(timeout=2.0)


def test_wait_done_false_for_stuck_worker() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    w._tick = lambda: time.sleep(10.0)  # type: ignore[method-assign]  # simulate a stuck tick
    w.start()
    w.stop()
    assert not w.wait_done(timeout=0.2)  # → mission must refuse to save


def test_velocity_snapshot_is_an_independent_copy() -> None:
    sh = Shared()
    w = ShadowWorker(sh)
    snap = w.velocity_snapshot()
    snap[0] = 999.0
    assert w.velocity_snapshot()[0] == 0.0  # mutating the copy doesn't touch worker state


# --- live worker vs arrival-order replay equivalence ----------------------------


# --- hard metric association guard on fusion ------------------------------------


def test_gate1_out_of_band_metric_refused_and_retired() -> None:
    """A track that becomes active (e.g. while bearing-only) but later resolves to 26 m
    must be refused by the 3-16 m gate-1 band, not fused, and the track retired."""
    sh = Shared()
    obs = _metric_obs(pose_cam=(2.0, 0.0, 25.9), bearing=0.3)  # slant ≈ 26 m, right side
    w = _fuse_one(sh, active_gate=1, obs=obs)
    assert w.assoc_refused == 1
    assert w.reanchors == 0 and w.fixes == 0  # nothing fused
    assert w.tracker.retired == 1  # unbound to re-acquire
    assert w.kf_needs_anchor  # ESKF never anchored from the wrong gate


def test_gate1_in_band_metric_is_fused() -> None:
    sh = Shared()
    obs = _metric_obs(pose_cam=(1.0, 0.0, 10.0), bearing=0.3)  # slant ≈ 10 m, in 3-16, right
    w = _fuse_one(sh, active_gate=1, obs=obs)
    assert w.assoc_refused == 0 and w.reanchors == 1  # anchored from the in-band gate


def test_gate1_wrong_turn_side_refused() -> None:
    sh = Shared()
    obs = _metric_obs(pose_cam=(-1.0, 0.0, 10.0), bearing=-0.3)  # in band but LEFT of centre
    w = _fuse_one(sh, active_gate=1, obs=obs)
    assert w.assoc_refused == 1 and w.reanchors == 0


def test_gate0_is_unconstrained_and_fuses_any_range() -> None:
    sh = Shared()
    obs = _metric_obs(pose_cam=(0.0, 0.0, 25.9), bearing=0.0)  # far, but gate 0 has no band
    w = _fuse_one(sh, active_gate=0, obs=obs)
    assert w.assoc_refused == 0 and w.reanchors == 1


def test_live_worker_matches_arrival_order_replay(tmp_path) -> None:
    sh = Shared()
    sh.active_gate = 0
    rec = RunRecorder()
    w = ShadowWorker(sh, rec)
    img = _ring()
    su = arr = 0
    for k in range(16):
        for _ in range(3):  # 3 IMU per tick
            su += 7000
            arr += 3_000_000
            sh.push_imu(su, arr, 0.2, 0.0, -G, 0.0, 0.0, 0.0)
        sh.push_frame(k, k * 33_000_000, arr - 1_000_000, img)  # frame BETWEEN this tick's IMU
        w._tick()
    su += 7000  # final IMU-only tick so the last event is a predict in worker AND replay
    arr += 3_000_000
    sh.push_imu(su, arr, 0.2, 0.0, -G, 0.0, 0.0, 0.0)
    w._tick()

    res = replay_arrival_order(load_run(rec.save(tmp_path / "equiv.npz")))
    assert w.reanchors >= 1 and res.reanchors >= 1  # fusion actually happened
    # identical event sequence ⇒ identical ESKF state (to float precision)
    assert np.allclose(w.kf.p, res.pos[-1], atol=1e-6)
    assert np.allclose(w.kf.v, res.vel[-1], atol=1e-6)
