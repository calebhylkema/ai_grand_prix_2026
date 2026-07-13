"""Background shadow-mode ESKF worker for VQ2 — strictly log-only, off the control loop.

The command loop does ZERO shadow work. The receiver threads (``mav_rx``/``vision_rx``)
enqueue timestamped IMU samples and unique camera frames into bounded queues on
``Shared`` with minimal lock time, and this ONE dedicated worker owns all shadow state
(tracker, PnP, ESKF, recorder) single-threaded. Each tick it:

- drains the IMU FIFO in order (per-sample prediction; IMU is never silently dropped),
- processes only the NEWEST queued frame (dropping any stale backlog),
- fuses only tracker-confirmed FULL_POSE/POSITION_ONLY measurements (body-frame cov),
- records IMU + fixes (both clocks) for offline arrival-order replay,
- tracks queue/latency/health counters.

It never feeds attitude, guidance, thrust, safety, or command generation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2 import recorder as recmod
from drone_racing.deployments.vq2.eskf import ESKF

if TYPE_CHECKING:
    from drone_racing.deployments.vq2.calibration import Shared
    from drone_racing.deployments.vq2.recorder import RunRecorder

logger = logging.getLogger(__name__)

IMU_MAX_DT = 0.1  # reject IMU dt above this (a dropout, not a step)
_TICK_S = 0.002  # worker poll period

# Gate-1 association constraints for the shadow tracker. The range band is SLANT range
# (‖pose_cam‖, matching GateObservation.slant_range) — NOT camera depth. Reviewed from
# ADR-0003 (gate 1 is a right/down turn, mapped ~3–16 m). Gate 0 has no next-turn
# constraint; unmapped later gates stay unconstrained until they are mined.
GATE1_TURN_SIGN = 1.0  # +1 = right
GATE1_SLANT_BAND = (3.0, 16.0)  # metres, Euclidean (slant) range
_SHUTDOWN_TIMEOUT_S = 3.0  # max wait for the worker to finish before a save is refused


class ShadowWorker:
    """Owns the shadow tracker + PnP + ESKF + recorder on its own thread (log-only)."""

    def __init__(
        self,
        sh: Shared,
        recorder: RunRecorder | None = None,
        gate1_turn_sign: float = GATE1_TURN_SIGN,
        gate1_band: tuple[float, float] = GATE1_SLANT_BAND,
    ) -> None:
        self.sh = sh
        self.recorder = recorder
        self._gate1_turn_sign = gate1_turn_sign
        self._gate1_band = gate1_band
        self.kf = ESKF()
        self.tracker = gates.GateTracker()
        self.kf_needs_anchor = True
        self._imu_prev_us: int | None = None
        self._kf_active_gate = -1
        self._stop = threading.Event()  # per-attempt stop (Shared.stop is session-wide)
        self._done = threading.Event()  # set when the run loop has fully exited
        self._state_lock = threading.Lock()  # guards the published velocity snapshot
        self._v_pub = np.zeros(3)  # atomic velocity snapshot for the trace diagnostic
        self._thread = threading.Thread(target=self._run, daemon=True, name="shadow-worker")
        # health / counters
        self.imu_predicts = 0
        self.imu_dt_rejects = 0
        self.imu_hwm = 0  # max IMU samples drained in one tick
        self.frames_processed = 0
        self.frames_stale_dropped = 0
        self.candidates_total = 0
        self.selected = 0
        self.mode_counts: dict[str, int] = {}
        self.fixes = 0
        self.reanchors = 0
        self.rejects = 0  # confirmed target that was bearing-only/degenerate → not fused
        self.assoc_refused = 0  # metric pose out of the gate's slant band / wrong turn side
        self.accepted = 0  # update_gate passed the NIS gate
        self.nis_rejected = 0  # update_gate failed the NIS gate
        self.nis: list[float] = []
        self.reanchor_log: list[tuple[int, int]] = []  # (arrival_ns, gate)
        self.proc_ms: list[float] = []
        self.eskf_nonfinite = 0
        self.eskf_nonpsd = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Signal this worker to finish (one final tick, then set the completion event)."""
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def wait_done(self, timeout: float = _SHUTDOWN_TIMEOUT_S) -> bool:
        """Block until the run loop has fully exited; True iff it terminated in time."""
        return self._done.wait(timeout)

    def velocity_snapshot(self) -> np.ndarray:
        """Atomic copy of the shadow ESKF velocity (safe cross-thread read for diagnostics)."""
        with self._state_lock:
            return self._v_pub.copy()

    def _assoc(self, active_gate: int) -> tuple[float | None, tuple[float, float] | None]:
        """Course-association constraints for the active gate: (turn sign, slant band)."""
        if active_gate == 1:
            return self._gate1_turn_sign, self._gate1_band
        return None, None  # gate 0 / unmapped later gates: unconstrained

    def _fusion_eligible(self, obs: gates.GateObservation, active_gate: int) -> bool:
        """Whether a metric observation may be fused into the ESKF (reanchor OR update).

        Applies the SAME course constraints used for selection to the resolved metric
        pose: correct turn side and, if a band is defined, slant range inside it. An
        observation with no metric range cannot be checked against the band (it should
        not reach here — callers gate on output mode first)."""
        turn_sign, band = self._assoc(active_gate)
        if turn_sign is not None and obs.bearing * turn_sign <= 0.0:
            return False
        if band is not None and obs.slant_range is not None:
            lo, hi = band
            if not (lo <= obs.slant_range <= hi):
                return False
        return True

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not (self.sh.stop.is_set() or self._stop.is_set()):
                self._tick()
                time.sleep(_TICK_S)
            self._tick()  # final tick so a recording finalizes cleanly
        finally:
            self._done.set()

    def _tick(self) -> None:
        """Drain both queues and process IMU + the retained newest frame in ARRIVAL order.

        This is the same event ordering (by ``arrival_ns``) that ``replay_arrival_order``
        reconstructs from the recording, so live fusion and offline replay execute an
        identical sequence. Stale camera backlog is dropped, but the retained newest frame
        keeps its arrival position among the IMU events.
        """
        with self.sh.lock:
            imu = list(self.sh.imu_fifo)
            self.sh.imu_fifo.clear()
            frames = list(self.sh.frame_q)
            self.sh.frame_q.clear()
            live = self.sh.is_live()
        if imu:
            self.imu_hwm = max(self.imu_hwm, len(imu))
        # merged event stream by arrival_ns (kind 0=imu sorts before 1=frame at a tie)
        events: list[tuple[int, int, tuple]] = [(s[1], 0, s) for s in imu]
        if frames:
            self.frames_stale_dropped += len(frames) - 1  # drop stale backlog; keep newest
            newest = frames[-1]
            events.append((newest[2], 1, newest))  # newest[2] = arrival_ns
        events.sort(key=lambda e: (e[0], e[1]))
        for _arr, kind, payload in events:
            if kind == 0:
                self._predict_sample(payload, live)
            else:
                self._process_frame(payload)
        with self._state_lock:
            self._v_pub = self.kf.v.copy()

    def _predict_sample(self, sample: tuple, live: bool) -> None:
        su, arr, ax, ay, az, gx, gy, gz = sample
        acc = np.array([ax, ay, az])
        gyro = np.array([gx, gy, gz])
        if self.recorder is not None:
            self.recorder.add_imu(su, arr, acc, gyro, live)
        if self._imu_prev_us is not None:
            d = (su - self._imu_prev_us) * 1e-6
            if 0.0 < d < IMU_MAX_DT:
                self.kf.predict(acc, gyro, d)
                self.imu_predicts += 1
                if not live:  # gravity aiding only pre-launch (ground), like the AttEstimator
                    self.kf.update_gravity(acc)
            else:
                self.imu_dt_rejects += 1
        self._imu_prev_us = su

    def _process_frame(self, frame_ev: tuple) -> None:
        frame_id, sensor_ns, arrival_ns, wall_ns, active_gate, img = frame_ev
        t0 = time.perf_counter()
        cands = gates.detect_candidates(img)
        for obs in cands:
            obs.frame_id = frame_id
            obs.timestamp_ns = sensor_ns
            gates.solve_gate_pose(obs)
        turn_sign, band = self._assoc(active_gate)
        sel = self.tracker.step(
            cands,
            active_gate_index=active_gate,
            expected_turn_sign=turn_sign,
            range_band=band,
        )
        self.proc_ms.append((time.perf_counter() - t0) * 1e3)
        self.frames_processed += 1
        self.candidates_total += len(cands)
        for obs in cands:
            self.mode_counts[obs.output_mode.value] = (
                self.mode_counts.get(obs.output_mode.value, 0) + 1
            )

        # classify the frame's disposition + fuse only eligible metric observations
        disposition, nis = "none", float("nan")
        if sel is not None:
            self.selected += 1
            if sel.pose_cam is None or sel.output_mode not in (
                gates.OutputMode.FULL_POSE,
                gates.OutputMode.POSITION_ONLY,
            ):
                self.rejects += 1  # bearing-only/degenerate/inconsistent never fuse
                disposition = "bearing_only"
            elif not self._fusion_eligible(sel, active_gate):
                # HARD metric guard on EVERY reanchor/update: out-of-band / wrong-side =
                # wrong-gate lock (e.g. a 24-27 m gate-2 imposter for the 3-16 m gate 1).
                # Refuse, retire to re-acquire; never relax NIS/covariance to accept it.
                self.assoc_refused += 1
                self.tracker.retire_active()
                disposition = "assoc_refused"
            else:
                gate_body = gates.CAM_TO_BODY @ np.array(sel.pose_cam)
                cov_body = None if sel.pose_cov is None else gates.pose_cov_body(sel.pose_cov)
                if self.kf_needs_anchor or active_gate != self._kf_active_gate:
                    self.kf.reanchor(gate_body, cov_body)  # only from confirmed metric target
                    self.kf_needs_anchor = False
                    self._kf_active_gate = active_gate
                    self.reanchors += 1
                    self.reanchor_log.append((arrival_ns, active_gate))
                    disposition = "anchored"
                else:
                    info = self.kf.update_gate(gate_body, cov_body)
                    self.fixes += 1
                    self.nis.append(info.nis)
                    nis = info.nis
                    if info.accepted:
                        self.accepted += 1
                        disposition = "accepted"
                    else:
                        self.nis_rejected += 1
                        disposition = "nis_rejected"
                if self.recorder is not None:
                    self.recorder.add_fix(
                        sensor_ns,
                        arrival_ns,
                        gate_body,
                        active_gate,
                        cov_body,
                        sel.output_mode.value,
                    )
                self._check_health()

        self._record_frame_diag(frame_ev, sel, disposition, nis)

    def _record_frame_diag(self, frame_ev: tuple, sel, disposition: str, nis: float) -> None:
        """Per-frame diagnostic row (EVERY frame, fused or not) — lets the next run
        explain each observation: latency, target identity, PnP geometry, disposition,
        and shadow-vs-active attitude."""
        if self.recorder is None:
            return
        frame_id, sensor_ns, arrival_ns, wall_ns, active_gate, _img = frame_ev
        latency_s = recmod.camera_latency_s(sensor_ns, wall_ns)
        with self.sh.lock:
            active_att = self.sh.active_att
        sr, sp, sy = self.kf.euler
        cov_tr = (
            float("nan") if sel is None or sel.pose_cov is None else float(np.trace(sel.pose_cov))
        )
        self.recorder.add_frame(
            frame_id=frame_id,
            cam_ns=sensor_ns,
            arrival_ns=arrival_ns,
            wall_ns=wall_ns,
            latency_s=latency_s,
            active_gate=active_gate,
            track_id=-1 if sel is None or sel.track_id is None else sel.track_id,
            race_gate=-1 if sel is None or sel.race_gate_id is None else sel.race_gate_id,
            bearing=float("nan") if sel is None else sel.bearing,
            slant=float("nan") if sel is None or sel.slant_range is None else sel.slant_range,
            conf=float("nan") if sel is None else sel.confidence,
            mode="none" if sel is None else sel.output_mode.value,
            solver="" if sel is None or sel.solver is None else sel.solver,
            reproj=float("nan") if sel is None or sel.reproj_rms is None else sel.reproj_rms,
            ambig=float("nan")
            if sel is None or sel.ambiguity_margin is None
            else sel.ambiguity_margin,
            cov_tr=cov_tr,
            disposition=disposition,
            nis=nis,
            shadow_att=(float(sr), float(sp), float(sy)),
            active_att=(float(active_att[0]), float(active_att[1]), float(active_att[2])),
        )

    def _check_health(self) -> None:
        if not (np.all(np.isfinite(self.kf.P)) and np.all(np.isfinite(self.kf.p))):
            self.eskf_nonfinite += 1
        elif float(np.linalg.eigvalsh(self.kf.P).min()) < -1e-6:
            self.eskf_nonpsd += 1

    # ------------------------------------------------------------------
    def summary(self) -> str:
        nis = np.array(self.nis) if self.nis else np.array([0.0])
        pm = np.array(self.proc_ms) if self.proc_ms else np.array([0.0])
        with self.sh.lock:
            ov, dup, bw, gap = (
                self.sh.imu_fifo_overflow,
                self.sh.imu_dup_ts,
                self.sh.imu_backward_ts,
                self.sh.imu_gaps,
            )
            fdrop, fhwm = self.sh.frame_q_dropped, self.sh.frame_q_hwm
        pd = self.kf.P.diagonal()
        return (
            f"SHADOW frames={self.frames_processed}(stale_drop={self.frames_stale_dropped}) "
            f"cand={self.candidates_total} sel={self.selected} modes={self.mode_counts} "
            f"reanchor={self.reanchors} fixes={self.fixes} acc={self.accepted} "
            f"nis_rej={self.nis_rejected} bearing_rej={self.rejects} "
            f"assoc_refused={self.assoc_refused} "
            f"nis[med={np.median(nis):.1f} p95={np.quantile(nis, 0.95):.1f} max={nis.max():.1f}] "
            f"perc_ms[mean={pm.mean():.1f} p95={np.quantile(pm, 0.95):.1f} max={pm.max():.1f}] "
            f"imu[predicts={self.imu_predicts} dt_rej={self.imu_dt_rejects} hwm={self.imu_hwm} "
            f"ovf={ov} dup={dup} bw={bw} gap={gap}] frameq[drop={fdrop} hwm={fhwm}] "
            f"eskf[p={np.round(self.kf.p, 1)} v={np.round(self.kf.v, 1)} "
            f"Pdiag(p,v)={np.round(pd[:3], 2)},{np.round(pd[3:6], 2)} "
            f"nonfinite={self.eskf_nonfinite} nonpsd={self.eskf_nonpsd}]"
        )
