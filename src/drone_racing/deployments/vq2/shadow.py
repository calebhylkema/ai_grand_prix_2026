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

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.deployments.vq2 import gate_relative as gr
from drone_racing.deployments.vq2 import perception as gates
from drone_racing.deployments.vq2 import recorder as recmod
from drone_racing.deployments.vq2.eskf import ESKF

if TYPE_CHECKING:
    from pathlib import Path

    from drone_racing.deployments.vq2.calibration import Shared
    from drone_racing.deployments.vq2.recorder import RunRecorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowState:
    """Atomic, immutable snapshot of the shadow ESKF for cross-thread diagnostic reads.

    Published by the worker each tick under its state lock; the control thread reads it
    (never the live ``kf``) so Phase-4A control telemetry can log the estimator alongside
    each command without any shared-mutable-state race. Log-only — never feeds control.
    """

    anchor_gate: int = -1  # race gate the world origin is anchored to (-1 = not yet anchored)
    initialised: bool = False  # attitude initialised (filter usable)
    last_fix_mono_ns: int = 0  # monotonic_ns of the last fused fix (0 = none) → freshness
    p: tuple[float, float, float] = (0.0, 0.0, 0.0)  # position rel. anchor gate (m)
    v: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world velocity (m/s)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll,pitch,yaw (rad)
    cov_diag: tuple[float, ...] = field(  # P diagonal: pos(3), vel(3), att(3)
        default_factory=lambda: (0.0,) * 9
    )
    sequence: int = 0
    published_mono_ns: int = 0


@dataclass(frozen=True)
class CoherentShadowSnapshot:
    """One atomically published estimator/target/guidance generation."""

    state: ShadowState
    target: gr.StructuredTarget
    next_target: gr.StructuredTarget
    decision: gr.GuidanceDecision


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
        frame_dir: Path | None = None,
    ) -> None:
        self.sh = sh
        self.recorder = recorder
        self._gate1_turn_sign = gate1_turn_sign
        self._gate1_band = gate1_band
        # Optional RAW-frame dump for offline perception analysis (diagnostic only). When
        # set, each processed frame's untouched BGR image is written here AFTER the
        # perception-latency window, so it never inflates proc_ms or the control loop.
        self._frame_dir = frame_dir
        if frame_dir is not None:
            frame_dir.mkdir(parents=True, exist_ok=True)
        self.kf = ESKF()
        self.tracker = gates.GateTracker()
        self.kf_needs_anchor = True
        self._imu_prev_us: int | None = None
        self._kf_active_gate = -1
        self._stop = threading.Event()  # per-attempt stop (Shared.stop is session-wide)
        self._done = threading.Event()  # set when the run loop has fully exited
        self._state_lock = threading.Lock()  # guards the published snapshots
        self._v_pub = np.zeros(3)  # atomic velocity snapshot for the trace diagnostic
        self._state_pub = ShadowState()  # atomic full-state snapshot (Phase-4A control log)
        self._last_fix_mono_ns = 0  # monotonic_ns of the last fused fix (freshness)
        # -- Phase-4B: worker-owned structured target + gate-relative guidance (log-only) --
        self._policy = gr.default_policy()  # evaluated HERE, never on the command loop
        self._target = gr.empty_structured_target()  # most recent selected structured target
        self._next_target = gr.empty_structured_target()  # provisional next race gate
        self._target_pub = self._target  # atomic published structured-target snapshot
        self._next_target_pub = self._next_target
        self._decision_pub = gr.idle_decision()  # atomic published guidance decision
        self._sel_track_id: int | None = None  # continuity: last selected track id
        self._sel_hits = 0  # consecutive frames the same track was selected
        self._transition_start_ns = 0  # monotonic_ns when active_gate advanced past anchor
        self._state_sequence = 0
        self._target_sequence = 0
        self._next_target_sequence = 0
        self._decision_sequence = 0
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
        self.transition_direction_only = 0  # promoted next target kept but not metric-fused
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

    def state_snapshot(self) -> ShadowState:
        """Atomic immutable snapshot of the shadow ESKF (safe cross-thread read, log-only)."""
        with self._state_lock:
            return self._state_pub

    def guidance_snapshot(self) -> gr.GuidanceDecision:
        """Atomic immutable gate-relative guidance decision (safe cross-thread read).

        Coherent with :meth:`state_snapshot` / :meth:`target_snapshot` — all three are
        republished together each tick under one lock, so a reader can never pair a fresh
        state with an unrelated old target. Log-only; the command loop only records it."""
        with self._state_lock:
            return self._decision_pub

    def target_snapshot(self) -> gr.StructuredTarget:
        """Atomic immutable snapshot of the worker's selected structured target (log-only)."""
        with self._state_lock:
            return self._target_pub

    def next_target_snapshot(self) -> gr.StructuredTarget:
        """Atomic provisional next-gate snapshot (never fused before promotion)."""
        with self._state_lock:
            return self._next_target_pub

    def coherent_snapshot(self) -> CoherentShadowSnapshot:
        """Estimator state + structured target + guidance decision from the SAME tick.

        Read together under one lock so the command loop cannot pair a fresh state with an
        unrelated older target/decision. Log-only."""
        with self._state_lock:
            return CoherentShadowSnapshot(
                self._state_pub,
                self._target_pub,
                self._next_target_pub,
                self._decision_pub,
            )

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
            active_gate_now = self.sh.active_gate
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
        pd = self.kf.P.diagonal()
        published_ns = time.monotonic_ns()
        self._state_sequence += 1
        snap = ShadowState(
            anchor_gate=self._kf_active_gate,
            initialised=self.kf.initialised,
            last_fix_mono_ns=self._last_fix_mono_ns,
            p=(float(self.kf.p[0]), float(self.kf.p[1]), float(self.kf.p[2])),
            v=(float(self.kf.v[0]), float(self.kf.v[1]), float(self.kf.v[2])),
            euler=tuple(float(x) for x in self.kf.euler),  # type: ignore[arg-type]
            cov_diag=tuple(float(x) for x in pd[:9]),
            sequence=self._state_sequence,
            published_mono_ns=published_ns,
        )
        decision = self._decide(active_gate_now, pd, snap)
        with self._state_lock:
            self._v_pub = self.kf.v.copy()
            self._state_pub = snap
            self._target_pub = self._target
            self._next_target_pub = self._next_target
            self._decision_pub = decision

    def _decide(
        self, active_gate: int, pd: np.ndarray, state: ShadowState | None = None
    ) -> gr.GuidanceDecision:
        """Evaluate the gate-relative policy on the COHERENT (state, target) pair (log-only).

        Runs ENTIRELY on the worker thread — the command loop performs no guidance. The
        transition timer measures time since the race gate advanced past the ESKF anchor.
        """
        now_ns = time.monotonic_ns()
        anchor = self._kf_active_gate
        if anchor >= 0 and active_gate == anchor + 1:
            if self._transition_start_ns == 0:
                self._transition_start_ns = now_ns
            transition_age_s = (now_ns - self._transition_start_ns) * 1e-9
        else:
            self._transition_start_ns = 0
            transition_age_s = 0.0
        fix_age_s = (now_ns - self._last_fix_mono_ns) * 1e-9 if self._last_fix_mono_ns else 1e9
        decision = self._policy.step(
            now_ns=now_ns,
            active_gate=active_gate,
            anchor_gate=anchor,
            initialised=self.kf.initialised,
            position_drone_rel_gate_ned_m=self.kf.p,
            velocity_ned_mps=self.kf.v,
            euler_rpy_rad=np.array(self.kf.euler),
            cov_diag=np.asarray(pd[:9], float),
            fix_age_s=fix_age_s,
            target=self._target,
            transition_age_s=transition_age_s,
        )
        self._decision_sequence += 1
        state_sequence = self._state_sequence if state is None else state.sequence
        state_mono_ns = now_ns if state is None else state.published_mono_ns
        return dataclasses.replace(
            decision,
            decision_mono_ns=time.monotonic_ns(),
            decision_sequence=self._decision_sequence,
            state_sequence=state_sequence,
            state_mono_ns=state_mono_ns,
            target_sequence=self._target.sequence,
            input_mono_ns=now_ns,
            initialised=self.kf.initialised,
            fix_age_s=fix_age_s,
            transition_age_s=transition_age_s,
        )

    def _predict_sample(self, sample: tuple, live: bool) -> None:
        su, arr, ax, ay, az, gx, gy, gz = sample
        acc = np.array([ax, ay, az])
        gyro = np.array([gx, gy, gz])
        if self.recorder is not None:
            self.recorder.add_imu(su, arr, acc, gyro, live)
        if self._imu_prev_us is not None:
            d = (su - self._imu_prev_us) * 1e-6
            if 0.0 < d < IMU_MAX_DT:
                # sim gyro is sign-inverted vs our FRD body frame (see recorder.GYRO_SIGN);
                # correct only at ESKF intake, matching replay_arrival_order exactly.
                self.kf.predict(acc, gyro * recmod.GYRO_SIGN, d)
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
        tracked = self.tracker.step_result(
            cands,
            active_gate_index=active_gate,
            expected_turn_sign=turn_sign,
            range_band=band,
            next_turn_sign=GATE1_TURN_SIGN if active_gate == 0 else None,
        )
        sel = tracked.active
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
                assoc_turn, assoc_band = self._assoc(active_gate)
                wrong_side = assoc_turn is not None and sel.bearing * assoc_turn <= 0.0
                slant = sel.slant_range
                out_of_band = (
                    assoc_band is not None
                    and slant is not None
                    and not (assoc_band[0] <= slant <= assoc_band[1])
                )
                # A pre-pass-confirmed next gate may remain a DIRECTION reference while
                # outside the metric fusion band. It is never fused until in-band. An
                # ordinary far acquisition or anything on the wrong side is still retired.
                if (
                    tracked.active_from_provisional
                    and active_gate == self._kf_active_gate + 1
                    and out_of_band
                    and not wrong_side
                ):
                    self.transition_direction_only += 1
                    disposition = "transition_direction"
                else:
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
                self._last_fix_mono_ns = arrival_ns  # freshness clock for the state snapshot
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

        self._update_structured_targets(frame_ev, tracked, disposition)
        self._record_frame_diag(frame_ev, sel, tracked.next, disposition, nis)
        if self._frame_dir is not None:  # raw image AFTER the latency window (diagnostic)
            trk = -1 if sel is None or sel.track_id is None else sel.track_id
            name = f"f{frame_id:04d}_g{active_gate}_t{trk}_{disposition}.png"
            cv2.imwrite(str(self._frame_dir / name), img)

    def _update_structured_targets(
        self, frame_ev: tuple, tracked: gates.GateTrackerResult, disposition: str
    ) -> None:
        """Build the immutable structured-target snapshot from the SELECTED tracker obs.

        The confirmed structured pipeline (never legacy ``gates.detect``). LOS is the
        metric-pose DIRECTION when posed (scale untrusted), else the horizontal bearing.
        Continuity = consecutive frames selecting the same track id."""
        frame_id, sensor_ns, arrival_ns, _wall_ns, active_gate, _img = frame_ev
        sel = tracked.active
        turn_sign, band = self._assoc(active_gate)
        tid = None if sel is None or sel.track_id is None else int(sel.track_id)
        if tid is not None and tid == self._sel_track_id:
            self._sel_hits += 1
        else:
            self._sel_track_id = tid
            self._sel_hits = 1 if tid is not None else 0
        self._target_sequence += 1
        if sel is None:
            self._target = dataclasses.replace(
                gr.empty_structured_target(arrival_ns), sequence=self._target_sequence
            )
        else:
            # The tracker owns visual continuity. In particular, promotion of a
            # pre-pass provisional must preserve its already-confirmed streak instead
            # of resetting policy confidence to one merely because it became active.
            # Test doubles/legacy observations may not publish a streak, so retain the
            # worker-local fallback for those callers.
            selected_hits = sel.track_streak if sel.track_streak > 0 else self._sel_hits
            self._target = self._structured_from_observation(
                sel,
                frame_id=frame_id,
                sensor_ns=sensor_ns,
                arrival_ns=arrival_ns,
                active_gate=active_gate,
                race_gate=active_gate,
                hits=selected_hits,
                turn_sign=turn_sign,
                band=band,
                disposition=disposition,
                sequence=self._target_sequence,
            )

        self._next_target_sequence += 1
        nxt = tracked.next
        if nxt is None:
            self._next_target = dataclasses.replace(
                gr.empty_structured_target(arrival_ns), sequence=self._next_target_sequence
            )
        else:
            self._next_target = self._structured_from_observation(
                nxt,
                frame_id=frame_id,
                sensor_ns=sensor_ns,
                arrival_ns=arrival_ns,
                active_gate=active_gate,
                race_gate=active_gate + 1,
                hits=max(1, nxt.track_streak),
                turn_sign=GATE1_TURN_SIGN,
                band=None,
                disposition="provisional",
                sequence=self._next_target_sequence,
            )

    @staticmethod
    def _structured_from_observation(
        sel: gates.GateObservation,
        *,
        frame_id: int,
        sensor_ns: int,
        arrival_ns: int,
        active_gate: int,
        race_gate: int,
        hits: int,
        turn_sign: float | None,
        band: tuple[float, float] | None,
        disposition: str,
        sequence: int,
    ) -> gr.StructuredTarget:
        """Freeze one fresh tracker observation for cross-thread publication."""
        has_metric = sel.pose_cam is not None and sel.output_mode in (
            gates.OutputMode.FULL_POSE,
            gates.OutputMode.POSITION_ONLY,
        )
        if has_metric:
            gate_body = gates.CAM_TO_BODY @ np.array(sel.pose_cam)
            n = float(np.linalg.norm(gate_body))
            los = gate_body / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        else:  # horizontal LOS from the image bearing (elevation unknown without a pose)
            b = float(sel.bearing)
            los = np.array([np.cos(b), np.sin(b), 0.0])
        slant = sel.slant_range
        wrong_side = turn_sign is not None and sel.bearing * turn_sign <= 0.0
        oob = band is not None and slant is not None and not (band[0] <= slant <= band[1])
        vroll = sel.vision_roll
        bx, by, bw, bh = sel.bbox
        return gr.StructuredTarget(
            valid=True,
            frame_id=frame_id,
            frame_sensor_ns=sensor_ns,
            arrival_ns=arrival_ns,
            active_gate_at_receipt=active_gate,
            track_id=-1 if sel.track_id is None else int(sel.track_id),
            race_gate_id=race_gate,
            track_hits=hits,
            output_mode=sel.output_mode.value,
            bearing_rad=float(sel.bearing),
            los_body_unit=los,
            slant_range_m=float("nan") if slant is None else float(slant),
            has_metric_pose=has_metric,
            confidence=float(sel.confidence),
            cov_trace=float("nan") if sel.pose_cov is None else float(np.trace(sel.pose_cov)),
            expected_turn_sign=0.0 if turn_sign is None else float(turn_sign),
            disposition=disposition,
            # out-of-band alone is NOT an identity refusal (right gate, right side, too far
            # to FUSE) — only a wrong-side lock is a genuine association refusal.
            association_refused=bool(disposition == "assoc_refused" and wrong_side),
            wrong_turn=bool(wrong_side),
            out_of_band=bool(oob),
            sequence=sequence,
            cx_px=float(sel.cx),
            cy_px=float(sel.cy),
            area_px2=float(sel.area),
            bbox_xywh=(int(bx), int(by), int(bw), int(bh)),
            visible_fraction=float(sel.visible_fraction),
            vision_roll_rad=float("nan") if vroll is None else float(vroll),
        )

    def _record_frame_diag(
        self, frame_ev: tuple, sel, next_sel, disposition: str, nis: float
    ) -> None:
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
        pnp_roll_value = None if sel is None else sel.vision_roll
        pnp_roll = float("nan") if pnp_roll_value is None else pnp_roll_value
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
            pnp_roll=pnp_roll,
            up_body=(float("nan"),) * 3 if sel is None or sel.up_body is None else sel.up_body,
            next_track=-1 if next_sel is None or next_sel.track_id is None else next_sel.track_id,
            next_streak=0 if next_sel is None else next_sel.track_streak,
            next_misses=0 if next_sel is None else next_sel.track_misses,
            next_bearing=float("nan") if next_sel is None else next_sel.bearing,
            next_slant=float("nan")
            if next_sel is None or next_sel.slant_range is None
            else next_sel.slant_range,
            next_conf=float("nan") if next_sel is None else next_sel.confidence,
            next_mode="none" if next_sel is None else next_sel.output_mode.value,
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
            f"transition_dir={self.transition_direction_only} "
            f"nis[med={np.median(nis):.1f} p95={np.quantile(nis, 0.95):.1f} max={nis.max():.1f}] "
            f"perc_ms[mean={pm.mean():.1f} p95={np.quantile(pm, 0.95):.1f} max={pm.max():.1f}] "
            f"imu[predicts={self.imu_predicts} dt_rej={self.imu_dt_rejects} hwm={self.imu_hwm} "
            f"ovf={ov} dup={dup} bw={bw} gap={gap}] frameq[drop={fdrop} hwm={fhwm}] "
            f"eskf[p={np.round(self.kf.p, 1)} v={np.round(self.kf.v, 1)} "
            f"Pdiag(p,v)={np.round(pd[:3], 2)},{np.round(pd[3:6], 2)} "
            f"nonfinite={self.eskf_nonfinite} nonpsd={self.eskf_nonpsd}]"
        )
