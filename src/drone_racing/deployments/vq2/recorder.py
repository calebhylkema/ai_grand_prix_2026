"""Log-only run recorder + offline ESKF replay for VQ2.

A dedicated shadow worker (see ``shadow.py``) attaches a :class:`RunRecorder` (via
``aigp-vq2 --record``) and captures the timestamped IMU stream and validated PnP gate
fixes to a ``.npz`` file, without touching the control path. :func:`replay_arrival_order`
re-runs the repaired ESKF over that log offline.

Two clocks are preserved per stream (they are on **different epochs and units**, so they
must never be merged directly):

- ``*_sensor_time`` — the raw sensor stamp used for integration ``dt``:
  IMU ``HIGHRES_IMU.time_usec`` (microseconds since sim boot); frame ``sim_time_ns``
  (nanoseconds, camera epoch).
- ``*_arrival_time_ns`` — ``time.monotonic_ns()`` at receipt, a single common clock used
  to order IMU and fixes against each other.

``replay_arrival_order`` merges by **arrival time** (reproducing current live behaviour,
including camera transport/decode latency). Exact capture-time delayed fusion is a later
step needing an explicit IMU↔camera clock-offset model.

``.npz`` schema
---------------
- ``session_id``       ()      int64   recorder session (rejects cross-session merges)
- ``imu_sensor_t``     (N,)    int64   IMU time_usec (µs) — integration dt source
- ``imu_arrival_t``    (N,)    int64   IMU monotonic_ns at receipt — merge clock
- ``imu_live``         (N,)    int8    1 if race live at receipt (else ground gravity aiding)
- ``imu``              (N,6)   float   ax,ay,az,gx,gy,gz
- ``fix_sensor_t``     (M,)    int64   frame sim_time_ns (ns) — diagnostics/alignment
- ``fix_arrival_t``    (M,)    int64   frame monotonic_ns at receipt — merge clock
- ``fix_body``         (M,3)   float   gate position in body frame
- ``fix_gate``         (M,)    int64   active race-gate index
- ``fix_cov``          (M,3,3) float   body-frame measurement covariance (NaN → default)
- ``fix_mode``         (M,)    <U16    perception output mode (position_only / full_pose)

Synchronized control-loop stream (``ctrl_*``, one row per command; Phase-4A). All rows
carry ``ctrl_mono_ns`` = ``time.monotonic_ns()`` — the SAME clock as ``imu_arrival_t`` /
``fix_arrival_t`` — so command, IMU, frame and fix events interleave on one timeline. The
command boundary is stored at three separate levels (``ctrl_req`` requested, ``ctrl_des``
post-clamp desired ACTUAL rates, ``ctrl_cal`` calibrated raw output) so the sim's
rate inversion/scale can never be applied twice in replay. Each is ``(thr, roll, pitch,
yaw)``. The shadow-ESKF snapshot (``ctrl_sh_*``) is the atomic :class:`ShadowState` read
off the worker — never the live filter. All log-only; nothing here feeds control.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone_racing.deployments.vq2.eskf import ESKF

_DROP_DT_S = 0.1  # a gap larger than this between IMU samples is a dropout, not a step
SCHEMA_VERSION = 9  # bump when the .npz field layout changes; replay validates it
_REPLAY_SCHEMA_MIN = 7  # preserve Phase-4C recordings as estimator/guidance evidence


def _supported_replay_schema(data: dict[str, np.ndarray]) -> bool:
    version = int(data.get("schema_version", -1))
    return _REPLAY_SCHEMA_MIN <= version <= SCHEMA_VERSION


#   v4: added the synchronized control-loop stream (ctrl_*) — monotonic-clocked per-loop
#       phase/gate/target/command-boundary/shadow-ESKF telemetry (Phase-4A observability).
#   v5: added the gate-relative shadow-policy decision (ctrl_sg_*) recorded alongside each
#       active command (Phase-4B). Log-only hypothetical setpoint + fail-closed reason;
#       never fed to Flight.command (see gate_relative.py).
#   v6: guidance moved into ShadowWorker (worker-owned). ctrl_sg_* now carries the atomic
#       GuidanceDecision (mode/anchor/track/decision-age); ctrl_st_* carries the worker's
#       SELECTED structured target (track id, race gate, mode, disposition, band flags) —
#       kept SEPARATE from the legacy detector ctrl_tgt_*.
#   v7: exact worker decision/state/target sequence identities + complete policy inputs,
#       command-loop deadline telemetry, and the default-off transition authority audit.
#   v8: provisional next-gate identity (ctrl_nt_*) so a pre-pass gate-1 handoff is auditable.
#   v9: Training-only visual gate-0→1 handoff decision (ctrl_vh_*) + active/provisional image
#       geometry (ctrl_st_cx/cy/area/visfrac/vroll, ctrl_nt_*) for exact visual-decision replay.
_MAX_PLAUSIBLE_LATENCY_S = 2.0  # camera transport+decode latency beyond this = bad clock map

# The VQ2 sim reports body angular rates in a convention inverted on all three axes
# relative to our FRD body frame — the same inversion the command path already
# compensates via ``calibration.RATE_GAIN=(-2.55,-2.55,-2.35)``. The control-path attitude
# estimator survives it (it leans on accel leveling), but the shadow ESKF *integrates*
# gyro, so the raw sign drove the filter's thrust into the wrong world axis (velocity went
# "up" instead of "forward") and every gate update failed the χ²(3) gate. The accelerometer
# needs no flip. Raw gyro is RECORDED as received (faithful sensor log); this sign is
# applied only at ESKF intake — in the live worker AND this offline replay, so live==replay
# still holds. Evidence: run 20260713_004939 — as-received NIS median 49.4 (1/25 fused);
# sign-corrected NIS median 1.0 (25/25 fused) with the gate UNCHANGED at 16.
GYRO_SIGN = np.array([-1.0, -1.0, -1.0])


def camera_latency_s(cam_sim_ns: int, arrival_wall_ns: int) -> float:
    """Transport+decode latency (s): arrival wall clock minus the camera capture stamp.

    The camera ``sim_time_ns`` is (observed) Unix-epoch nanoseconds, so it is comparable
    to ``time.time_ns()`` captured at receipt. This is measured, then validated — never
    assumed (see :func:`camera_clock_valid`).
    """
    return (arrival_wall_ns - cam_sim_ns) * 1e-9


def camera_clock_valid(cam_sim_ns: int, arrival_wall_ns: int) -> bool:
    """True if the sim↔wall clock mapping is plausible (latency in [0, 2 s])."""
    lat = camera_latency_s(cam_sim_ns, arrival_wall_ns)
    return 0.0 <= lat <= _MAX_PLAUSIBLE_LATENCY_S


def capture_time_monotonic_ns(
    cam_sim_ns: int, arrival_wall_ns: int, arrival_monotonic_ns: int
) -> int:
    """Estimate the frame's CAPTURE time on the monotonic clock.

    ``capture_monotonic = arrival_monotonic - latency``, mapping the camera epoch stamp
    onto the receiver's monotonic clock so a future capture-time (delayed) fusion can
    place the measurement when it was taken, not when it arrived.
    """
    latency_ns = arrival_wall_ns - cam_sim_ns
    return int(arrival_monotonic_ns - latency_ns)


class RunRecorder:
    """Accumulates a run's IMU + gate-fix stream (dual clocks) and writes it to ``.npz``."""

    def __init__(self) -> None:
        self.session_id = time.time_ns()  # identifies this process/recording session
        self._imu_sensor: list[int] = []
        self._imu_arrival: list[int] = []
        self._imu_live: list[int] = []  # 1 if race live at receipt (→ no ground gravity aiding)
        self._imu: list[tuple[float, ...]] = []
        self._fix_sensor: list[int] = []
        self._fix_arrival: list[int] = []
        self._fix_body: list[tuple[float, float, float]] = []
        self._fix_gate: list[int] = []
        self._fix_cov: list[np.ndarray] = []
        self._fix_mode: list[str] = []
        self._frames: list[dict] = []  # per-frame diagnostics (every frame, fused or not)
        # Control-loop stream: written by the CONTROL thread (add_control); IMU/fix/frame
        # streams are written by the shadow WORKER thread. Separate lists + a lock keep the
        # two writers safe; save() runs only after both threads have stopped.
        self._ctrl: list[dict] = []
        self._ctrl_lock = threading.Lock()

    def add_frame(self, **fields: object) -> None:
        """Record one per-frame diagnostic row (see :func:`_frame_columns` for fields)."""
        self._frames.append(fields)

    def add_control(self, **fields: object) -> None:
        """Record one control-loop telemetry row (see :func:`_control_columns` for fields).

        Thread-safe: called from the control thread while the worker writes the sensor
        streams. Log-only — this never influences the command it describes.
        """
        with self._ctrl_lock:
            self._ctrl.append(fields)

    def add_imu(
        self,
        sensor_usec: int,
        arrival_ns: int,
        accel: np.ndarray,
        gyro: np.ndarray,
        live: bool = True,
    ) -> None:
        self._imu_sensor.append(int(sensor_usec))
        self._imu_arrival.append(int(arrival_ns))
        self._imu_live.append(1 if live else 0)
        self._imu.append((*map(float, accel), *map(float, gyro)))

    def add_fix(
        self,
        sensor_ns: int,
        arrival_ns: int,
        gate_body: np.ndarray,
        active_gate: int,
        cov: np.ndarray | None = None,
        mode: str = "full_pose",
    ) -> None:
        self._fix_sensor.append(int(sensor_ns))
        self._fix_arrival.append(int(arrival_ns))
        self._fix_body.append(tuple(float(x) for x in gate_body))  # type: ignore[arg-type]
        self._fix_gate.append(int(active_gate))
        self._fix_cov.append(np.full((3, 3), np.nan) if cov is None else np.asarray(cov, float))
        self._fix_mode.append(mode)

    @property
    def n_imu(self) -> int:
        return len(self._imu_sensor)

    @property
    def n_fix(self) -> int:
        return len(self._fix_sensor)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def n_ctrl(self) -> int:
        with self._ctrl_lock:
            return len(self._ctrl)

    def _frame_columns(self) -> dict[str, np.ndarray]:
        """Serialize per-frame diagnostic rows into named ``frm_*`` arrays."""
        fr = self._frames

        def fcol(k: str) -> np.ndarray:
            return np.asarray([f.get(k, np.nan) for f in fr], np.float64)

        def icol(k: str) -> np.ndarray:
            return np.asarray([f.get(k, -1) for f in fr], np.int64)

        def scol(k: str) -> np.ndarray:
            return np.asarray([f.get(k, "") for f in fr], dtype="<U16")

        def att(k: str) -> np.ndarray:
            return np.asarray([f.get(k, (np.nan, np.nan, np.nan)) for f in fr], np.float64).reshape(
                -1, 3
            )

        return {
            "frm_frame_id": icol("frame_id"),
            "frm_cam_ns": icol("cam_ns"),
            "frm_arrival_ns": icol("arrival_ns"),
            "frm_wall_ns": icol("wall_ns"),
            "frm_latency_s": fcol("latency_s"),
            "frm_active_gate": icol("active_gate"),
            "frm_track_id": icol("track_id"),
            "frm_race_gate": icol("race_gate"),
            "frm_bearing": fcol("bearing"),
            "frm_slant": fcol("slant"),
            "frm_conf": fcol("conf"),
            "frm_mode": scol("mode"),
            "frm_solver": scol("solver"),
            "frm_reproj": fcol("reproj"),
            "frm_ambig": fcol("ambig"),
            "frm_cov_tr": fcol("cov_tr"),
            "frm_disposition": scol("disposition"),
            "frm_nis": fcol("nis"),
            "frm_shadow_att": att("shadow_att"),
            "frm_active_att": att("active_att"),
            "frm_pnp_roll": fcol("pnp_roll"),
            "frm_up_body": att("up_body"),
            "frm_next_track": icol("next_track"),
            "frm_next_streak": icol("next_streak"),
            "frm_next_misses": icol("next_misses"),
            "frm_next_bearing": fcol("next_bearing"),
            "frm_next_slant": fcol("next_slant"),
            "frm_next_conf": fcol("next_conf"),
            "frm_next_mode": scol("next_mode"),
        }

    def _control_columns(self) -> dict[str, np.ndarray]:
        """Serialize control-loop telemetry rows into named ``ctrl_*`` arrays."""
        with self._ctrl_lock:
            cr = list(self._ctrl)

        def fcol(k: str) -> np.ndarray:
            return np.asarray([c.get(k, np.nan) for c in cr], np.float64)

        def icol(k: str) -> np.ndarray:
            return np.asarray([c.get(k, -1) for c in cr], np.int64)

        def scol(k: str) -> np.ndarray:
            return np.asarray([c.get(k, "") for c in cr], dtype="<U16")

        def vcol(k: str, width: int) -> np.ndarray:
            fill = (np.nan,) * width
            return np.asarray([c.get(k, fill) for c in cr], np.float64).reshape(-1, width)

        def s32(k: str) -> np.ndarray:
            return np.asarray([c.get(k, "") for c in cr], dtype="<U64")

        return {
            "ctrl_mono_ns": icol("mono_ns"),
            "ctrl_phase": scol("phase"),
            "ctrl_active_gate": icol("active_gate"),
            "ctrl_tgt_track": icol("tgt_track"),
            "ctrl_tgt_mode": scol("tgt_mode"),
            "ctrl_tgt_fresh_s": fcol("tgt_fresh_s"),
            "ctrl_tgt_bearing": fcol("tgt_bearing"),
            "ctrl_tgt_slant": fcol("tgt_slant"),
            "ctrl_req": vcol("req", 4),  # (thr, roll, pitch, yaw) requested
            "ctrl_des": vcol("des", 4),  # post-clamp desired ACTUAL rates
            "ctrl_cal": vcol("cal", 4),  # calibrated raw command (after RATE_GAIN)
            "ctrl_sat": vcol("sat", 4),  # per-axis clamp flags (thr always 0)
            "ctrl_att": vcol("att", 3),  # active-estimator roll,pitch,yaw (rad)
            "ctrl_vz": fcol("vz"),
            "ctrl_sh_anchor": icol("sh_anchor"),
            "ctrl_sh_init": icol("sh_init"),
            "ctrl_sh_fresh_s": fcol("sh_fresh_s"),
            "ctrl_sh_p": vcol("sh_p", 3),
            "ctrl_sh_v": vcol("sh_v", 3),
            "ctrl_sh_euler": vcol("sh_euler", 3),
            "ctrl_sh_cov": vcol("sh_cov", 9),  # P diagonal: pos(3), vel(3), att(3)
            "ctrl_sh_seq": icol("sh_seq"),
            "ctrl_sh_pub_ns": icol("sh_pub_ns"),
            "ctrl_failsafe": scol("failsafe"),
            "ctrl_loop_scheduled_ns": icol("loop_scheduled_ns"),
            "ctrl_loop_released_ns": icol("loop_released_ns"),
            "ctrl_loop_lateness_ms": fcol("loop_lateness_ms"),
            "ctrl_loop_missed": icol("loop_missed"),
            "ctrl_loop_exec_ms": fcol("loop_exec_ms"),
            "ctrl_loop_compute_ms": fcol("loop_compute_ms"),
            "ctrl_loop_post_ms": fcol("loop_post_ms"),
            # -- Phase-4B worker-owned guidance decision (log-only; never fed to command) --
            "ctrl_sg_status": s32("sg_status"),  # eligibility / fail-closed reason
            "ctrl_sg_mode": scol("sg_mode"),  # anchored / transition / none
            "ctrl_sg_track": icol("sg_track"),  # structured track id used by guidance
            "ctrl_sg_active": icol("sg_active"),  # worker race gate at decision time
            "ctrl_sg_anchor": icol("sg_anchor"),  # ESKF anchor gate at decision time
            "ctrl_sg_decage": fcol("sg_decage"),  # decision age at record time (s)
            "ctrl_sg_seq": icol("sg_seq"),
            "ctrl_sg_state_seq": icol("sg_state_seq"),
            "ctrl_sg_state_ns": icol("sg_state_ns"),
            "ctrl_sg_target_seq": icol("sg_target_seq"),
            "ctrl_sg_input_ns": icol("sg_input_ns"),
            "ctrl_sg_decision_ns": icol("sg_decision_ns"),
            "ctrl_sg_fix_age": fcol("sg_fix_age"),
            "ctrl_sg_transition_age": fcol("sg_transition_age"),
            "ctrl_sg_initialised": icol("sg_initialised"),
            "ctrl_sg_set": vcol("sg_set", 4),  # hypothetical desired ACTUAL (thr,r,p,y rates)
            "ctrl_sg_acc": vcol("sg_acc", 3),  # requested NED acceleration (m/s^2)
            "ctrl_sg_heading": fcol("sg_heading"),  # desired heading (rad)
            "ctrl_sg_closure": fcol("sg_closure"),  # closure-speed objective (m/s)
            "ctrl_sg_dist": fcol("sg_dist"),  # ESKF gate-relative distance (m; nan in transit)
            "ctrl_sg_verr": vcol("sg_verr", 3),  # NED velocity error (m/s)
            "ctrl_sg_fov": fcol("sg_fov"),  # gate FOV margin (rad)
            "ctrl_sg_sat": s32("sg_sat"),  # limiting/saturation flags ('|'-joined)
            # -- worker's SELECTED structured target (separate from legacy ctrl_tgt_*) -----
            "ctrl_st_valid": icol("st_valid"),
            "ctrl_st_track": icol("st_track"),
            "ctrl_st_race": icol("st_race"),
            "ctrl_st_hits": icol("st_hits"),
            "ctrl_st_mode": scol("st_mode"),
            "ctrl_st_bearing": fcol("st_bearing"),
            "ctrl_st_slant": fcol("st_slant"),
            "ctrl_st_disp": scol("st_disp"),
            "ctrl_st_oob": icol("st_oob"),
            "ctrl_st_wrong": icol("st_wrong"),
            "ctrl_st_refused": icol("st_refused"),
            "ctrl_st_seq": icol("st_seq"),
            "ctrl_st_frame": icol("st_frame"),
            "ctrl_st_sensor_ns": icol("st_sensor_ns"),
            "ctrl_st_arrival_ns": icol("st_arrival_ns"),
            "ctrl_st_active": icol("st_active"),
            "ctrl_st_los": vcol("st_los", 3),
            "ctrl_st_metric": icol("st_metric"),
            "ctrl_st_conf": fcol("st_conf"),
            "ctrl_st_cov": fcol("st_cov"),
            "ctrl_st_turn_sign": fcol("st_turn_sign"),
            "ctrl_st_cx": fcol("st_cx"),
            "ctrl_st_cy": fcol("st_cy"),
            "ctrl_st_area": fcol("st_area"),
            "ctrl_st_visfrac": fcol("st_visfrac"),
            "ctrl_st_vroll": fcol("st_vroll"),
            # -- confirmed provisional next-gate identity (never fused pre-promotion) ----
            "ctrl_nt_valid": icol("nt_valid"),
            "ctrl_nt_track": icol("nt_track"),
            "ctrl_nt_race": icol("nt_race"),
            "ctrl_nt_hits": icol("nt_hits"),
            "ctrl_nt_mode": scol("nt_mode"),
            "ctrl_nt_bearing": fcol("nt_bearing"),
            "ctrl_nt_slant": fcol("nt_slant"),
            "ctrl_nt_seq": icol("nt_seq"),
            "ctrl_nt_frame": icol("nt_frame"),
            "ctrl_nt_sensor_ns": icol("nt_sensor_ns"),
            "ctrl_nt_arrival_ns": icol("nt_arrival_ns"),
            "ctrl_nt_active": icol("nt_active"),
            "ctrl_nt_los": vcol("nt_los", 3),
            "ctrl_nt_conf": fcol("nt_conf"),
            "ctrl_nt_turn_sign": fcol("nt_turn_sign"),
            "ctrl_nt_refused": icol("nt_refused"),
            "ctrl_nt_wrong": icol("nt_wrong"),
            "ctrl_nt_oob": icol("nt_oob"),
            "ctrl_nt_metric": icol("nt_metric"),
            "ctrl_nt_cov": fcol("nt_cov"),
            "ctrl_nt_disp": scol("nt_disp"),
            "ctrl_nt_cx": fcol("nt_cx"),
            "ctrl_nt_cy": fcol("nt_cy"),
            "ctrl_nt_area": fcol("nt_area"),
            "ctrl_nt_visfrac": fcol("nt_visfrac"),
            "ctrl_nt_vroll": fcol("nt_vroll"),
            # -- Phase-4C default-off Training-only command-source audit ----------------
            "ctrl_auth_requested": icol("auth_requested"),
            "ctrl_auth_source": scol("auth_source"),
            "ctrl_auth_reason": s32("auth_reason"),
            "ctrl_auth_entered": icol("auth_entered"),
            "ctrl_auth_exited": icol("auth_exited"),
            "ctrl_auth_age": fcol("auth_age"),
            "ctrl_auth_elapsed": fcol("auth_elapsed"),
            "ctrl_auth_legacy": vcol("auth_legacy", 4),
            "ctrl_auth_transition": vcol("auth_transition", 4),
            "ctrl_auth_selected": vcol("auth_selected", 4),
            "ctrl_auth_slew": icol("auth_slew"),
            # -- schema-v9 Training-only visual gate-0→1 handoff decision (log-only) ------
            "ctrl_vh_state": scol("vh_state"),
            "ctrl_vh_reason": s32("vh_reason"),
            "ctrl_vh_source": scol("vh_source"),
            "ctrl_vh_active": icol("vh_active"),
            "ctrl_vh_locked": icol("vh_locked"),
            "ctrl_vh_in_track": icol("vh_in_track"),
            "ctrl_vh_in_seq": icol("vh_in_seq"),
            "ctrl_vh_bearing": fcol("vh_bearing"),
            "ctrl_vh_bearing_rate": fcol("vh_bearing_rate"),
            "ctrl_vh_pred_bearing": fcol("vh_pred_bearing"),
            "ctrl_vh_cy_err": fcol("vh_cy_err"),
            "ctrl_vh_cy_rate": fcol("vh_cy_rate"),
            "ctrl_vh_dropout": fcol("vh_dropout"),
            "ctrl_vh_roll_tgt": fcol("vh_roll_tgt"),
            "ctrl_vh_roll_est": fcol("vh_roll_est"),
            "ctrl_vh_att_cov": fcol("vh_att_cov"),
            "ctrl_vh_roll_valid": icol("vh_roll_valid"),
            "ctrl_vh_in_state_s": fcol("vh_in_state_s"),
            "ctrl_vh_transition_s": fcol("vh_transition_s"),
            "ctrl_vh_safe_exit": s32("vh_safe_exit"),
            "ctrl_vh_chan": scol("vh_chan"),  # authoritative channels bitmask "trpy"
            "ctrl_vh_entered": icol("vh_entered"),
            "ctrl_vh_exited": icol("vh_exited"),
            "ctrl_vh_slew": icol("vh_slew"),
            "ctrl_vh_succeeded": icol("vh_succeeded"),
            "ctrl_vh_legacy": vcol("vh_legacy", 4),
            "ctrl_vh_requested": vcol("vh_requested", 4),
            "ctrl_vh_selected": vcol("vh_selected", 4),  # exactly what Flight.command sent
            "ctrl_vh_terminate": icol("vh_terminate"),
            "ctrl_vh_reacq_elapsed": fcol("vh_reacq_elapsed"),
            "ctrl_vh_reacq_yaw": fcol("vh_reacq_yaw"),
            "ctrl_vh_reacq_prev": icol("vh_reacq_prev"),
            "ctrl_vh_reacq_cand": icol("vh_reacq_cand"),
            "ctrl_vh_reacq_hits": icol("vh_reacq_hits"),
            "ctrl_vh_reacq_resid": fcol("vh_reacq_resid"),
            "ctrl_vh_reacq_result": s32("vh_reacq_result"),
            # -- gate-0 exit-speed control (Phase 3) diagnostics -------------------------
            "ctrl_vh_spd_active": icol("vh_spd_active"),
            "ctrl_vh_spd_reason": s32("vh_spd_reason"),
            "ctrl_vh_spd_eskf": fcol("vh_spd_eskf"),
            "ctrl_vh_spd_velcov": fcol("vh_spd_velcov"),
            "ctrl_vh_spd_fixage": fcol("vh_spd_fixage"),
            "ctrl_vh_spd_closure": fcol("vh_spd_closure"),
            "ctrl_vh_spd_nsamp": icol("vh_spd_nsamp"),
            "ctrl_vh_spd_resid": fcol("vh_spd_resid"),
            "ctrl_vh_spd_err": fcol("vh_spd_err"),
            "ctrl_vh_spd_pitch": fcol("vh_spd_pitch"),
            "ctrl_vh_spd_impulse": fcol("vh_spd_impulse"),
            "ctrl_vh_spd_exit": fcol("vh_spd_exit"),
            "ctrl_vh_spd_corner": icol("vh_spd_corner"),
            # -- bank observer + roll arbitration (Phase C/D) ---------------------------
            "ctrl_vh_body_gyro": vcol("vh_body_gyro", 3),  # raw body rates the observer used
            "ctrl_vh_obs_bank": fcol("vh_obs_bank"),
            "ctrl_vh_obs_src": s32("vh_obs_src"),
            "ctrl_vh_obs_age": fcol("vh_obs_age"),
            "ctrl_vh_obs_pnp_acc": icol("vh_obs_pnp_acc"),
            "ctrl_vh_obs_pnp_rej": icol("vh_obs_pnp_rej"),
            "ctrl_vh_obs_pnp_innov": fcol("vh_obs_pnp_innov"),
            "ctrl_vh_cmd_integral": fcol("vh_cmd_integral"),
            "ctrl_vh_roll_fb_src": s32("vh_roll_fb_src"),
            "ctrl_vh_roll_arb": s32("vh_roll_arb"),
            "ctrl_vh_shadow_degraded": icol("vh_shadow_degraded"),
            "ctrl_vh_active_degraded": icol("vh_active_degraded"),
            "ctrl_vh_bank_cap": fcol("vh_bank_cap"),
            "ctrl_vh_preturn_reject": s32("vh_preturn_reject"),
            # -- logical gate-1 handoff identity + geometry bank (Phase 1/3) -------------
            "ctrl_vh_ctrl_yaw": fcol("vh_ctrl_yaw"),  # control-attitude yaw used this tick
            "ctrl_vh_lg_prov": icol("vh_lg_prov"),
            "ctrl_vh_lg_raw": icol("vh_lg_raw"),
            "ctrl_vh_lg_prev": icol("vh_lg_prev"),
            "ctrl_vh_lg_streak": icol("vh_lg_streak"),
            "ctrl_vh_lg_az": fcol("vh_lg_az"),
            "ctrl_vh_lg_resid": fcol("vh_lg_resid"),
            "ctrl_vh_lg_gap": fcol("vh_lg_gap"),
            "ctrl_vh_lg_reacq": icol("vh_lg_reacq"),
            "ctrl_vh_lg_reason": s32("vh_lg_reason"),
            "ctrl_vh_bank_speed": fcol("vh_bank_speed"),
            "ctrl_vh_bank_raw": fcol("vh_bank_raw"),
            "ctrl_vh_bank": fcol("vh_bank"),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, np.int64),
            "session_id": np.asarray(self.session_id, np.int64),
            "imu_sensor_t": np.asarray(self._imu_sensor, np.int64),
            "imu_arrival_t": np.asarray(self._imu_arrival, np.int64),
            "imu_live": np.asarray(self._imu_live, np.int8),
            "imu": np.asarray(self._imu, np.float64).reshape(-1, 6),
            "fix_sensor_t": np.asarray(self._fix_sensor, np.int64),
            "fix_arrival_t": np.asarray(self._fix_arrival, np.int64),
            "fix_body": np.asarray(self._fix_body, np.float64).reshape(-1, 3),
            "fix_gate": np.asarray(self._fix_gate, np.int64),
            "fix_cov": np.asarray(self._fix_cov, np.float64).reshape(-1, 3, 3),
            "fix_mode": np.asarray(self._fix_mode, dtype="<U16"),
            **self._frame_columns(),
            **self._control_columns(),
        }
        np.savez(p, **arrays)  # type: ignore[arg-type]  # numpy stub mistypes **kwds unpack
        return p if p.suffix else p.with_suffix(".npz")


@dataclass
class ReplayResult:
    """Outputs of an offline arrival-order ESKF replay over a recorded run."""

    t: np.ndarray  # (K,) seconds from first prediction
    pos: np.ndarray  # (K,3) filter position (anchor frame)
    vel: np.ndarray  # (K,3) filter velocity (world)
    euler: np.ndarray  # (K,3) roll,pitch,yaw (rad)
    n_predict: int = 0
    n_dup: int = 0  # samples with sensor dt<=0 (duplicate/backwards)
    n_drop: int = 0  # samples after a > _DROP_DT_S sensor gap (dropouts)
    n_fix_seen: int = 0
    n_fix_accepted: int = 0
    reanchors: int = 0
    nis: list[float] = field(default_factory=list)
    vel_at_gate_change: dict[int, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def horizontal_speed(self) -> np.ndarray:
        return np.hypot(self.vel[:, 0], self.vel[:, 1]) if len(self.vel) else np.array([])


@dataclass
class GuidanceReplayResult:
    """Exact replay comparison for worker-owned gate-relative decisions."""

    n_rows: int = 0
    n_unique_decisions: int = 0
    n_compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        """Whether every unique recorded decision reproduced exactly."""
        return self.n_compared == self.n_unique_decisions and not self.mismatches


_GUIDANCE_REPLAY_KEYS = (
    "ctrl_sg_seq",
    "ctrl_sg_state_seq",
    "ctrl_sg_state_ns",
    "ctrl_sg_target_seq",
    "ctrl_sg_input_ns",
    "ctrl_sg_active",
    "ctrl_sg_anchor",
    "ctrl_sg_initialised",
    "ctrl_sg_fix_age",
    "ctrl_sg_transition_age",
    "ctrl_sg_mode",
    "ctrl_sg_status",
    "ctrl_sg_track",
    "ctrl_sg_set",
    "ctrl_sh_seq",
    "ctrl_sh_pub_ns",
    "ctrl_sh_p",
    "ctrl_sh_v",
    "ctrl_sh_euler",
    "ctrl_sh_cov",
    "ctrl_st_seq",
    "ctrl_st_valid",
    "ctrl_st_frame",
    "ctrl_st_sensor_ns",
    "ctrl_st_arrival_ns",
    "ctrl_st_active",
    "ctrl_st_track",
    "ctrl_st_race",
    "ctrl_st_hits",
    "ctrl_st_mode",
    "ctrl_st_bearing",
    "ctrl_st_los",
    "ctrl_st_slant",
    "ctrl_st_metric",
    "ctrl_st_conf",
    "ctrl_st_cov",
    "ctrl_st_turn_sign",
    "ctrl_st_disp",
    "ctrl_st_refused",
    "ctrl_st_wrong",
    "ctrl_st_oob",
)


def replay_guidance_decisions(data: dict[str, np.ndarray]) -> GuidanceReplayResult:
    """Recompute every unique worker decision from its exact recorded inputs.

    Control rows may sample the same worker publication more than once. Sequence IDs
    de-duplicate those rows and, critically, prevent replay from substituting the mission
    thread's phase/gate state for the worker state that actually produced the decision.
    """
    from drone_racing.deployments.vq2 import gate_relative as gr

    if not _supported_replay_schema(data):
        raise ValueError("guidance replay requires the current recorder schema")
    missing = [key for key in _GUIDANCE_REPLAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"recording lacks guidance replay fields {missing}")
    n_rows = len(data["ctrl_sg_seq"])
    inconsistent = [key for key in _GUIDANCE_REPLAY_KEYS if len(data[key]) != n_rows]
    if inconsistent:
        raise ValueError(f"guidance arrays have inconsistent lengths {inconsistent}")

    result = GuidanceReplayResult(n_rows=n_rows)
    seen: set[int] = set()
    policy = gr.default_policy()
    for i in range(n_rows):
        sequence = int(data["ctrl_sg_seq"][i])
        if sequence <= 0 or sequence in seen:
            continue
        seen.add(sequence)
        result.n_unique_decisions += 1

        # The coherent publication contract itself is part of replay validity.
        if int(data["ctrl_sg_state_seq"][i]) != int(data["ctrl_sh_seq"][i]):
            result.mismatches.append(f"row {i} decision {sequence}: state sequence mismatch")
            continue
        if int(data["ctrl_sg_target_seq"][i]) != int(data["ctrl_st_seq"][i]):
            result.mismatches.append(f"row {i} decision {sequence}: target sequence mismatch")
            continue
        if int(data["ctrl_sg_state_ns"][i]) != int(data["ctrl_sh_pub_ns"][i]):
            result.mismatches.append(f"row {i} decision {sequence}: state timestamp mismatch")
            continue

        target = gr.StructuredTarget(
            valid=bool(data["ctrl_st_valid"][i]),
            frame_id=int(data["ctrl_st_frame"][i]),
            frame_sensor_ns=int(data["ctrl_st_sensor_ns"][i]),
            arrival_ns=int(data["ctrl_st_arrival_ns"][i]),
            active_gate_at_receipt=int(data["ctrl_st_active"][i]),
            track_id=int(data["ctrl_st_track"][i]),
            race_gate_id=int(data["ctrl_st_race"][i]),
            track_hits=int(data["ctrl_st_hits"][i]),
            output_mode=str(data["ctrl_st_mode"][i]),
            bearing_rad=float(data["ctrl_st_bearing"][i]),
            los_body_unit=np.asarray(data["ctrl_st_los"][i], dtype=np.float64),
            slant_range_m=float(data["ctrl_st_slant"][i]),
            has_metric_pose=bool(data["ctrl_st_metric"][i]),
            confidence=float(data["ctrl_st_conf"][i]),
            cov_trace=float(data["ctrl_st_cov"][i]),
            expected_turn_sign=float(data["ctrl_st_turn_sign"][i]),
            disposition=str(data["ctrl_st_disp"][i]),
            association_refused=bool(data["ctrl_st_refused"][i]),
            wrong_turn=bool(data["ctrl_st_wrong"][i]),
            out_of_band=bool(data["ctrl_st_oob"][i]),
            sequence=int(data["ctrl_st_seq"][i]),
        )
        replayed = policy.step(
            now_ns=int(data["ctrl_sg_input_ns"][i]),
            active_gate=int(data["ctrl_sg_active"][i]),
            anchor_gate=int(data["ctrl_sg_anchor"][i]),
            initialised=bool(data["ctrl_sg_initialised"][i]),
            position_drone_rel_gate_ned_m=np.asarray(data["ctrl_sh_p"][i], dtype=np.float64),
            velocity_ned_mps=np.asarray(data["ctrl_sh_v"][i], dtype=np.float64),
            euler_rpy_rad=np.asarray(data["ctrl_sh_euler"][i], dtype=np.float64),
            cov_diag=np.asarray(data["ctrl_sh_cov"][i], dtype=np.float64),
            fix_age_s=float(data["ctrl_sg_fix_age"][i]),
            target=target,
            transition_age_s=float(data["ctrl_sg_transition_age"][i]),
        )
        reasons: list[str] = []
        if replayed.mode != str(data["ctrl_sg_mode"][i]):
            reasons.append("mode")
        if replayed.result.status.value != str(data["ctrl_sg_status"][i]):
            reasons.append("status")
        if replayed.track_id != int(data["ctrl_sg_track"][i]):
            reasons.append("track")
        expected_setpoint = np.asarray(data["ctrl_sg_set"][i], dtype=np.float64)
        if replayed.result.setpoint is None:
            if not np.all(np.isnan(expected_setpoint)):
                reasons.append("setpoint")
        else:
            actual_setpoint = np.array(
                [
                    replayed.result.setpoint.collective_thrust,
                    *replayed.result.setpoint.body_rates_rps,
                ],
                dtype=np.float64,
            )
            if not np.allclose(actual_setpoint, expected_setpoint, rtol=0.0, atol=1e-12):
                reasons.append("setpoint")
        if reasons:
            result.mismatches.append(f"row {i} decision {sequence}: {','.join(reasons)} mismatch")
        else:
            result.n_compared += 1
    return result


@dataclass
class AuthorityReplayResult:
    """Exact replay comparison for the stateful transition command mux."""

    n_rows: int = 0
    n_compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.n_rows == self.n_compared and not self.mismatches


_AUTHORITY_REPLAY_KEYS = _GUIDANCE_REPLAY_KEYS + (
    "ctrl_active_gate",
    "ctrl_sh_anchor",
    "ctrl_loop_released_ns",
    "ctrl_sg_decision_ns",
    "ctrl_auth_requested",
    "ctrl_auth_source",
    "ctrl_auth_reason",
    "ctrl_auth_entered",
    "ctrl_auth_exited",
    "ctrl_auth_elapsed",
    "ctrl_auth_legacy",
    "ctrl_auth_selected",
    "ctrl_auth_slew",
    "ctrl_failsafe",
)


def replay_authority_decisions(data: dict[str, np.ndarray]) -> AuthorityReplayResult:
    """Replay the command mux row-for-row from its exact recorded worker snapshots."""
    from drone_racing.deployments.vq2 import authority as auth
    from drone_racing.deployments.vq2 import gate_relative as gr
    from drone_racing.deployments.vq2.follower import RateThrustSetpoint

    if not _supported_replay_schema(data):
        raise ValueError("authority replay requires the current recorder schema")
    missing = [key for key in _AUTHORITY_REPLAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"recording lacks authority replay fields {missing}")
    n_rows = len(data["ctrl_auth_source"])
    inconsistent = [key for key in _AUTHORITY_REPLAY_KEYS if len(data[key]) != n_rows]
    if inconsistent:
        raise ValueError(f"authority arrays have inconsistent lengths {inconsistent}")
    enabled_values = np.asarray(data["ctrl_auth_requested"], dtype=np.int64)
    enabled = bool(enabled_values[0]) if n_rows else False
    if n_rows and np.any(enabled_values != enabled_values[0]):
        raise ValueError("authority requested state changed inside one attempt")

    mux = auth.TransitionAuthorityMux(enabled=enabled)
    result = AuthorityReplayResult(n_rows=n_rows)
    for i in range(n_rows):
        now_ns = int(data["ctrl_loop_released_ns"][i])
        target = gr.StructuredTarget(
            valid=bool(data["ctrl_st_valid"][i]),
            frame_id=int(data["ctrl_st_frame"][i]),
            frame_sensor_ns=int(data["ctrl_st_sensor_ns"][i]),
            arrival_ns=int(data["ctrl_st_arrival_ns"][i]),
            active_gate_at_receipt=int(data["ctrl_st_active"][i]),
            track_id=int(data["ctrl_st_track"][i]),
            race_gate_id=int(data["ctrl_st_race"][i]),
            track_hits=int(data["ctrl_st_hits"][i]),
            output_mode=str(data["ctrl_st_mode"][i]),
            bearing_rad=float(data["ctrl_st_bearing"][i]),
            los_body_unit=np.asarray(data["ctrl_st_los"][i], dtype=np.float64),
            slant_range_m=float(data["ctrl_st_slant"][i]),
            has_metric_pose=bool(data["ctrl_st_metric"][i]),
            confidence=float(data["ctrl_st_conf"][i]),
            cov_trace=float(data["ctrl_st_cov"][i]),
            expected_turn_sign=float(data["ctrl_st_turn_sign"][i]),
            disposition=str(data["ctrl_st_disp"][i]),
            association_refused=bool(data["ctrl_st_refused"][i]),
            wrong_turn=bool(data["ctrl_st_wrong"][i]),
            out_of_band=bool(data["ctrl_st_oob"][i]),
            sequence=int(data["ctrl_st_seq"][i]),
        )
        setpoint_values = np.asarray(data["ctrl_sg_set"][i], dtype=np.float64)
        setpoint = None
        if np.all(np.isfinite(setpoint_values)):
            setpoint = RateThrustSetpoint(
                collective_thrust=float(setpoint_values[0]),
                body_rates_rps=setpoint_values[1:4],
                requested_acceleration_ned_mps2=np.zeros(3),
                feasible_acceleration_ned_mps2=np.zeros(3),
                desired_rotation_body_to_ned=np.eye(3),
                saturation_flags=(),
            )
        approach_result = gr.GateApproachResult(
            status=gr.GateApproachStatus(str(data["ctrl_sg_status"][i])),
            setpoint=setpoint,
        )
        decision = gr.GuidanceDecision(
            decision_mono_ns=int(data["ctrl_sg_decision_ns"][i]),
            now_ns=int(data["ctrl_sg_input_ns"][i]),
            active_gate=int(data["ctrl_sg_active"][i]),
            anchor_gate=int(data["ctrl_sg_anchor"][i]),
            track_id=int(data["ctrl_sg_track"][i]),
            target_arrival_ns=int(data["ctrl_st_arrival_ns"][i]),
            mode=str(data["ctrl_sg_mode"][i]),
            result=approach_result,
            decision_sequence=int(data["ctrl_sg_seq"][i]),
            state_sequence=int(data["ctrl_sg_state_seq"][i]),
            state_mono_ns=int(data["ctrl_sg_state_ns"][i]),
            target_sequence=int(data["ctrl_sg_target_seq"][i]),
            input_mono_ns=int(data["ctrl_sg_input_ns"][i]),
            initialised=bool(data["ctrl_sg_initialised"][i]),
            fix_age_s=float(data["ctrl_sg_fix_age"][i]),
            transition_age_s=float(data["ctrl_sg_transition_age"][i]),
        )
        elapsed_s = float(data["ctrl_auth_elapsed"][i])
        transition_start_ns = (
            int(round(now_ns - elapsed_s * 1e9)) if math.isfinite(elapsed_s) else 0
        )
        replayed = mux.select(
            now_ns=now_ns,
            transition_start_ns=transition_start_ns,
            active_gate=int(data["ctrl_active_gate"][i]),
            anchor_gate=int(data["ctrl_sh_anchor"][i]),
            state_sequence=int(data["ctrl_sh_seq"][i]),
            target=target,
            decision=decision,
            legacy=tuple(float(value) for value in data["ctrl_auth_legacy"][i]),  # type: ignore[arg-type]
            abort_reason=str(data["ctrl_failsafe"][i]),
        )
        reasons: list[str] = []
        if replayed.source != str(data["ctrl_auth_source"][i]):
            reasons.append("source")
        if replayed.reason != str(data["ctrl_auth_reason"][i]):
            reasons.append("reason")
        if replayed.entered != bool(data["ctrl_auth_entered"][i]):
            reasons.append("entered")
        if replayed.exited != bool(data["ctrl_auth_exited"][i]):
            reasons.append("exited")
        if replayed.slew_limited != bool(data["ctrl_auth_slew"][i]):
            reasons.append("slew")
        if not np.allclose(
            replayed.selected,
            np.asarray(data["ctrl_auth_selected"][i], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            reasons.append("selected")
        if reasons:
            result.mismatches.append(f"row {i}: {','.join(reasons)} mismatch")
        else:
            result.n_compared += 1
    return result


@dataclass(frozen=True, slots=True)
class HandoffReplayResult:
    """Recorded provisional-to-active identity result at the first gate-0→1 advance."""

    transition_row: int
    provisional_rows: int
    provisional_track_id: int
    provisional_track_hits: int
    active_track_id: int
    active_track_hits: int

    @property
    def identity_match(self) -> bool:
        return self.provisional_track_id >= 0 and self.provisional_track_id == self.active_track_id

    @property
    def confirmation_preserved(self) -> bool:
        """Same id crosses the boundary without discarding its visual confirmation."""
        return self.identity_match and self.active_track_hits >= self.provisional_track_hits


_HANDOFF_REPLAY_KEYS = (
    "ctrl_active_gate",
    "ctrl_nt_valid",
    "ctrl_nt_track",
    "ctrl_nt_hits",
    "ctrl_st_valid",
    "ctrl_st_track",
    "ctrl_st_hits",
)


def replay_handoff_identity(data: dict[str, np.ndarray]) -> HandoffReplayResult:
    """Recover the first recorded provisional→active gate-1 identity handoff."""
    if int(data.get("schema_version", -1)) < 8:
        raise ValueError("handoff identity replay requires recorder schema v8")
    missing = [key for key in _HANDOFF_REPLAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"recording lacks handoff replay fields {missing}")
    n_rows = len(data["ctrl_active_gate"])
    inconsistent = [key for key in _HANDOFF_REPLAY_KEYS if len(data[key]) != n_rows]
    if inconsistent:
        raise ValueError(f"handoff arrays have inconsistent lengths {inconsistent}")
    gate = np.asarray(data["ctrl_active_gate"], dtype=np.int64)
    changes = np.flatnonzero((gate[1:] == 1) & (gate[:-1] == 0)) + 1
    if not len(changes):
        raise ValueError("recording has no gate-0→1 transition")
    transition = int(changes[0])
    rows = np.arange(n_rows)
    before = np.flatnonzero(np.asarray(data["ctrl_nt_valid"], dtype=bool) & (rows < transition))
    after = np.flatnonzero(np.asarray(data["ctrl_st_valid"], dtype=bool) & (rows >= transition))
    provisional = int(data["ctrl_nt_track"][before[-1]]) if len(before) else -1
    provisional_hits = int(data["ctrl_nt_hits"][before[-1]]) if len(before) else 0
    active = int(data["ctrl_st_track"][after[0]]) if len(after) else -1
    active_hits = int(data["ctrl_st_hits"][after[0]]) if len(after) else 0
    return HandoffReplayResult(
        transition_row=transition,
        provisional_rows=len(before),
        provisional_track_id=provisional,
        provisional_track_hits=provisional_hits,
        active_track_id=active,
        active_track_hits=active_hits,
    )


@dataclass
class VisualReplayResult:
    """Exact row-for-row replay comparison for the stateful visual handoff controller."""

    n_rows: int = 0
    n_visual_rows: int = 0  # rows where the recorded controller engaged (source == visual)
    n_compared: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.n_rows > 0 and self.n_rows == self.n_compared and not self.mismatches


def _structured_from_row(data: dict[str, np.ndarray], i: int, prefix: str):
    """Rebuild a :class:`StructuredTarget` from ``ctrl_{st|nt}_*`` columns at row ``i``."""
    from drone_racing.deployments.vq2 import gate_relative as gr

    def col(name: str) -> np.ndarray:
        return data[f"ctrl_{prefix}_{name}"][i]

    return gr.StructuredTarget(
        valid=bool(col("valid")),
        frame_id=int(col("frame")),
        frame_sensor_ns=int(col("sensor_ns")),
        arrival_ns=int(col("arrival_ns")),
        active_gate_at_receipt=int(col("active")),
        track_id=int(col("track")),
        race_gate_id=int(col("race")),
        track_hits=int(col("hits")),
        output_mode=str(col("mode")),
        bearing_rad=float(col("bearing")),
        los_body_unit=np.asarray(col("los"), dtype=np.float64),
        slant_range_m=float(col("slant")),
        has_metric_pose=bool(col("metric")),
        confidence=float(col("conf")),
        cov_trace=float(col("cov")),
        expected_turn_sign=float(col("turn_sign")),
        disposition=str(col("disp")),
        association_refused=bool(col("refused")),
        wrong_turn=bool(col("wrong")),
        out_of_band=bool(col("oob")),
        sequence=int(col("seq")),
        cx_px=float(col("cx")),
        cy_px=float(col("cy")),
        area_px2=float(col("area")),
        bbox_xywh=(-1, -1, -1, -1),
        visible_fraction=float(col("visfrac")),
        vision_roll_rad=float(col("vroll")),
    )


_VISUAL_REPLAY_KEYS = (
    "ctrl_loop_released_ns",
    "ctrl_active_gate",
    "ctrl_sh_anchor",
    "ctrl_sh_euler",
    "ctrl_sh_cov",
    "ctrl_sh_pub_ns",
    "ctrl_failsafe",
    "ctrl_vh_state",
    "ctrl_vh_source",
    "ctrl_vh_legacy",
    "ctrl_vh_selected",
    "ctrl_st_valid",
    "ctrl_st_cx",
    "ctrl_nt_valid",
    "ctrl_nt_cx",
)


def replay_visual_decisions(data: dict[str, np.ndarray]) -> VisualReplayResult:
    """Replay the visual handoff controller row-for-row from its exact recorded inputs.

    A fresh controller is stepped once per recorded control row (in order) with the active
    + provisional structured targets, ESKF roll channel, state age, and legacy command that
    were recorded that loop, and its selected command / state / source are compared to the
    recorded values. This is the exact-replay guarantee for the Training-only visual path.
    """
    from drone_racing.deployments.vq2 import visual_handoff as vh

    if int(data.get("schema_version", -1)) < 9:
        raise ValueError("visual replay requires recorder schema v9")
    missing = [key for key in _VISUAL_REPLAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"recording lacks visual replay fields {missing}")
    n_rows = len(data["ctrl_vh_state"])
    inconsistent = [key for key in _VISUAL_REPLAY_KEYS if len(data[key]) != n_rows]
    if inconsistent:
        raise ValueError(f"visual arrays have inconsistent lengths {inconsistent}")
    states = np.asarray([str(s) for s in data["ctrl_vh_state"]])
    if not np.any(states != ""):
        raise ValueError("recording has no visual handoff rows (not a visual-handoff run)")

    controller = vh.VisualHandoffController()
    result = VisualReplayResult(n_rows=n_rows)
    for i in range(n_rows):
        released_ns = int(data["ctrl_loop_released_ns"][i])
        cov = np.asarray(data["ctrl_sh_cov"][i], dtype=np.float64)
        euler = np.asarray(data["ctrl_sh_euler"][i], dtype=np.float64)
        pub_ns = int(data["ctrl_sh_pub_ns"][i])
        age = (released_ns - pub_ns) * 1e-9 if pub_ns else 1e9

        def _spd(key: str, default: float = float("nan"), _i: int = i) -> float:
            return float(data[key][_i]) if key in data else default

        inp = vh.VisualHandoffInput(
            now_ns=released_ns,
            training=True,
            flag_enabled=True,
            active_gate=int(data["ctrl_active_gate"][i]),
            anchor_gate=int(data["ctrl_sh_anchor"][i]),
            active_target=_structured_from_row(data, i, "st"),
            provisional_target=_structured_from_row(data, i, "nt"),
            eskf_roll_rad=float(euler[0]),
            eskf_roll_cov_rad2=float(cov[6]) if cov.size > 6 else float("nan"),
            eskf_att_cov_trace_rad2=float(np.sum(cov[6:9])) if cov.size >= 9 else float("nan"),
            state_age_s=age,
            legacy_command=tuple(float(v) for v in data["ctrl_vh_legacy"][i]),  # type: ignore[arg-type]
            abort_reason=str(data["ctrl_failsafe"][i]),
            # exact recorded exit-speed inputs (absent → NaN, which fails closed, matching a
            # recording made before the gate-0 speed control existed)
            eskf_speed_mps=_spd("ctrl_vh_spd_eskf"),
            eskf_vel_cov_trace=_spd("ctrl_vh_spd_velcov"),
            eskf_fix_age_s=_spd("ctrl_vh_spd_fixage"),
            eskf_initialised=bool(data["ctrl_sh_init"][i]),
            # exact recorded control-attitude yaw. New runs record it directly; older runs
            # predate the dedicated column, so fall back to the control-attitude euler yaw
            # (``ctrl_att[:,2]``) — the identical signal the live tick read from ``fl.att``.
            control_yaw_rad=(
                _spd("ctrl_vh_ctrl_yaw")
                if "ctrl_vh_ctrl_yaw" in data and math.isfinite(float(data["ctrl_vh_ctrl_yaw"][i]))
                else float(np.asarray(data["ctrl_att"][i], dtype=np.float64)[2])
            ),
            # bank-observer inputs: control roll/pitch come from the recorded control attitude;
            # the exact per-tick body gyro is recorded (absent in pre-observer runs → NaN, which
            # keeps the observer un-established and replays the pre-observer path exactly).
            control_roll_rad=float(np.asarray(data["ctrl_att"][i], dtype=np.float64)[0]),
            control_pitch_rad=float(np.asarray(data["ctrl_att"][i], dtype=np.float64)[1]),
            body_gyro_rps=(
                tuple(float(v) for v in data["ctrl_vh_body_gyro"][i])  # type: ignore[arg-type]
                if "ctrl_vh_body_gyro" in data
                else (float("nan"),) * 3
            ),
        )
        decision = controller.step(inp)
        if decision.status.source == "visual":
            result.n_visual_rows += 1
        reasons: list[str] = []
        if decision.state.value != str(data["ctrl_vh_state"][i]):
            reasons.append("state")
        if decision.status.source != str(data["ctrl_vh_source"][i]):
            reasons.append("source")
        if not np.allclose(
            decision.selected,
            np.asarray(data["ctrl_vh_selected"][i], dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            reasons.append("selected")
        if reasons:
            result.mismatches.append(f"row {i}: {','.join(reasons)} mismatch")
        else:
            result.n_compared += 1
    return result


_REQUIRED_KEYS = (
    "imu_sensor_t",
    "imu_arrival_t",
    "imu_live",
    "imu",
    "fix_sensor_t",
    "fix_arrival_t",
    "fix_body",
    "fix_gate",
    "fix_cov",
)


def replay_arrival_order(
    data: dict[str, np.ndarray],
    *,
    gyro_sign: tuple[float, float, float] | np.ndarray = GYRO_SIGN,
    **eskf_kwargs: float,
) -> ReplayResult:
    """Re-run the ESKF over a recorded run, merging IMU and fixes by **arrival time**.

    This reproduces current live behaviour (fuse-in-receive-order), so it INCLUDES camera
    transport/decode latency — it is NOT exact capture-time delayed fusion. IMU
    integration ``dt`` still comes from the IMU sensor clock (``imu_sensor_t``, µs).

    Rejects recordings missing the common arrival clock or with non-monotonic arrival
    timestamps (a corrupt/cross-session file) rather than silently merging raw clocks.
    """
    if "schema_version" not in data:
        raise ValueError("recording has no schema_version; refusing to replay")
    ver = int(data["schema_version"])
    if not _supported_replay_schema(data):
        raise ValueError(
            f"recording schema v{ver} outside supported v{_REPLAY_SCHEMA_MIN}..v{SCHEMA_VERSION}; "
            "refusing to replay"
        )
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"recording lacks arrival-clock fields {missing}; cannot replay")
    n_imu = len(data["imu_sensor_t"])
    n_fix = len(data["fix_sensor_t"])
    if not (
        len(data["imu_arrival_t"]) == n_imu
        and len(data["imu"]) == n_imu
        and len(data["imu_live"]) == n_imu
    ):
        raise ValueError("IMU arrays have inconsistent lengths; refusing to replay")
    if not (
        len(data["fix_arrival_t"]) == n_fix
        and len(data["fix_body"]) == n_fix
        and len(data["fix_gate"]) == n_fix
        and len(data["fix_cov"]) == n_fix
    ):
        raise ValueError("fix arrays have inconsistent lengths; refusing to replay")
    imu_arr = data["imu_arrival_t"].astype(np.int64)
    fix_arr = data["fix_arrival_t"].astype(np.int64)
    if np.any(np.diff(imu_arr) < 0) or np.any(np.diff(fix_arr) < 0):
        raise ValueError("arrival timestamps are non-monotonic; refusing to replay")

    imu_sensor = data["imu_sensor_t"].astype(np.int64)
    imu = data["imu"]
    imu_live = data["imu_live"]
    fix_body, fix_gate, fix_cov = data["fix_body"], data["fix_gate"], data["fix_cov"]

    # merged, ARRIVAL-ordered event stream: (arrival_ns, kind 0=imu/1=fix, index).
    # kind is a tiebreak so an IMU sample sorts before a fix at the same arrival stamp.
    events = [(int(imu_arr[i]), 0, i) for i in range(len(imu_arr))]
    events += [(int(fix_arr[i]), 1, i) for i in range(len(fix_arr))]
    events.sort(key=lambda e: (e[0], e[1]))

    sign = np.asarray(gyro_sign, dtype=np.float64)
    if sign.shape != (3,) or not np.all(np.isfinite(sign)):
        raise ValueError("gyro_sign must contain three finite axis signs")
    kf = ESKF(**eskf_kwargs)
    active: int | None = None
    prev_us: int | None = None
    res = ReplayResult(
        t=np.empty(0), pos=np.empty((0, 3)), vel=np.empty((0, 3)), euler=np.empty((0, 3))
    )
    ts, ps, vs, es = [], [], [], []
    t0: int | None = None

    for _arr, kind, idx in events:
        if kind == 0:
            acc, gyro = imu[idx, :3], imu[idx, 3:]
            su = int(imu_sensor[idx])
            # Mirror the live worker EXACTLY: the first sample only seeds prev_us (no
            # predict); attitude initialises on the first VALID-dt predict; ground gravity
            # aiding (not-live) is applied after that predict.
            if prev_us is not None:
                dt = (su - prev_us) * 1e-6  # dt from the IMU SENSOR clock (µs)
                if dt <= 0.0:
                    res.n_dup += 1
                elif dt > _DROP_DT_S:
                    res.n_drop += 1
                else:
                    kf.predict(acc, gyro * sign, dt)  # sensor-to-FRD convention
                    res.n_predict += 1
                    if not bool(imu_live[idx]):
                        kf.update_gravity(acc)
                    t0 = su if t0 is None else t0
                    ts.append((su - t0) * 1e-6)
                    ps.append(kf.p.copy())
                    vs.append(kf.v.copy())
                    es.append(np.array(kf.euler))
            prev_us = su
        else:
            res.n_fix_seen += 1
            body = fix_body[idx]
            gate = int(fix_gate[idx])
            cov = fix_cov[idx]
            cov_arg = None if not np.all(np.isfinite(cov)) else cov
            if gate != active:
                kf.reanchor(body, cov_arg)
                res.reanchors += 1
                active = gate
                res.vel_at_gate_change[gate] = tuple(float(x) for x in kf.v)  # type: ignore[assignment]
            else:
                info = kf.update_gate(body, cov_arg)
                res.nis.append(info.nis)
                if info.accepted:
                    res.n_fix_accepted += 1

    res.t = np.asarray(ts)
    res.pos = np.asarray(ps).reshape(-1, 3)
    res.vel = np.asarray(vs).reshape(-1, 3)
    res.euler = np.asarray(es).reshape(-1, 3)
    return res


def event_timeline(data: dict[str, np.ndarray]) -> list[tuple[int, str, int]]:
    """Merge every monotonic-clocked stream into one ordered timeline for alignment checks.

    All of IMU, fix, frame, and control rows carry the SAME ``time.monotonic_ns()`` receipt
    clock (unlike the per-sensor epochs used for integration ``dt``), so they can be sorted
    against each other directly. Returns ``(mono_ns, stream, index)`` sorted by time, with a
    stable per-stream tiebreak so a deterministic order survives equal timestamps. Rows with
    a negative/sentinel stamp (e.g. an uninitialised value) are skipped.
    """
    streams = {
        "imu": ("imu_arrival_t", 0),
        "fix": ("fix_arrival_t", 1),
        "frame": ("frm_arrival_ns", 2),
        "ctrl": ("ctrl_mono_ns", 3),
    }
    ev: list[tuple[int, int, str, int]] = []
    for name, (key, tie) in streams.items():
        arr = data.get(key)
        if arr is None:
            continue
        for i, t in enumerate(np.asarray(arr).ravel()):
            if int(t) >= 0:
                ev.append((int(t), tie, name, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    return [(t, name, i) for t, _tie, name, i in ev]


def load_run(path: str | Path) -> dict[str, np.ndarray]:
    """Load a recorded ``.npz`` run into a plain dict of arrays."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
