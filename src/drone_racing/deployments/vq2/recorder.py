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
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone_racing.deployments.vq2.eskf import ESKF

_DROP_DT_S = 0.1  # a gap larger than this between IMU samples is a dropout, not a step
SCHEMA_VERSION = 3  # bump when the .npz field layout changes; replay validates it
_MAX_PLAUSIBLE_LATENCY_S = 2.0  # camera transport+decode latency beyond this = bad clock map


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

    def add_frame(self, **fields: object) -> None:
        """Record one per-frame diagnostic row (see :func:`_frame_columns` for fields)."""
        self._frames.append(fields)

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


def replay_arrival_order(data: dict[str, np.ndarray], **eskf_kwargs: float) -> ReplayResult:
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
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f"recording schema v{ver} != supported v{SCHEMA_VERSION}; refusing to replay"
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
                    kf.predict(acc, gyro, dt)
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


def load_run(path: str | Path) -> dict[str, np.ndarray]:
    """Load a recorded ``.npz`` run into a plain dict of arrays."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
