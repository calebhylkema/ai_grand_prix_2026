"""Offline, fail-closed course-map evidence pipeline for VQ2.

This module turns independently recorded schema-v3 shadow runs into a *candidate*
course-map artifact.  It deliberately has no flight-runtime imports and no command
authority.  A successful audit means the evidence is strong enough for human review;
it never writes ``status = "validated"`` and never promotes a map into the mission.

The important separation is:

* frames are reprocessed by the current structured perception stack;
* visual identity comes from a temporal gate-track handoff, never nearest/second-nearest;
* every gate-pair delta is rotated camera -> body -> local NED with the recorded full
  roll/pitch/yaw attitude;
* frames are reduced to one median per run before independent runs are aggregated;
* every promotion requirement is emitted as a machine-readable pass/fail result.

The legacy :mod:`mapping` miner remains historical evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from drone_racing.deployments.vq2.course_map import (
    CourseMap,
    GateLeg,
    GateSpec,
    MapEvidence,
    MapFrame,
    MapStatus,
    load_course_map,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


SUPPORTED_RECORDING_SCHEMA = 3
MIN_JOIN_FRACTION = 0.95
MIN_INDEPENDENT_RUNS = 3
MIN_METRIC_PAIRS_PER_RUN = 5
MAX_RUN_RESIDUAL_M = 0.50
MAX_LEAVE_ONE_OUT_SHIFT_M = 0.25
MAX_NORMAL_P95_DEG = 10.0
MAX_NORMAL_LEAVE_ONE_OUT_DEG = 5.0

_FRAME_RE = re.compile(
    r"^f(?P<frame>\d+)_g(?P<gate>-?\d+)_t(?P<track>-?\d+)_(?P<disp>.+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)
_METRIC_MODES = frozenset({"position_only", "full_pose"})


class EvidenceInputError(ValueError):
    """A recording/frame bundle is corrupt, mismatched, or unsupported."""


@dataclass(frozen=True, slots=True)
class EvidenceRun:
    """One independent recording and its joined raw-frame directory."""

    run_id: str
    recording_path: Path
    frame_dir: Path
    session_id: int
    schema_version: int
    perception_fingerprint: str


@dataclass(frozen=True, slots=True)
class JoinedFrame:
    """A schema-v3 diagnostic row joined one-to-one to a raw image."""

    run_id: str
    frame_id: int
    image_path: Path
    camera_time_ns: int
    arrival_time_ns: int
    active_gate: int
    shadow_att_rad: Vec3


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """One reprocessed candidate with persistent visual identity."""

    run_id: str
    perception_fingerprint: str
    frame_id: int
    camera_time_ns: int
    arrival_time_ns: int
    active_gate: int
    track_id: int
    race_gate_id: int | None
    selected: bool
    bbox: tuple[int, int, int, int]
    bearing_rad: float
    confidence: float
    output_mode: str
    solver: str | None
    reproj_rms_px: float | None
    ambiguity_margin_px: float | None
    pose_cam_m: Vec3 | None
    pose_cov_cam_m2: Mat3 | None
    # Reserved for a future public PnP orientation output.  ``up_body`` is not a
    # traversal normal and is intentionally never substituted here.
    normal_cam: Vec3 | None
    shadow_att_rad: Vec3

    @property
    def metric(self) -> bool:
        """Whether this observation has a quality-gated metric position."""
        return (
            self.output_mode in _METRIC_MODES
            and self.pose_cam_m is not None
            and self.pose_cov_cam_m2 is not None
        )


@dataclass(frozen=True, slots=True)
class GatePairEvidence:
    """One simultaneous, identity-proven observation of two physical gates."""

    run_id: str
    perception_fingerprint: str
    frame_id: int
    from_gate: int
    to_gate: int
    from_track_id: int
    to_track_id: int
    delta_body_m: Vec3
    delta_map_m: Vec3
    covariance_map_m2: Mat3
    from_normal_map: Vec3 | None
    to_normal_map: Vec3 | None
    identity_provenance: str


@dataclass(frozen=True, slots=True)
class RunEstimate:
    """One statistically independent run-level estimate."""

    run_id: str
    pair_count: int
    median_delta_m: Vec3
    propagated_std_m: Vec3
    to_normal_map: Vec3 | None


@dataclass(frozen=True, slots=True)
class AggregateEstimate:
    """Cross-run robust gate-pair estimate and diagnostics."""

    center_ned_m: Vec3
    position_std_m: Vec3
    normal_ned: Vec3 | None
    run_estimates: tuple[RunEstimate, ...]
    run_residual_m: tuple[float, ...]
    leave_one_out_shift_m: tuple[float, ...]
    normal_residual_deg: tuple[float, ...]
    normal_leave_one_out_deg: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RequirementResult:
    """One explicit map-promotion acceptance requirement."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class MapPromotionReport:
    """Complete, machine-readable promotion audit."""

    batch_id: str
    from_gate: int
    to_gate: int
    candidate_map: str
    evidence_runs: tuple[EvidenceRun, ...]
    inherited_constraints: dict[str, Any]
    requirements: tuple[RequirementResult, ...]
    aggregate: AggregateEstimate | None

    @property
    def ready(self) -> bool:
        """True only when every explicit requirement passes."""
        return bool(self.requirements) and all(item.passed for item in self.requirements)


def _finite_vec(value: object, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise EvidenceInputError(f"{name} must be a finite ({length},) vector")
    return tuple(float(x) for x in array)


def _vec3(value: object) -> Vec3:
    parsed = _finite_vec(value, 3, "vector")
    return parsed[0], parsed[1], parsed[2]


def _mat3(value: object) -> Mat3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise EvidenceInputError("matrix must be finite and (3,3)")
    return _vec3(array[0]), _vec3(array[1]), _vec3(array[2])


def _array_dict(path: Path) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as raw:
            return {name: np.asarray(raw[name]).copy() for name in raw.files}
    except (OSError, ValueError) as exc:
        raise EvidenceInputError(f"cannot read recording {path}: {exc}") from exc


def perception_fingerprint(source_path: Path | None = None) -> str:
    """Return a reproducible SHA-256 fingerprint of the exact perception source."""
    if source_path is None:
        from drone_racing.deployments.vq2 import perception

        source_path = Path(perception.__file__).resolve()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def join_recording_frames(
    recording_path: str | Path,
    frame_dir: str | Path,
    *,
    fingerprint: str | None = None,
) -> tuple[EvidenceRun, tuple[JoinedFrame, ...]]:
    """Validate schema v3 and join diagnostic rows to image files by frame id.

    Duplicate recording IDs, duplicate image IDs, filename/row active-gate mismatch,
    unmatched extra images, and less than 95% row coverage fail closed.  Missing images
    within the 5% allowance remain visible in the join fraction rather than being
    fabricated.
    """
    recording = Path(recording_path).resolve()
    frames = Path(frame_dir).resolve()
    if not recording.is_file():
        raise EvidenceInputError(f"recording does not exist: {recording}")
    if not frames.is_dir():
        raise EvidenceInputError(f"frame directory does not exist: {frames}")
    arrays = _array_dict(recording)
    required = {
        "schema_version",
        "session_id",
        "frm_frame_id",
        "frm_cam_ns",
        "frm_arrival_ns",
        "frm_active_gate",
        "frm_shadow_att",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise EvidenceInputError(f"recording is missing schema-v3 fields: {missing}")
    schema = int(np.asarray(arrays["schema_version"]).reshape(()))
    if schema != SUPPORTED_RECORDING_SCHEMA:
        raise EvidenceInputError(
            f"recording schema v{schema} is unsupported; expected v{SUPPORTED_RECORDING_SCHEMA}"
        )
    frame_ids = np.asarray(arrays["frm_frame_id"], dtype=np.int64)
    n_rows = len(frame_ids)
    if n_rows == 0:
        raise EvidenceInputError("recording contains no frame diagnostic rows")
    if len(set(int(x) for x in frame_ids)) != n_rows:
        raise EvidenceInputError("recording contains duplicate frm_frame_id values")
    column_names = ("frm_cam_ns", "frm_arrival_ns", "frm_active_gate")
    for name in column_names:
        if len(arrays[name]) != n_rows:
            raise EvidenceInputError(f"{name} length does not match frm_frame_id")
    attitudes = np.asarray(arrays["frm_shadow_att"], dtype=np.float64)
    if attitudes.shape != (n_rows, 3) or not np.all(np.isfinite(attitudes)):
        raise EvidenceInputError("frm_shadow_att must be a finite (N,3) array")

    images: dict[int, tuple[Path, int]] = {}
    for path in sorted(frames.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = _FRAME_RE.match(path.name)
        if match is None:
            raise EvidenceInputError(f"unrecognized frame filename: {path.name}")
        frame_id = int(match.group("frame"))
        if frame_id in images:
            raise EvidenceInputError(f"duplicate image frame id {frame_id}")
        images[frame_id] = (path, int(match.group("gate")))
    if not images:
        raise EvidenceInputError(f"no raw frame images in {frames}")
    recording_ids = {int(x) for x in frame_ids}
    extra_ids = sorted(set(images) - recording_ids)
    if extra_ids:
        raise EvidenceInputError(f"images have no recording row: {extra_ids[:10]}")
    matched = recording_ids & set(images)
    join_fraction = len(matched) / n_rows
    if join_fraction < MIN_JOIN_FRACTION:
        raise EvidenceInputError(
            f"frame join coverage {join_fraction:.1%} is below {MIN_JOIN_FRACTION:.0%}"
        )

    index = {int(frame_id): i for i, frame_id in enumerate(frame_ids)}
    run_id = recording.stem
    joined: list[JoinedFrame] = []
    active = np.asarray(arrays["frm_active_gate"], dtype=np.int64)
    cam_ns = np.asarray(arrays["frm_cam_ns"], dtype=np.int64)
    arrival_ns = np.asarray(arrays["frm_arrival_ns"], dtype=np.int64)
    for frame_id in sorted(matched, key=lambda value: (int(arrival_ns[index[value]]), value)):
        row = index[frame_id]
        path, filename_gate = images[frame_id]
        row_gate = int(active[row])
        if filename_gate != row_gate:
            raise EvidenceInputError(
                f"frame {frame_id} active gate mismatch: filename={filename_gate}, row={row_gate}"
            )
        joined.append(
            JoinedFrame(
                run_id=run_id,
                frame_id=frame_id,
                image_path=path,
                camera_time_ns=int(cam_ns[row]),
                arrival_time_ns=int(arrival_ns[row]),
                active_gate=row_gate,
                shadow_att_rad=_vec3(attitudes[row]),
            )
        )
    if any(
        b.arrival_time_ns <= a.arrival_time_ns for a, b in zip(joined, joined[1:], strict=False)
    ):
        raise EvidenceInputError("joined frame arrival timestamps are not strictly increasing")
    run = EvidenceRun(
        run_id=run_id,
        recording_path=recording,
        frame_dir=frames,
        session_id=int(np.asarray(arrays["session_id"]).reshape(())),
        schema_version=schema,
        perception_fingerprint=fingerprint or perception_fingerprint(),
    )
    return run, tuple(joined)


def _body_to_world(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """ZYX body-FRD to local-NED rotation using recorded roll, pitch, and yaw."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def extract_candidate_observations(
    run: EvidenceRun,
    frames: Sequence[JoinedFrame],
    *,
    from_gate: int,
    to_gate: int,
    leg: GateLeg,
    overlay_dir: Path | None = None,
) -> tuple[CandidateObservation, ...]:
    """Reprocess raw frames through structured detection, PnP, and tracking."""
    from drone_racing.deployments.vq2 import perception

    tracker = perception.GateTracker()
    output: list[CandidateObservation] = []
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    for joined in frames:
        image = cv2.imread(str(joined.image_path))
        if image is None:
            raise EvidenceInputError(f"OpenCV could not decode {joined.image_path}")
        candidates = perception.detect_candidates(image)
        for obs in candidates:
            obs.frame_id = joined.frame_id
            obs.timestamp_ns = joined.camera_time_ns
            perception.solve_gate_pose(obs)
        turn_sign = float(leg.expected_turn_sign) if joined.active_gate == to_gate else None
        band = leg.slant_range_m if joined.active_gate == to_gate else None
        selected = tracker.step(
            candidates,
            active_gate_index=joined.active_gate,
            expected_turn_sign=turn_sign,
            range_band=band,
        )
        selected_track = None if selected is None else selected.track_id
        for obs in candidates:
            pose_cov: Mat3 | None = None
            if obs.pose_cov is not None:
                covariance = np.asarray(obs.pose_cov, dtype=np.float64)
                if covariance.shape == (3, 3) and np.all(np.isfinite(covariance)):
                    pose_cov = _mat3(covariance)
            normal_cam: Vec3 | None = None
            public_normal = getattr(obs, "normal_cam", None)
            if obs.output_mode is perception.OutputMode.FULL_POSE and public_normal is not None:
                parsed = _vec3(public_normal)
                norm = float(np.linalg.norm(parsed))
                if norm > 1e-9:
                    normal_cam = _vec3(np.asarray(parsed) / norm)
            output.append(
                CandidateObservation(
                    run_id=run.run_id,
                    perception_fingerprint=run.perception_fingerprint,
                    frame_id=joined.frame_id,
                    camera_time_ns=joined.camera_time_ns,
                    arrival_time_ns=joined.arrival_time_ns,
                    active_gate=joined.active_gate,
                    track_id=-1 if obs.track_id is None else obs.track_id,
                    race_gate_id=obs.race_gate_id,
                    selected=selected_track is not None and obs.track_id == selected_track,
                    bbox=obs.bbox,
                    bearing_rad=float(obs.bearing),
                    confidence=float(obs.confidence),
                    output_mode=obs.output_mode.value,
                    solver=obs.solver,
                    reproj_rms_px=obs.reproj_rms,
                    ambiguity_margin_px=obs.ambiguity_margin,
                    pose_cam_m=obs.pose_cam,
                    pose_cov_cam_m2=pose_cov,
                    normal_cam=normal_cam,
                    shadow_att_rad=joined.shadow_att_rad,
                )
            )
        if overlay_dir is not None:
            overlay = image.copy()
            for obs in candidates:
                x, y, width, height = obs.bbox
                is_selected = selected_track is not None and obs.track_id == selected_track
                color = (0, 0, 255) if is_selected else (0, 200, 255)
                cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
                label = f"t{obs.track_id} {obs.output_mode.value}"
                cv2.putText(
                    overlay,
                    label,
                    (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            cv2.imwrite(str(overlay_dir / joined.image_path.name), overlay)
    return tuple(output)


def _single_selected_track(
    observations: Sequence[CandidateObservation], gate_id: int
) -> tuple[int | None, str]:
    counts: dict[int, int] = defaultdict(int)
    for obs in observations:
        if obs.active_gate == gate_id and obs.selected and obs.track_id >= 0:
            counts[obs.track_id] += 1
    if not counts:
        return None, f"no selected track for gate {gate_id}"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1:
        return None, f"gate {gate_id} selected multiple tracks: {dict(ordered)}"
    return ordered[0][0], f"gate {gate_id} track {ordered[0][0]} selected {ordered[0][1]} frames"


def pair_gate_observations(
    observations: Sequence[CandidateObservation],
    *,
    from_gate: int,
    to_gate: int,
    cam_to_body: NDArray[np.float64] | None = None,
) -> tuple[tuple[GatePairEvidence, ...], tuple[RequirementResult, ...]]:
    """Prove temporal handoff and form simultaneous metric gate-pair deltas."""
    from drone_racing.deployments.vq2 import perception

    rotation_cb = perception.CAM_TO_BODY if cam_to_body is None else np.asarray(cam_to_body, float)
    if rotation_cb.shape != (3, 3):
        raise ValueError("cam_to_body must be (3,3)")
    source_track, source_detail = _single_selected_track(observations, from_gate)
    target_track, target_detail = _single_selected_track(observations, to_gate)
    identity_results = [
        RequirementResult("from_gate_single_track", source_track is not None, source_detail),
        RequirementResult("to_gate_single_track", target_track is not None, target_detail),
    ]
    if source_track is None or target_track is None:
        return (), tuple(identity_results)
    if source_track == target_track:
        identity_results.append(
            RequirementResult(
                "distinct_gate_tracks", False, "both race gates resolved to one track"
            )
        )
        return (), tuple(identity_results)
    identity_results.append(
        RequirementResult("distinct_gate_tracks", True, f"tracks {source_track} and {target_track}")
    )
    handoff_frames = [
        obs.frame_id
        for obs in observations
        if obs.active_gate == from_gate and obs.track_id == target_track and not obs.selected
    ]
    handoff_ok = bool(handoff_frames)
    identity_results.append(
        RequirementResult(
            "temporal_track_handoff",
            handoff_ok,
            (
                f"target track {target_track} visible before race advance in frames "
                f"{handoff_frames[:10]}"
                if handoff_ok
                else f"target track {target_track} was never observed while gate {from_gate} was active"
            ),
        )
    )
    if not handoff_ok:
        return (), tuple(identity_results)

    by_frame: dict[int, dict[int, CandidateObservation]] = defaultdict(dict)
    for obs in observations:
        if obs.track_id in {source_track, target_track}:
            by_frame[obs.frame_id][obs.track_id] = obs
    pairs: list[GatePairEvidence] = []
    for frame_id, tracks in sorted(by_frame.items()):
        if source_track not in tracks or target_track not in tracks:
            continue
        source, target = tracks[source_track], tracks[target_track]
        if not source.metric or not target.metric:
            continue
        assert source.pose_cam_m is not None and target.pose_cam_m is not None
        source_cam = np.asarray(source.pose_cam_m, dtype=np.float64)
        target_cam = np.asarray(target.pose_cam_m, dtype=np.float64)
        delta_body = rotation_cb @ (target_cam - source_cam)
        r_map_body = _body_to_world(*source.shadow_att_rad)
        delta_map = r_map_body @ delta_body
        assert source.pose_cov_cam_m2 is not None and target.pose_cov_cam_m2 is not None
        source_cov = np.asarray(source.pose_cov_cam_m2, dtype=np.float64)
        target_cov = np.asarray(target.pose_cov_cam_m2, dtype=np.float64)
        covariance_map = (
            r_map_body @ rotation_cb @ (source_cov + target_cov) @ rotation_cb.T @ r_map_body.T
        )

        pairs.append(
            GatePairEvidence(
                run_id=source.run_id,
                perception_fingerprint=source.perception_fingerprint,
                frame_id=frame_id,
                from_gate=from_gate,
                to_gate=to_gate,
                from_track_id=source_track,
                to_track_id=target_track,
                delta_body_m=_vec3(delta_body),
                delta_map_m=_vec3(delta_map),
                covariance_map_m2=_mat3(covariance_map),
                from_normal_map=_rotate_normal(source, r_map_body, rotation_cb),
                to_normal_map=_rotate_normal(target, r_map_body, rotation_cb),
                identity_provenance=(
                    f"tracker handoff: t{target_track} co-visible during gate {from_gate}, "
                    f"then selected for gate {to_gate}"
                ),
            )
        )
    identity_results.append(
        RequirementResult(
            "metric_pair_count",
            len(pairs) >= MIN_METRIC_PAIRS_PER_RUN,
            f"{len(pairs)} simultaneous metric pairs (minimum {MIN_METRIC_PAIRS_PER_RUN})",
        )
    )
    return tuple(pairs), tuple(identity_results)


def _rotate_normal(
    observation: CandidateObservation,
    body_to_map: NDArray[np.float64],
    cam_to_body: NDArray[np.float64],
) -> Vec3 | None:
    """Rotate an explicitly physical FULL_POSE normal into the map frame."""
    if observation.output_mode != "full_pose" or observation.normal_cam is None:
        return None
    vector = body_to_map @ cam_to_body @ np.asarray(observation.normal_cam, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-9 else _vec3(vector / norm)


def _unit_mean(vectors: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        return None
    total = np.sum(vectors / np.linalg.norm(vectors, axis=1, keepdims=True), axis=0)
    norm = float(np.linalg.norm(total))
    return None if norm < 1e-9 else total / norm


def _angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def aggregate_pair_evidence(pairs: Sequence[GatePairEvidence]) -> AggregateEstimate | None:
    """Reduce frames within runs first, then robustly aggregate independent runs."""
    if not pairs:
        return None
    grouped: dict[str, list[GatePairEvidence]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.run_id].append(pair)
    estimates: list[RunEstimate] = []
    for run_id, run_pairs in sorted(grouped.items()):
        deltas = np.asarray([pair.delta_map_m for pair in run_pairs], dtype=np.float64)
        covariances = np.asarray([pair.covariance_map_m2 for pair in run_pairs], dtype=np.float64)
        normal_values = [pair.to_normal_map for pair in run_pairs if pair.to_normal_map is not None]
        normal: Vec3 | None = None
        if normal_values:
            mean_normal = _unit_mean(np.asarray(normal_values, dtype=np.float64))
            if mean_normal is not None:
                normal = _vec3(mean_normal)
        estimates.append(
            RunEstimate(
                run_id=run_id,
                pair_count=len(run_pairs),
                median_delta_m=_vec3(np.median(deltas, axis=0)),
                propagated_std_m=_vec3(
                    np.sqrt(
                        np.maximum(
                            np.median(np.diagonal(covariances, axis1=1, axis2=2), axis=0),
                            0.0,
                        )
                    )
                ),
                to_normal_map=normal,
            )
        )
    run_centers = np.asarray([estimate.median_delta_m for estimate in estimates])
    center = np.median(run_centers, axis=0)
    sample_std = np.std(run_centers, axis=0, ddof=1) if len(estimates) > 1 else np.zeros(3)
    mad_std = 1.4826 * np.median(np.abs(run_centers - center), axis=0)
    propagated = np.median(np.asarray([e.propagated_std_m for e in estimates]), axis=0)
    position_std = np.maximum.reduce([sample_std, mad_std, propagated])
    residuals = np.linalg.norm(run_centers - center, axis=1)
    loo_shifts: list[float] = []
    for index in range(len(estimates)):
        remaining = np.delete(run_centers, index, axis=0)
        loo_center = center if len(remaining) == 0 else np.median(remaining, axis=0)
        loo_shifts.append(float(np.linalg.norm(loo_center - center)))

    run_normals = [e.to_normal_map for e in estimates if e.to_normal_map is not None]
    aggregate_normal: NDArray[np.float64] | None = None
    normal_residuals: list[float] = []
    normal_loo: list[float] = []
    if len(run_normals) == len(estimates):
        normal_array = np.asarray(run_normals, dtype=np.float64)
        aggregate_normal = _unit_mean(normal_array)
        if aggregate_normal is not None:
            normal_residuals = [_angle_deg(row, aggregate_normal) for row in normal_array]
            for index in range(len(normal_array)):
                remaining = np.delete(normal_array, index, axis=0)
                loo = aggregate_normal if len(remaining) == 0 else _unit_mean(remaining)
                normal_loo.append(180.0 if loo is None else _angle_deg(loo, aggregate_normal))
    return AggregateEstimate(
        center_ned_m=_vec3(center),
        position_std_m=_vec3(position_std),
        normal_ned=None if aggregate_normal is None else _vec3(aggregate_normal),
        run_estimates=tuple(estimates),
        run_residual_m=tuple(float(x) for x in residuals),
        leave_one_out_shift_m=tuple(loo_shifts),
        normal_residual_deg=tuple(normal_residuals),
        normal_leave_one_out_deg=tuple(normal_loo),
    )


def build_candidate_map(
    aggregate: AggregateEstimate | None,
    pairs: Sequence[GatePairEvidence],
    *,
    from_gate: int,
    to_gate: int,
    leg: GateLeg,
    course_id: str,
) -> CourseMap:
    """Build a typed candidate map; this function can never emit validated status."""
    fingerprints = sorted({pair.perception_fingerprint for pair in pairs})
    if len(fingerprints) == 1:
        revision = fingerprints[0]
    elif fingerprints:
        revision = "mixed-perception-fingerprints"
    else:
        revision = "no-eligible-pair-evidence"
    run_ids = tuple(sorted({pair.run_id for pair in pairs}))
    if aggregate is None:
        center, std, normal = (0.0, 0.0, 0.0), (99.0, 99.0, 99.0), None
    else:
        center, std, normal = aggregate.center_ned_m, aggregate.position_std_m, aggregate.normal_ned
    evidence = MapEvidence(revision, run_ids, len(run_ids), len(pairs))
    gates = (
        GateSpec(
            gate_id=from_gate,
            center_ned_m=(0.0, 0.0, 0.0),
            position_std_m=(0.0, 0.0, 0.0),
            inner_size_m=(1.5, 1.5),
            normal_ned=(1.0, 0.0, 0.0),
            evidence=evidence,
        ),
        GateSpec(
            gate_id=to_gate,
            center_ned_m=center,
            position_std_m=std,
            inner_size_m=(1.5, 1.5),
            normal_ned=normal,
            evidence=evidence,
        ),
    )
    return CourseMap(
        schema_version=1,
        course_id=course_id,
        status=MapStatus.CANDIDATE,
        frame=MapFrame.GATE0_LOCAL_NED,
        units="si",
        drone_radius_m=0.14,
        tracking_margin_m=0.20,
        gates=gates,
        legs=(leg,),
    )


def _reference_feasible(candidate: CourseMap) -> tuple[bool, str]:
    if any(gate.normal_ned is None for gate in candidate.gates):
        return False, "candidate has an unresolved traversal normal"
    try:
        from drone_racing.deployments.vq2.reference import build_course_reference
        from drone_racing.utils.types import PlannerConfig

        reviewed = replace(candidate, status=MapStatus.VALIDATED)
        reference = build_course_reference(reviewed, PlannerConfig())
    except (ValueError, RuntimeError) as exc:
        return False, str(exc)
    clearances = [passage.clearance_m for passage in reference.gate_passages]
    return True, f"3-sigma aperture/direction check passed; min clearance={min(clearances):.3f} m"


def audit_promotion(
    pairs: Sequence[GatePairEvidence],
    *,
    batch_id: str,
    from_gate: int,
    to_gate: int,
    leg: GateLeg,
    candidate_map_path: Path,
    identity_requirements: Sequence[RequirementResult] = (),
    evidence_runs: Sequence[EvidenceRun] = (),
) -> tuple[CourseMap, MapPromotionReport]:
    """Aggregate evidence and evaluate every non-negotiable promotion requirement."""
    aggregate = aggregate_pair_evidence(pairs)
    candidate = build_candidate_map(
        aggregate,
        pairs,
        from_gate=from_gate,
        to_gate=to_gate,
        leg=leg,
        course_id=f"vq2-{from_gate}-{to_gate}-{batch_id}",
    )
    grouped: dict[str, list[GatePairEvidence]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.run_id].append(pair)
    run_count = len(grouped)
    audit_fingerprints = {run.perception_fingerprint for run in evidence_runs} or {
        pair.perception_fingerprint for pair in pairs
    }
    requirements = list(identity_requirements)
    requirements.extend(
        [
            RequirementResult(
                "independent_runs",
                run_count >= MIN_INDEPENDENT_RUNS,
                f"{run_count} independent runs (minimum {MIN_INDEPENDENT_RUNS})",
            ),
            RequirementResult(
                "metric_pairs_per_run",
                run_count >= MIN_INDEPENDENT_RUNS
                and all(len(values) >= MIN_METRIC_PAIRS_PER_RUN for values in grouped.values()),
                ", ".join(f"{name}={len(values)}" for name, values in sorted(grouped.items()))
                or "no pair evidence",
            ),
            RequirementResult(
                "single_perception_fingerprint",
                len(audit_fingerprints) == 1 and bool(audit_fingerprints),
                str(sorted(audit_fingerprints)),
            ),
        ]
    )
    if aggregate is None:
        requirements.extend(
            [
                RequirementResult("run_residuals", False, "no aggregate"),
                RequirementResult("leave_one_run_out", False, "no aggregate"),
                RequirementResult("physical_gate_normals", False, "no aggregate"),
                RequirementResult("normal_residuals", False, "no aggregate"),
            ]
        )
    else:
        requirements.extend(
            [
                RequirementResult(
                    "run_residuals",
                    bool(aggregate.run_residual_m)
                    and max(aggregate.run_residual_m) <= MAX_RUN_RESIDUAL_M,
                    f"max={max(aggregate.run_residual_m, default=math.inf):.3f} m; limit={MAX_RUN_RESIDUAL_M:.2f} m",
                ),
                RequirementResult(
                    "leave_one_run_out",
                    bool(aggregate.leave_one_out_shift_m)
                    and max(aggregate.leave_one_out_shift_m) <= MAX_LEAVE_ONE_OUT_SHIFT_M,
                    f"max={max(aggregate.leave_one_out_shift_m, default=math.inf):.3f} m; limit={MAX_LEAVE_ONE_OUT_SHIFT_M:.2f} m",
                ),
                RequirementResult(
                    "physical_gate_normals",
                    aggregate.normal_ned is not None
                    and len(aggregate.normal_residual_deg) == run_count
                    and run_count >= MIN_INDEPENDENT_RUNS,
                    f"normals from {len(aggregate.normal_residual_deg)}/{run_count} runs; FULL_POSE only",
                ),
                RequirementResult(
                    "normal_residuals",
                    bool(aggregate.normal_residual_deg)
                    and float(np.percentile(aggregate.normal_residual_deg, 95))
                    <= MAX_NORMAL_P95_DEG
                    and max(aggregate.normal_leave_one_out_deg, default=math.inf)
                    <= MAX_NORMAL_LEAVE_ONE_OUT_DEG,
                    (
                        f"p95={float(np.percentile(aggregate.normal_residual_deg, 95)):.2f} deg, "
                        f"LOO max={max(aggregate.normal_leave_one_out_deg, default=math.inf):.2f} deg"
                        if aggregate.normal_residual_deg
                        else "normal unavailable"
                    ),
                ),
            ]
        )
    feasible, feasible_detail = _reference_feasible(candidate)
    requirements.append(RequirementResult("reference_3sigma_feasible", feasible, feasible_detail))
    inherited = {
        "source": "existing reviewed gate-association contract; not new geometry evidence",
        "expected_turn_sign": leg.expected_turn_sign,
        "slant_range_m": list(leg.slant_range_m),
        "turn_window_deg": [math.degrees(x) for x in leg.turn_window_rad],
    }
    report = MapPromotionReport(
        batch_id=batch_id,
        from_gate=from_gate,
        to_gate=to_gate,
        candidate_map=str(candidate_map_path),
        evidence_runs=tuple(evidence_runs),
        inherited_constraints=inherited,
        requirements=tuple(requirements),
        aggregate=aggregate,
    )
    return candidate, report


def _toml_array(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(value):.9g}" for value in values) + "]"


def write_candidate_map(course: CourseMap, path: str | Path) -> Path:
    """Write deterministic TOML while enforcing candidate-only status."""
    if course.status is not MapStatus.CANDIDATE:
        raise ValueError("map evidence may only write status='candidate'")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by the offline VQ2 map-evidence audit. Human review required.",
        f"schema_version = {course.schema_version}",
        f"course_id = {json.dumps(course.course_id)}",
        'status = "candidate"',
        f'frame = "{course.frame.value}"',
        f'units = "{course.units}"',
        f"drone_radius_m = {course.drone_radius_m:.9g}",
        f"tracking_margin_m = {course.tracking_margin_m:.9g}",
        "",
    ]
    for gate in course.gates:
        lines.extend(
            [
                "[[gates]]",
                f"id = {gate.gate_id}",
                f"center_ned_m = {_toml_array(gate.center_ned_m)}",
                f"position_std_m = {_toml_array(gate.position_std_m)}",
                f"inner_size_m = {_toml_array(gate.inner_size_m)}",
            ]
        )
        if gate.normal_ned is not None:
            lines.append(f"normal_ned = {_toml_array(gate.normal_ned)}")
        lines.extend(
            [
                "",
                "[gates.evidence]",
                f"perception_revision = {json.dumps(gate.evidence.perception_revision)}",
                "source_runs = ["
                + ", ".join(json.dumps(run) for run in gate.evidence.source_runs)
                + "]",
                f"independent_runs = {gate.evidence.independent_runs}",
                f"observation_count = {gate.evidence.observation_count}",
                "",
            ]
        )
    for leg in course.legs:
        lines.extend(
            [
                "[[legs]]",
                f"from_gate = {leg.from_gate}",
                f"to_gate = {leg.to_gate}",
                f"expected_turn_sign = {leg.expected_turn_sign}",
                f"slant_range_m = {_toml_array(leg.slant_range_m)}",
                f"turn_window_deg = {_toml_array(math.degrees(x) for x in leg.turn_window_rad)}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    # Parse our own output immediately: malformed candidate artifacts never escape.
    load_course_map(output)
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(asdict(row)), sort_keys=True) + "\n")


def _default_leg(from_gate: int, to_gate: int) -> GateLeg:
    historical = load_course_map("vq2_historical_v0")
    try:
        return historical.leg(from_gate, to_gate)
    except KeyError as exc:
        raise EvidenceInputError(
            f"no reviewed inherited association constraints for {from_gate}->{to_gate}"
        ) from exc


def run_audit(
    bundles: Sequence[tuple[Path, Path]],
    *,
    from_gate: int,
    to_gate: int,
    output_root: Path,
) -> tuple[Path, MapPromotionReport]:
    """Execute the complete offline audit and write all review artifacts."""
    if from_gate != 0 or to_gate != 1:
        raise EvidenceInputError(
            "the current gate0_local_ned evidence pipeline supports only gate 0->1"
        )
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / batch_id
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_root = output_dir / "review_overlays"
    overlay_root.mkdir()
    leg = _default_leg(from_gate, to_gate)
    all_observations: list[CandidateObservation] = []
    all_pairs: list[GatePairEvidence] = []
    identity: list[RequirementResult] = []
    seen_sessions: set[int] = set()
    seen_run_ids: set[str] = set()
    evidence_runs: list[EvidenceRun] = []
    for recording, frames in bundles:
        run, joined = join_recording_frames(recording, frames)
        if run.session_id in seen_sessions:
            raise EvidenceInputError(
                f"duplicate session_id {run.session_id}; runs are not independent"
            )
        if run.run_id in seen_run_ids:
            raise EvidenceInputError(
                f"duplicate run id {run.run_id}; use uniquely named recordings"
            )
        seen_sessions.add(run.session_id)
        seen_run_ids.add(run.run_id)
        evidence_runs.append(run)
        observations = extract_candidate_observations(
            run,
            joined,
            from_gate=from_gate,
            to_gate=to_gate,
            leg=leg,
            overlay_dir=overlay_root / run.run_id,
        )
        pairs, results = pair_gate_observations(
            observations,
            from_gate=from_gate,
            to_gate=to_gate,
        )
        all_observations.extend(observations)
        all_pairs.extend(pairs)
        identity.extend(
            RequirementResult(f"{run.run_id}:{item.name}", item.passed, item.detail)
            for item in results
        )
    candidate_path = output_dir / "candidate_map.toml"
    candidate, report = audit_promotion(
        all_pairs,
        batch_id=batch_id,
        from_gate=from_gate,
        to_gate=to_gate,
        leg=leg,
        candidate_map_path=candidate_path,
        identity_requirements=identity,
        evidence_runs=evidence_runs,
    )
    write_candidate_map(candidate, candidate_path)
    _write_jsonl(output_dir / "observations.jsonl", all_observations)
    _write_jsonl(output_dir / "pair_evidence.jsonl", all_pairs)
    (output_dir / "promotion_report.json").write_text(
        json.dumps(_jsonable(asdict(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir, report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("RECORDING_NPZ", "FRAMES_DIR"),
        required=True,
        help="independent schema-v3 recording and matching raw-frame directory; repeatable",
    )
    parser.add_argument("--from-gate", type=int, default=0)
    parser.add_argument("--to-gate", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".local/datasets/vq2/map_evidence"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Evidence failure is exit 0; corrupt input is exit 2."""
    args = _parse_args(argv)
    try:
        output, report = run_audit(
            [(Path(recording), Path(frames)) for recording, frames in args.run],
            from_gate=args.from_gate,
            to_gate=args.to_gate,
            output_root=args.output_root,
        )
    except EvidenceInputError as exc:
        print(f"INVALID_INPUT: {exc}", file=sys.stderr)
        return 2
    verdict = "PROMOTION_REQUIREMENTS_MET" if report.ready else "CANDIDATE_NOT_READY"
    print(f"{verdict}: {output}")
    for requirement in report.requirements:
        mark = "PASS" if requirement.passed else "FAIL"
        print(f"  [{mark}] {requirement.name}: {requirement.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
