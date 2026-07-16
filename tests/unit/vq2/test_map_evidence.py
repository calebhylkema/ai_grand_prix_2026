"""Deterministic tests for the offline VQ2 map-evidence promotion gate."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from drone_racing.deployments.vq2.course_map import GateLeg, MapStatus, load_course_map
from drone_racing.deployments.vq2.map_evidence import (
    MIN_METRIC_PAIRS_PER_RUN,
    CandidateObservation,
    EvidenceInputError,
    GatePairEvidence,
    MapPromotionReport,
    RequirementResult,
    aggregate_pair_evidence,
    audit_promotion,
    join_recording_frames,
    pair_gate_observations,
    write_candidate_map,
)

if TYPE_CHECKING:
    from pathlib import Path

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

IDENTITY_COV: Mat3 = ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01))
LEG = GateLeg(0, 1, 1, (3.0, 16.0), (math.radians(25.0), math.radians(110.0)))


def _write_bundle(
    root: Path,
    *,
    frame_ids: list[int],
    image_ids: list[int] | None = None,
    row_gates: list[int] | None = None,
    filename_gates: dict[int, int] | None = None,
    session_id: int = 7,
    schema: int = 3,
) -> tuple[Path, Path]:
    recording = root / "run_test.npz"
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True)
    n = len(frame_ids)
    gates = np.asarray(row_gates if row_gates is not None else [0] * n, np.int64)
    np.savez(
        recording,
        schema_version=np.asarray(schema, np.int64),
        session_id=np.asarray(session_id, np.int64),
        frm_frame_id=np.asarray(frame_ids, np.int64),
        frm_cam_ns=np.arange(n, dtype=np.int64) + 100,
        frm_arrival_ns=np.arange(n, dtype=np.int64) + 1000,
        frm_active_gate=gates,
        frm_shadow_att=np.zeros((n, 3), dtype=np.float64),
    )
    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    row_gate_by_id = {frame_id: int(gate) for frame_id, gate in zip(frame_ids, gates, strict=True)}
    for frame_id in frame_ids if image_ids is None else image_ids:
        gate = (filename_gates or {}).get(frame_id, row_gate_by_id.get(frame_id, 0))
        assert cv2.imwrite(str(frame_dir / f"f{frame_id:04d}_g{gate}_t-1_none.png"), blank)
    return recording, frame_dir


def _obs(
    *,
    frame: int,
    active_gate: int,
    track: int,
    selected: bool,
    pose: Vec3 | None,
    run: str = "run-a",
    attitude: Vec3 = (0.0, 0.0, 0.0),
    mode: str = "position_only",
    normal: Vec3 | None = None,
    fingerprint: str = "sha256:same",
    covariance: Mat3 | None = IDENTITY_COV,
) -> CandidateObservation:
    return CandidateObservation(
        run_id=run,
        perception_fingerprint=fingerprint,
        frame_id=frame,
        camera_time_ns=frame * 10,
        arrival_time_ns=frame * 10 + 1,
        active_gate=active_gate,
        track_id=track,
        race_gate_id=active_gate if selected else None,
        selected=selected,
        bbox=(0, 0, 10, 10),
        bearing_rad=0.2 if track == 20 else 0.0,
        confidence=0.9,
        output_mode=mode,
        solver="inner4",
        reproj_rms_px=0.5,
        ambiguity_margin_px=1.0,
        pose_cam_m=pose,
        pose_cov_cam_m2=None if pose is None else covariance,
        normal_cam=normal,
        shadow_att_rad=attitude,
    )


def _handoff_observations(
    *,
    attitude: Vec3 = (0.0, 0.0, 0.0),
    normal_mode: str = "position_only",
) -> list[CandidateObservation]:
    rows: list[CandidateObservation] = []
    for frame in range(1, 6):
        rows.extend(
            [
                _obs(
                    frame=frame,
                    active_gate=0,
                    track=10,
                    selected=True,
                    pose=(0.0, 0.0, 5.0),
                    attitude=attitude,
                    mode=normal_mode,
                    normal=(1.0, 0.0, 0.0),
                ),
                _obs(
                    frame=frame,
                    active_gate=0,
                    track=20,
                    selected=False,
                    pose=(1.0, 0.0, 5.0),
                    attitude=attitude,
                    mode=normal_mode,
                    normal=(1.0, 0.0, 0.0),
                ),
                # A high-confidence/nearby imposter has no temporal handoff and must
                # never enter the pair evidence merely because of its image ordering.
                _obs(
                    frame=frame,
                    active_gate=0,
                    track=30,
                    selected=False,
                    pose=(0.2, 0.0, 2.0),
                    attitude=attitude,
                ),
            ]
        )
    for frame in range(6, 11):
        rows.extend(
            [
                _obs(
                    frame=frame,
                    active_gate=1,
                    track=10,
                    selected=False,
                    pose=(0.0, 0.0, 5.0),
                    attitude=attitude,
                    mode=normal_mode,
                    normal=(1.0, 0.0, 0.0),
                ),
                _obs(
                    frame=frame,
                    active_gate=1,
                    track=20,
                    selected=True,
                    pose=(1.0, 0.0, 5.0),
                    attitude=attitude,
                    mode=normal_mode,
                    normal=(1.0, 0.0, 0.0),
                ),
            ]
        )
    return rows


def _pair(
    run: str,
    frame: int,
    delta: Vec3,
    *,
    covariance: Mat3 = IDENTITY_COV,
    normal: Vec3 | None = (1.0, 0.0, 0.0),
    fingerprint: str = "sha256:same",
) -> GatePairEvidence:
    return GatePairEvidence(
        run_id=run,
        perception_fingerprint=fingerprint,
        frame_id=frame,
        from_gate=0,
        to_gate=1,
        from_track_id=10,
        to_track_id=20,
        delta_body_m=delta,
        delta_map_m=delta,
        covariance_map_m2=covariance,
        from_normal_map=(1.0, 0.0, 0.0) if normal is not None else None,
        to_normal_map=normal,
        identity_provenance="temporal tracker handoff",
    )


def _passing_pairs() -> list[GatePairEvidence]:
    pairs: list[GatePairEvidence] = []
    offsets = {"run-a": -0.04, "run-b": 0.0, "run-c": 0.04}
    covariance: Mat3 = ((0.0004, 0.0, 0.0), (0.0, 0.0004, 0.0), (0.0, 0.0, 0.0004))
    for run, offset in offsets.items():
        for frame in range(MIN_METRIC_PAIRS_PER_RUN):
            jitter = (frame - 2) * 0.002
            pairs.append(
                _pair(
                    run,
                    frame,
                    (10.0 + offset + jitter, 0.0, 0.0),
                    covariance=covariance,
                )
            )
    return pairs


def _requirement(report: MapPromotionReport, name: str) -> RequirementResult:
    return next(item for item in report.requirements if item.name == name)


def test_join_recording_frames_accepts_exact_schema_v3_bundle(tmp_path: Path) -> None:
    recording, frames = _write_bundle(
        tmp_path, frame_ids=list(range(20)), image_ids=list(range(19))
    )
    run, joined = join_recording_frames(recording, frames, fingerprint="sha256:test")
    assert run.schema_version == 3
    assert run.session_id == 7
    assert run.perception_fingerprint == "sha256:test"
    assert len(joined) == 19  # exactly 95% is allowed and remains visible


@pytest.mark.parametrize("schema", [2, 4])
def test_join_recording_frames_rejects_wrong_schema(tmp_path: Path, schema: int) -> None:
    recording, frames = _write_bundle(tmp_path, frame_ids=[1], schema=schema)
    with pytest.raises(EvidenceInputError, match="unsupported"):
        join_recording_frames(recording, frames, fingerprint="sha256:test")


def test_join_recording_frames_rejects_duplicate_recording_ids(tmp_path: Path) -> None:
    recording, frames = _write_bundle(tmp_path, frame_ids=[1, 1])
    with pytest.raises(EvidenceInputError, match="duplicate frm_frame_id"):
        join_recording_frames(recording, frames, fingerprint="sha256:test")


def test_join_recording_frames_rejects_filename_gate_mismatch(tmp_path: Path) -> None:
    recording, frames = _write_bundle(
        tmp_path,
        frame_ids=[1],
        row_gates=[0],
        filename_gates={1: 1},
    )
    with pytest.raises(EvidenceInputError, match="active gate mismatch"):
        join_recording_frames(recording, frames, fingerprint="sha256:test")


def test_temporal_handoff_selects_physical_tracks_and_rejects_imposter() -> None:
    pairs, requirements = pair_gate_observations(
        _handoff_observations(), from_gate=0, to_gate=1, cam_to_body=np.eye(3)
    )
    assert all(result.passed for result in requirements)
    assert len(pairs) == 10
    assert {(pair.from_track_id, pair.to_track_id) for pair in pairs} == {(10, 20)}
    assert all(pair.to_track_id != 30 for pair in pairs)


def test_temporal_handoff_fails_when_target_was_not_visible_before_advance() -> None:
    observations = [
        row for row in _handoff_observations() if not (row.active_gate == 0 and row.track_id == 20)
    ]
    pairs, requirements = pair_gate_observations(
        observations, from_gate=0, to_gate=1, cam_to_body=np.eye(3)
    )
    assert pairs == ()
    assert not next(r for r in requirements if r.name == "temporal_track_handoff").passed


def test_full_yaw_rotation_is_used_for_pair_delta() -> None:
    pairs, _ = pair_gate_observations(
        _handoff_observations(attitude=(0.0, 0.0, math.pi / 2)),
        from_gate=0,
        to_gate=1,
        cam_to_body=np.eye(3),
    )
    assert np.allclose(pairs[0].delta_body_m, (1.0, 0.0, 0.0), atol=1e-9)
    assert np.allclose(pairs[0].delta_map_m, (0.0, 1.0, 0.0), atol=1e-9)


def test_missing_covariance_cannot_form_metric_pair() -> None:
    observations = _handoff_observations()
    observations = [
        row
        if row.track_id != 20
        else CandidateObservation(
            **{
                **{name: getattr(row, name) for name in row.__dataclass_fields__},
                "pose_cov_cam_m2": None,
            }
        )
        for row in observations
    ]
    pairs, requirements = pair_gate_observations(
        observations, from_gate=0, to_gate=1, cam_to_body=np.eye(3)
    )
    assert pairs == ()
    assert not next(r for r in requirements if r.name == "metric_pair_count").passed


def test_position_only_never_contributes_a_traversal_normal() -> None:
    pairs, _ = pair_gate_observations(
        _handoff_observations(normal_mode="position_only"),
        from_gate=0,
        to_gate=1,
        cam_to_body=np.eye(3),
    )
    assert pairs
    assert all(pair.to_normal_map is None for pair in pairs)


def test_aggregation_uses_one_median_per_run_and_keeps_uncertainty_floor() -> None:
    covariance: Mat3 = ((0.09, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.01))
    pairs = [
        *[_pair("run-a", frame, (10.0, 2.0, 0.0), covariance=covariance) for frame in range(50)],
        _pair("run-a", 99, (1000.0, 1000.0, 1000.0), covariance=covariance),
        *[_pair("run-b", frame, (10.0, 2.0, 0.0), covariance=covariance) for frame in range(5)],
        *[_pair("run-c", frame, (10.0, 2.0, 0.0), covariance=covariance) for frame in range(5)],
    ]
    aggregate = aggregate_pair_evidence(pairs)
    assert aggregate is not None
    assert len(aggregate.run_estimates) == 3
    assert aggregate.center_ned_m == pytest.approx((10.0, 2.0, 0.0))
    assert aggregate.position_std_m == pytest.approx((0.3, 0.2, 0.1))


def test_many_frames_from_one_run_do_not_inflate_independent_run_count(tmp_path: Path) -> None:
    pairs = [_pair("only-run", frame, (10.0, 0.0, 0.0)) for frame in range(100)]
    _, report = audit_promotion(
        pairs,
        batch_id="one-run",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
    )
    assert not _requirement(report, "independent_runs").passed
    assert "1 independent" in _requirement(report, "independent_runs").detail


def test_outlier_and_leave_one_out_instability_are_reported_not_removed(tmp_path: Path) -> None:
    pairs: list[GatePairEvidence] = []
    for run, x in (("run-a", 9.4), ("run-b", 10.0), ("run-c", 10.6)):
        pairs.extend(_pair(run, frame, (x, 0.0, 0.0)) for frame in range(5))
    _, report = audit_promotion(
        pairs,
        batch_id="unstable",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
    )
    assert not _requirement(report, "run_residuals").passed
    assert not _requirement(report, "leave_one_run_out").passed
    assert report.aggregate is not None
    assert len(report.aggregate.run_estimates) == 3  # no silent outlier deletion


def test_mixed_perception_fingerprints_fail_promotion(tmp_path: Path) -> None:
    pairs = _passing_pairs()
    pairs[-1] = _pair("run-c", 4, (10.04, 0.0, 0.0), fingerprint="sha256:different")
    _, report = audit_promotion(
        pairs,
        batch_id="mixed",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
    )
    assert not _requirement(report, "single_perception_fingerprint").passed
    assert not report.ready


def test_candidate_toml_round_trip_never_promotes_status(tmp_path: Path) -> None:
    candidate, report = audit_promotion(
        _passing_pairs(),
        batch_id="passing",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
    )
    assert report.ready
    output = write_candidate_map(candidate, tmp_path / "candidate.toml")
    loaded = load_course_map(output)
    assert loaded.status is MapStatus.CANDIDATE
    with pytest.raises(ValueError, match="not flight-ready"):
        load_course_map(output, require_flight_ready=True)


def test_deterministic_three_run_fixture_meets_all_requirements(tmp_path: Path) -> None:
    identity = (
        RequirementResult("run-a:temporal_track_handoff", True, "proven"),
        RequirementResult("run-b:temporal_track_handoff", True, "proven"),
        RequirementResult("run-c:temporal_track_handoff", True, "proven"),
    )
    candidate, report = audit_promotion(
        _passing_pairs(),
        batch_id="passing",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
        identity_requirements=identity,
    )
    assert candidate.status is MapStatus.CANDIDATE
    assert report.ready, [result for result in report.requirements if not result.passed]
    assert all(requirement.passed for requirement in report.requirements)


def test_historical_map_and_latest_shaped_single_run_are_not_promotable(tmp_path: Path) -> None:
    historical = load_course_map("vq2_historical_v0")
    with pytest.raises(ValueError, match="not flight-ready"):
        historical.require_flight_ready()
    # Mirrors the current live evidence shape: one run and position-only observations,
    # even if it had enough co-visible pairs.  It must remain candidate/not ready.
    latest = [
        _pair("run_20260713_080452", frame, (10.0, 6.0, 0.0), normal=None) for frame in range(8)
    ]
    _, report = audit_promotion(
        latest,
        batch_id="latest-shaped",
        from_gate=0,
        to_gate=1,
        leg=LEG,
        candidate_map_path=tmp_path / "candidate.toml",
    )
    assert not report.ready
    assert not _requirement(report, "independent_runs").passed
    assert not _requirement(report, "physical_gate_normals").passed
