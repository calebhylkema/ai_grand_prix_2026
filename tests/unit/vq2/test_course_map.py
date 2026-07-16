"""Tests for the versioned VQ2 course-map contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_racing.deployments.vq2.course_map import (
    MapStatus,
    load_course_map,
)


def test_historical_map_loads_for_offline_inspection() -> None:
    course = load_course_map("vq2_historical_v0")
    assert course.status is MapStatus.HISTORICAL_UNVALIDATED
    assert course.gate(1).center_ned_m == pytest.approx((9.64, 6.52, -0.65))
    assert course.leg(0, 1).slant_range_m == (3.0, 16.0)


def test_historical_map_is_rejected_for_flight() -> None:
    with pytest.raises(ValueError, match="not flight-ready"):
        load_course_map("vq2_historical_v0", require_flight_ready=True)


def test_unknown_map_lists_known_artifact() -> None:
    with pytest.raises(FileNotFoundError, match="vq2_historical_v0"):
        load_course_map("does-not-exist")


def test_non_unit_normal_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
schema_version = 1
course_id = "bad"
status = "candidate"
frame = "gate0_local_ned"
units = "si"
drone_radius_m = 0.14
tracking_margin_m = 0.2
[[gates]]
id = 0
center_ned_m = [0, 0, 0]
position_std_m = [0, 0, 0]
inner_size_m = [1.5, 1.5]
normal_ned = [2, 0, 0]
[gates.evidence]
perception_revision = "structured-v1"
source_runs = []
independent_runs = 0
observation_count = 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unit length"):
        load_course_map(path)


def test_gate_ids_must_be_ordered_and_contiguous(tmp_path: Path) -> None:
    src = Path("deployments/vq2/course_maps/vq2_historical_v0.toml").read_text(encoding="utf-8")
    bad = tmp_path / "bad_ids.toml"
    bad.write_text(src.replace("id = 1", "id = 2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered and contiguous"):
        load_course_map(bad)
