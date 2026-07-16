"""Versioned, validated course-map contract for the VQ2 deployment.

Course maps are reviewed deployment data, not controller tuning constants.  This
module intentionally separates *structural validity* (the file is well formed) from
*flight readiness* (the evidence is strong enough to authorize map-following flight).
An experimental map can therefore be inspected and used by offline tools without
silently becoming part of the command path.

The only supported map frame is ``gate0_local_ned``: origin at gate 0, X along the
verified gate-0 exit heading, Y right, and Z down.  It is a yaw-anchored local NED
frame, not the simulator's unavailable absolute world position.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from drone_racing.utils.workspace import Workspace

SCHEMA_VERSION = 1
COURSE_MAP_DIR = Workspace.locate().root / "deployments" / "vq2" / "course_maps"


class MapFrame(StrEnum):
    """Coordinate frames accepted by the course-map loader."""

    GATE0_LOCAL_NED = "gate0_local_ned"


class MapStatus(StrEnum):
    """Review status of a course-map artifact."""

    HISTORICAL_UNVALIDATED = "historical_unvalidated"
    CANDIDATE = "candidate"
    VALIDATED = "validated"


@dataclass(frozen=True, slots=True)
class MapEvidence:
    """Provenance and independence information for one gate estimate."""

    perception_revision: str
    source_runs: tuple[str, ...]
    independent_runs: int
    observation_count: int


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One ordered racing gate expressed in the map frame."""

    gate_id: int
    center_ned_m: tuple[float, float, float]
    position_std_m: tuple[float, float, float]
    inner_size_m: tuple[float, float]
    normal_ned: tuple[float, float, float] | None
    evidence: MapEvidence

    @property
    def center(self) -> np.ndarray:
        """Gate center as a fresh ``float64`` vector."""
        return np.asarray(self.center_ned_m, dtype=np.float64)

    @property
    def normal(self) -> np.ndarray | None:
        """Unit traversal normal as a fresh vector, or ``None`` if unresolved."""
        if self.normal_ned is None:
            return None
        return np.asarray(self.normal_ned, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GateLeg:
    """Association constraints for a transition between consecutive gates."""

    from_gate: int
    to_gate: int
    expected_turn_sign: int
    slant_range_m: tuple[float, float]
    turn_window_rad: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CourseMap:
    """Complete typed course-map artifact."""

    schema_version: int
    course_id: str
    status: MapStatus
    frame: MapFrame
    units: str
    drone_radius_m: float
    tracking_margin_m: float
    gates: tuple[GateSpec, ...]
    legs: tuple[GateLeg, ...]

    def __post_init__(self) -> None:
        _validate_structure(self)

    def gate(self, gate_id: int) -> GateSpec:
        """Return a gate by race index.

        Raises
        ------
        KeyError
            If the map does not contain ``gate_id``.
        """
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        raise KeyError(f"course {self.course_id!r} has no gate {gate_id}")

    def leg(self, from_gate: int, to_gate: int) -> GateLeg:
        """Return the association contract for one gate transition."""
        for leg in self.legs:
            if leg.from_gate == from_gate and leg.to_gate == to_gate:
                return leg
        raise KeyError(f"course {self.course_id!r} has no leg {from_gate}->{to_gate}")

    def flight_readiness_issues(self) -> tuple[str, ...]:
        """Return every reason this map must not command a flight.

        Structural errors are rejected while loading.  These readiness checks cover
        review/evidence requirements that are intentionally allowed in offline maps.
        """
        issues: list[str] = []
        if self.status is not MapStatus.VALIDATED:
            issues.append(f"map status is {self.status.value}, not validated")
        if len(self.gates) < 2:
            issues.append("fewer than two gates are mapped")
        for gate in self.gates:
            if gate.normal_ned is None:
                issues.append(f"gate {gate.gate_id} traversal normal is unresolved")
            if gate.gate_id > 0 and gate.evidence.independent_runs < 3:
                issues.append(
                    f"gate {gate.gate_id} has only {gate.evidence.independent_runs} independent runs"
                )
            revision = gate.evidence.perception_revision.lower()
            if "legacy" in revision or "inverted" in revision or "unknown" in revision:
                issues.append(
                    f"gate {gate.gate_id} uses obsolete/unknown perception revision "
                    f"{gate.evidence.perception_revision!r}"
                )
        expected_legs = {(i, i + 1) for i in range(len(self.gates) - 1)}
        actual_legs = {(leg.from_gate, leg.to_gate) for leg in self.legs}
        for missing in sorted(expected_legs - actual_legs):
            issues.append(f"missing consecutive leg {missing[0]}->{missing[1]}")
        return tuple(issues)

    def require_flight_ready(self) -> None:
        """Raise if this map has not met the flight-evidence gate."""
        issues = self.flight_readiness_issues()
        if issues:
            raise ValueError("course map is not flight-ready: " + "; ".join(issues))


def _tuple_floats(value: object, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be an array of {length} numbers")
    out = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{field_name} contains a non-finite value")
    return out


def _parse_evidence(raw: dict[str, Any], prefix: str) -> MapEvidence:
    runs_raw = raw.get("source_runs", [])
    if not isinstance(runs_raw, list) or not all(isinstance(run, str) for run in runs_raw):
        raise ValueError(f"{prefix}.source_runs must be an array of strings")
    runs = tuple(dict.fromkeys(runs_raw))
    independent = int(raw.get("independent_runs", len(runs)))
    observations = int(raw.get("observation_count", 0))
    if independent < 0 or observations < 0 or independent > len(runs):
        raise ValueError(f"{prefix} evidence counts are inconsistent")
    revision = str(raw.get("perception_revision", "unknown")).strip()
    if not revision:
        raise ValueError(f"{prefix}.perception_revision must not be empty")
    return MapEvidence(revision, runs, independent, observations)


def _parse_gate(raw: dict[str, Any]) -> GateSpec:
    gate_id = int(raw["id"])
    center = _tuple_floats(raw["center_ned_m"], 3, f"gate {gate_id}.center_ned_m")
    std = _tuple_floats(raw["position_std_m"], 3, f"gate {gate_id}.position_std_m")
    size = _tuple_floats(raw["inner_size_m"], 2, f"gate {gate_id}.inner_size_m")
    if any(x < 0.0 for x in std):
        raise ValueError(f"gate {gate_id}.position_std_m must be non-negative")
    if any(x <= 0.0 for x in size):
        raise ValueError(f"gate {gate_id}.inner_size_m must be positive")
    normal_raw = raw.get("normal_ned")
    normal: tuple[float, float, float] | None = None
    if normal_raw is not None:
        parsed = _tuple_floats(normal_raw, 3, f"gate {gate_id}.normal_ned")
        norm = float(np.linalg.norm(parsed))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"gate {gate_id}.normal_ned must be unit length (got {norm:.6f})")
        normal = (parsed[0], parsed[1], parsed[2])
    evidence_raw = raw.get("evidence", {})
    if not isinstance(evidence_raw, dict):
        raise ValueError(f"gate {gate_id}.evidence must be a table")
    return GateSpec(
        gate_id=gate_id,
        center_ned_m=(center[0], center[1], center[2]),
        position_std_m=(std[0], std[1], std[2]),
        inner_size_m=(size[0], size[1]),
        normal_ned=normal,
        evidence=_parse_evidence(evidence_raw, f"gate {gate_id}.evidence"),
    )


def _parse_leg(raw: dict[str, Any]) -> GateLeg:
    source, target = int(raw["from_gate"]), int(raw["to_gate"])
    sign = int(raw["expected_turn_sign"])
    if sign not in {-1, 0, 1}:
        raise ValueError(f"leg {source}->{target}.expected_turn_sign must be -1, 0, or 1")
    slant = _tuple_floats(raw["slant_range_m"], 2, f"leg {source}->{target}.slant_range_m")
    turn_deg = _tuple_floats(raw["turn_window_deg"], 2, f"leg {source}->{target}.turn_window_deg")
    if not 0.0 <= slant[0] < slant[1]:
        raise ValueError(f"leg {source}->{target}.slant_range_m must be increasing")
    if not 0.0 <= turn_deg[0] < turn_deg[1] <= 180.0:
        raise ValueError(f"leg {source}->{target}.turn_window_deg must lie in [0, 180]")
    return GateLeg(
        from_gate=source,
        to_gate=target,
        expected_turn_sign=sign,
        slant_range_m=(slant[0], slant[1]),
        turn_window_rad=(math.radians(turn_deg[0]), math.radians(turn_deg[1])),
    )


def _validate_structure(course: CourseMap) -> None:
    if course.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"course-map schema v{course.schema_version} != supported v{SCHEMA_VERSION}"
        )
    if not course.course_id.strip():
        raise ValueError("course_id must not be empty")
    if course.units != "si":
        raise ValueError("course-map units must be 'si'")
    if course.drone_radius_m <= 0.0 or course.tracking_margin_m < 0.0:
        raise ValueError("drone radius must be positive and tracking margin non-negative")
    ids = [gate.gate_id for gate in course.gates]
    if ids != list(range(len(ids))):
        raise ValueError(f"gate ids must be ordered and contiguous from 0 (got {ids})")
    if course.gates and not np.allclose(course.gates[0].center, 0.0, atol=1e-9):
        raise ValueError("gate 0 must be the origin of gate0_local_ned")
    min_opening = 2.0 * (course.drone_radius_m + course.tracking_margin_m)
    for gate in course.gates:
        if not np.all(np.isfinite(gate.center)):
            raise ValueError(f"gate {gate.gate_id} center contains a non-finite value")
        if any(value < 0.0 or not math.isfinite(value) for value in gate.position_std_m):
            raise ValueError(f"gate {gate.gate_id} position uncertainty is invalid")
        if any(value <= 0.0 or not math.isfinite(value) for value in gate.inner_size_m):
            raise ValueError(f"gate {gate.gate_id} inner size is invalid")
        if gate.normal is not None and not math.isclose(
            float(np.linalg.norm(gate.normal)), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"gate {gate.gate_id} traversal normal must be unit length")
        evidence = gate.evidence
        unique_runs = len(set(evidence.source_runs))
        if (
            not evidence.perception_revision.strip()
            or evidence.independent_runs < 0
            or evidence.independent_runs > unique_runs
            or evidence.observation_count < evidence.independent_runs
        ):
            raise ValueError(f"gate {gate.gate_id} evidence is inconsistent")
        if min(gate.inner_size_m) <= min_opening:
            raise ValueError(
                f"gate {gate.gate_id} opening leaves no positive clearance after safety margin"
            )
    gate_ids = set(ids)
    seen_legs: set[tuple[int, int]] = set()
    for leg in course.legs:
        key = (leg.from_gate, leg.to_gate)
        if key in seen_legs:
            raise ValueError(f"duplicate leg {key[0]}->{key[1]}")
        seen_legs.add(key)
        if leg.from_gate not in gate_ids or leg.to_gate not in gate_ids:
            raise ValueError(f"leg {key[0]}->{key[1]} references an unknown gate")
        if leg.to_gate != leg.from_gate + 1:
            raise ValueError(f"leg {key[0]}->{key[1]} is not a consecutive race transition")


def load_course_map(name_or_path: str | Path, *, require_flight_ready: bool = False) -> CourseMap:
    """Load and validate a version-controlled VQ2 course map.

    Parameters
    ----------
    name_or_path : str | Path
        Map stem in ``deployments/vq2/course_maps`` or an explicit TOML path.
    require_flight_ready : bool
        If true, reject candidate/historical maps and unresolved evidence.
    """
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = COURSE_MAP_DIR / f"{path}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in COURSE_MAP_DIR.glob("*.toml"))
        raise FileNotFoundError(f"No VQ2 course map at {path} (known: {', '.join(known)})")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    gates_raw = raw.get("gates", [])
    legs_raw = raw.get("legs", [])
    if not isinstance(gates_raw, list) or not isinstance(legs_raw, list):
        raise ValueError("gates and legs must be TOML table arrays")
    course = CourseMap(
        schema_version=int(raw.get("schema_version", -1)),
        course_id=str(raw.get("course_id", "")),
        status=MapStatus(str(raw.get("status", ""))),
        frame=MapFrame(str(raw.get("frame", ""))),
        units=str(raw.get("units", "")),
        drone_radius_m=float(raw["drone_radius_m"]),
        tracking_margin_m=float(raw["tracking_margin_m"]),
        gates=tuple(_parse_gate(gate) for gate in gates_raw),
        legs=tuple(_parse_leg(leg) for leg in legs_raw),
    )
    if require_flight_ready:
        course.require_flight_ready()
    return course
