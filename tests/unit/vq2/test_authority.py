"""Safety-boundary tests for Training-only transition authority."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_racing.deployments.vq2.authority import TransitionAuthorityMux
from drone_racing.deployments.vq2.config import load
from drone_racing.deployments.vq2.follower import RateThrustSetpoint
from drone_racing.deployments.vq2.gate_relative import (
    GateApproachResult,
    GateApproachStatus,
    GuidanceDecision,
    StructuredTarget,
)

NOW = 5_000_000_000
LEGACY = (0.30, 0.1, -0.2, 0.3)
TRANSITION = (0.34, 0.8, -0.5, 0.6)


def target(**over: object) -> StructuredTarget:
    base: dict[str, object] = {
        "valid": True,
        "frame_id": 10,
        "frame_sensor_ns": 1,
        "arrival_ns": NOW - 5_000_000,
        "active_gate_at_receipt": 1,
        "track_id": 7,
        "race_gate_id": 1,
        "track_hits": 5,
        "output_mode": "position_only",
        "bearing_rad": math.radians(30.0),
        "los_body_unit": np.array([0.86, 0.5, 0.0]),
        "slant_range_m": 25.0,
        "has_metric_pose": True,
        "confidence": 0.9,
        "cov_trace": 0.3,
        "expected_turn_sign": 1.0,
        "disposition": "assoc_refused",
        "out_of_band": True,
        "sequence": 4,
    }
    base.update(over)
    return StructuredTarget(**base)  # type: ignore[arg-type]


def decision(**over: object) -> GuidanceDecision:
    result = GateApproachResult(
        status=GateApproachStatus.TRANSITION_ACQUIRE,
        setpoint=RateThrustSetpoint(
            collective_thrust=TRANSITION[0],
            body_rates_rps=np.asarray(TRANSITION[1:]),
            requested_acceleration_ned_mps2=np.zeros(3),
            feasible_acceleration_ned_mps2=np.zeros(3),
            desired_rotation_body_to_ned=np.eye(3),
            saturation_flags=(),
        ),
    )
    base: dict[str, object] = {
        "decision_mono_ns": NOW - 1_000_000,
        "now_ns": NOW - 1_000_000,
        "active_gate": 1,
        "anchor_gate": 0,
        "track_id": 7,
        "target_arrival_ns": NOW - 5_000_000,
        "mode": "transition",
        "result": result,
        "decision_sequence": 9,
        "state_sequence": 3,
        "state_mono_ns": NOW - 2_000_000,
        "target_sequence": 4,
        "input_mono_ns": NOW - 2_000_000,
        "initialised": True,
    }
    base.update(over)
    return GuidanceDecision(**base)  # type: ignore[arg-type]


def select(mux: TransitionAuthorityMux, **over: object):  # noqa: ANN201
    values: dict[str, object] = {
        "now_ns": NOW,
        "transition_start_ns": NOW - 200_000_000,
        "active_gate": 1,
        "anchor_gate": 0,
        "state_sequence": 3,
        "target": target(),
        "decision": decision(),
        "legacy": LEGACY,
    }
    values.update(over)
    return mux.select(**values)  # type: ignore[arg-type]


def test_default_off_is_byte_identical_legacy() -> None:
    result = select(TransitionAuthorityMux())
    assert result.source == "legacy" and result.reason == "disabled"
    assert result.selected == LEGACY


def test_gate0_is_always_legacy_even_when_enabled() -> None:
    result = select(TransitionAuthorityMux(True), active_gate=0)
    assert result.source == "legacy" and result.selected == LEGACY


def test_only_fresh_transition_decision_enters_authority() -> None:
    mux = TransitionAuthorityMux(True)
    result = select(mux)
    assert result.source == "transition" and result.entered
    assert result.selected == TRANSITION


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"anchor_gate": 1}, "reanchored"),
        ({"anchor_gate": -1}, "anchor_mismatch"),
        ({"transition_start_ns": NOW - 3_000_000_000}, "window_expired"),
        ({"decision": decision(decision_mono_ns=NOW - 100_000_000)}, "stale_decision"),
        ({"target": target(wrong_turn=True)}, "wrong_turn"),
        ({"target": target(association_refused=True)}, "association_refused"),
        ({"target": target(track_id=8)}, "track_mismatch"),
        ({"target": target(track_hits=1)}, "unconfirmed_track"),
    ],
)
def test_fail_closed_authority_paths(over: dict[str, object], reason: str) -> None:
    result = select(TransitionAuthorityMux(True), **over)
    assert result.source == "legacy" and result.reason == reason
    assert result.selected == LEGACY


def test_abort_bypasses_slew_and_restores_legacy() -> None:
    mux = TransitionAuthorityMux(True)
    select(mux)
    result = select(mux, now_ns=NOW + 10_000_000, abort_reason="collision")
    assert result.source == "legacy" and result.exited
    assert result.selected == LEGACY and not result.slew_limited


def test_entry_and_exit_are_slew_limited_after_legacy_history() -> None:
    mux = TransitionAuthorityMux(True)
    select(mux, now_ns=NOW - 20_000_000, active_gate=0)
    entered = select(mux)
    assert entered.source == "transition" and entered.slew_limited
    assert entered.selected != TRANSITION
    exited = select(mux, now_ns=NOW + 5_000_000, anchor_gate=1)
    assert exited.source == "legacy" and exited.exited and exited.slew_limited


def test_reset_clears_attempt_authority_state() -> None:
    mux = TransitionAuthorityMux(True)
    select(mux)
    mux.reset()
    result = select(mux, active_gate=0)
    assert result.source == "legacy" and not result.exited


def test_qualification_cannot_permit_experiment() -> None:
    from drone_racing.deployments.vq2 import mission

    assert load("training").run.permit_transition_authority
    assert not load("qualification").run.permit_transition_authority
    assert mission._validate_transition_authority(load("training"), True)
    assert not mission._validate_transition_authority(load("training"), False)
    with pytest.raises(ValueError, match="only in Training"):
        mission._validate_transition_authority(load("qualification"), True)


def test_visual_handoff_validation_gate() -> None:
    from drone_racing.deployments.vq2 import mission

    assert load("training").run.permit_visual_handoff
    assert not load("qualification").run.permit_visual_handoff
    # Training + flag → enabled; flag off → disabled.
    assert mission._validate_visual_handoff(load("training"), True, False)
    assert not mission._validate_visual_handoff(load("training"), False, False)
    # Qualification rejects the flag.
    with pytest.raises(ValueError, match="only in Training"):
        mission._validate_visual_handoff(load("qualification"), True, False)
    # Mutually exclusive with the older ESKF transition authority.
    with pytest.raises(ValueError, match="mutually exclusive"):
        mission._validate_visual_handoff(load("training"), True, True)


def test_mux_never_imports_calibration_or_sends_commands() -> None:
    from pathlib import Path

    source = Path("src/drone_racing/deployments/vq2/authority.py").read_text()
    assert "calibration" not in source
    assert "Flight.command" not in source and "send_att" not in source


def test_visual_handoff_never_touches_calibrated_boundary() -> None:
    from pathlib import Path

    source = Path("src/drone_racing/deployments/vq2/visual_handoff.py").read_text()
    # It emits desired ACTUAL rates but never IMPORTS calibration/runtime nor CALLS the
    # sender/calibration (docstring prose may name them; usage patterns must be absent).
    import_lines = [
        ln for ln in source.splitlines() if ln.lstrip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    assert "calibration" not in joined
    assert "runtime" not in joined
    assert "send_att(" not in source
    assert ".RATE_GAIN" not in source
    assert "desired_to_cmd(" not in source
