"""Unit tests for the VQ2 controller's pure control math (fly.py).

Run: uv run pytest scripts/vq2/test_control.py -q

Locks in the 2026-07-02 control logic that got the drone airborne past the floor
sink: the camera-row vertical PD servo (``vertical_thrust``) and the tilt-
compensated thrust (``Flight.tilt_comp``). These are the pieces that must not
silently regress — a sign flip or a broken clamp here re-sinks the drone.
"""

from __future__ import annotations

import math
import threading

import control_check as cc
import fly
import numpy as np
from gates import GateTarget


def _gate(cy: float, area: float, bearing: float = 0.0) -> GateTarget:
    """A GateTarget with just the fields the vertical servo reads."""
    return GateTarget(cx=160.0, cy=cy, area=area, bearing=bearing, pose_cam=None)


# --- vertical_thrust: camera-row PD height servo -------------------------------


def test_vertical_thrust_no_gate_is_mild_climb() -> None:
    """No gate visible => a mild climb bias above hover (counters forward sink)."""
    thr = fly.vertical_thrust(None)
    expected = min(fly.THR_MAX, fly.HOVER_THRUST + fly.NO_GATE_CLIMB)
    assert thr == expected
    assert thr > fly.HOVER_THRUST


def test_vertical_thrust_high_gate_climbs() -> None:
    """Gate above centre (cy < CY_TARGET) => climb (thrust above hover)."""
    thr = fly.vertical_thrust(_gate(cy=30.0, area=1000.0))
    assert thr > fly.HOVER_THRUST


def test_vertical_thrust_low_far_gate_holds() -> None:
    """Gate below centre but small/far => hold hover, do NOT dive at the floor."""
    thr = fly.vertical_thrust(_gate(cy=250.0, area=fly.DESCEND_AREA - 1.0))
    assert thr == fly.HOVER_THRUST


def test_vertical_thrust_low_close_gate_descends() -> None:
    """Gate below centre AND close (big area) => descend (thrust below hover)."""
    thr = fly.vertical_thrust(_gate(cy=250.0, area=fly.DESCEND_AREA + 2000.0))
    assert thr < fly.HOVER_THRUST


def test_vertical_thrust_cy_rate_damps_climb() -> None:
    """Rising cy (climbing, gate moving down the frame) bleeds thrust vs. no rate."""
    gate = _gate(cy=30.0, area=1000.0)
    thr_static = fly.vertical_thrust(gate, cy_rate=0.0)
    thr_damped = fly.vertical_thrust(gate, cy_rate=200.0)  # climbing fast
    assert thr_damped < thr_static


def test_vertical_thrust_clamped_to_bounds() -> None:
    """An extreme error saturates at THR_MAX, never beyond."""
    thr = fly.vertical_thrust(_gate(cy=-5000.0, area=1000.0))
    assert thr == fly.THR_MAX


# --- Flight.tilt_comp: hold altitude through banked/pitched turns --------------


def _flight() -> fly.Flight:
    sh = cc.Shared()
    return fly.Flight(sh, conn=None, tgt_sys=1, tgt_comp=1, lock=threading.Lock())


def _set_pitch(fl: fly.Flight, pitch_rad: float) -> None:
    """Force the attitude filter to a pure pitch (rotation about body y)."""
    fl.att.q = np.array([math.cos(pitch_rad / 2), 0.0, math.sin(pitch_rad / 2), 0.0])


def test_tilt_comp_level_is_identity() -> None:
    """Level (cos=1) leaves an in-range thrust unchanged."""
    fl = _flight()
    _set_pitch(fl, 0.0)
    assert fl.tilt_comp(0.30) == 0.30


def test_tilt_comp_boosts_when_pitched() -> None:
    """A 30° pitch scales thrust up by ~1/cos(30°) to hold vertical force."""
    fl = _flight()
    _set_pitch(fl, math.radians(30.0))
    assert fl.tilt_comp(0.30) == float(
        np.clip(0.30 / math.cos(math.radians(30.0)), fly.THR_MIN, fly.THR_MAX)
    )
    assert fl.tilt_comp(0.30) > 0.30


def test_tilt_comp_bounded_at_extreme_tilt() -> None:
    """Near-90° tilt cannot command runaway thrust: floored cos + THR_MAX clamp."""
    fl = _flight()
    _set_pitch(fl, math.radians(85.0))
    assert fl.tilt_comp(0.40) == fly.THR_MAX
