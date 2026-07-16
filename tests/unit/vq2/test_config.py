"""Unit tests for the VQ2 typed config loader.

These guard the reorg's config extraction: the deployment TOMLs must load into
fully-typed params, angles must convert to radians exactly once (at the load
boundary), and CLI overrides must not mutate mission parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from drone_racing.deployments.vq2 import config

if TYPE_CHECKING:
    from pathlib import Path


def test_training_config_loads() -> None:
    cfg = config.load("training")
    assert cfg.name == "training"
    assert cfg.run.trace is True
    assert cfg.run.attempts >= 1
    assert cfg.run.mavlink_endpoint.startswith("udpin:")


def test_qualification_config_loads() -> None:
    cfg = config.load("qualification")
    assert cfg.name == "qualification"
    assert cfg.run.trace is False  # no trace I/O overhead on runs that count


def test_flight_values_carried_over_verbatim() -> None:
    """The fly2.py tuning values must survive the TOML round-trip exactly."""
    m = config.load("training").mission
    assert m.cmd_hz == 60.0
    assert m.pitch_fwd == -0.05
    assert m.kp_yaw == 1.5
    assert m.brake_secs == 1.6
    assert m.brake_arrest_s == 1.0
    assert m.accept_depth_m == (3.0, 13.0)
    assert m.pass_area == 15000.0
    assert m.pass_range_m == 3.5


def test_accept_turn_converted_to_radians() -> None:
    """TOML stores accept_turn_deg in degrees; the loaded value must be radians."""
    m = config.load("training").mission
    assert m.accept_turn == pytest.approx((math.radians(25.0), math.radians(110.0)))
    # sanity: a degrees value sneaking through would be > 2*pi
    assert 0 < m.accept_turn[0] < m.accept_turn[1] < math.tau


def test_load_by_explicit_path(tmp_path: Path) -> None:
    src = config.CONFIG_DIR / "training.toml"
    custom = tmp_path / "custom.toml"
    custom.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = config.load(custom)
    assert cfg.name == "training"  # name key in the file wins over the filename


def test_unknown_name_lists_known_configs() -> None:
    with pytest.raises(FileNotFoundError, match="training"):
        config.load("no_such_config")


def test_replace_run_overrides_only_run_params() -> None:
    cfg = config.load("training")
    out = config.replace_run(cfg, secs=99.0, attempts=7)
    assert out.run.secs == 99.0
    assert out.run.attempts == 7
    assert out.run.trace == cfg.run.trace  # untouched fields survive
    assert out.mission == cfg.mission  # mission params never change
    assert cfg.run.secs != 99.0  # original is not mutated (frozen)


def test_mission_params_frozen() -> None:
    m = config.load("training").mission
    with pytest.raises(AttributeError):
        m.cmd_hz = 120.0  # type: ignore[misc]
