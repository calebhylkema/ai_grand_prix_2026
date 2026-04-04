"""Tests for drone_racing.utils.config — TOML loader."""

from __future__ import annotations

import pytest

from drone_racing.utils.config import load_config


class TestLoadConfig:
    """Config loading from default.toml."""

    def test_loads_default(self) -> None:
        cfg = load_config()
        assert cfg.estimation.imu_rate_hz == 200.0
        assert cfg.camera.width == 1280
        assert cfg.gate.width_m == 1.5

    def test_raw_dict_populated(self) -> None:
        cfg = load_config()
        assert "estimation" in cfg.raw
        assert "camera" in cfg.raw

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.toml")
