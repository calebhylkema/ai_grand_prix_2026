"""Tests for drone_racing.logging.frame_recorder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from drone_racing.logging.frame_recorder import FrameRecorder, FrameRecorderConfig
from drone_racing.utils.types import CameraConfig, GateDetection, State


def _fake_state(ts_ns: int = 1_000_000) -> State:
    return State.identity(timestamp_ns=ts_ns)


def _fake_frame(shape: tuple[int, int, int] = (120, 160, 3)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _fake_detection(ts_ns: int = 1_000_000) -> GateDetection:
    corners = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    return GateDetection(timestamp_ns=ts_ns, corners_2d=corners, confidence=0.9, gate_id=3)


class TestFrameRecorder:
    def test_writes_frame_and_metadata(self, tmp_path: Path) -> None:
        cam = CameraConfig(width=160, height=120)
        rec = FrameRecorder(tmp_path, camera_config=cam)
        ok = rec.record(_fake_frame(), 1_000_000, _fake_state(), [_fake_detection()])
        rec.close()

        assert ok is True
        assert (tmp_path / "frames" / "000000.jpg").exists()
        assert (tmp_path / "manifest.json").exists()

        lines = (tmp_path / "meta.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        meta = json.loads(lines[0])
        assert meta["idx"] == 0
        assert meta["timestamp_ns"] == 1_000_000
        assert len(meta["state"]["position"]) == 3
        assert meta["detections"][0]["gate_id"] == 3
        assert meta["detections"][0]["confidence"] == 0.9

    def test_stride_skips_frames(self, tmp_path: Path) -> None:
        cam = CameraConfig(width=160, height=120)
        rec = FrameRecorder(tmp_path, camera_config=cam, config=FrameRecorderConfig(stride=3))
        results = [rec.record(_fake_frame(), 1000 * i, _fake_state(1000 * i)) for i in range(7)]
        rec.close()

        # Expect records at seen-indices 0, 3, 6 → 3 frames written.
        assert results == [True, False, False, True, False, False, True]
        assert rec.frames_written == 3
        assert len(list((tmp_path / "frames").iterdir())) == 3

    def test_max_frames_caps_output(self, tmp_path: Path) -> None:
        cam = CameraConfig(width=160, height=120)
        rec = FrameRecorder(tmp_path, camera_config=cam, config=FrameRecorderConfig(max_frames=2))
        for i in range(5):
            rec.record(_fake_frame(), 1000 * i, _fake_state(1000 * i))
        rec.close()

        assert rec.frames_written == 2

    def test_manifest_records_intrinsics(self, tmp_path: Path) -> None:
        cam = CameraConfig(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0)
        rec = FrameRecorder(tmp_path, camera_config=cam)
        rec.record(_fake_frame((240, 320, 3)), 1000, _fake_state())
        rec.close()

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["camera"]["width"] == 320
        assert manifest["camera"]["fx"] == 300.0
        assert manifest["frames_written"] == 1

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        cam = CameraConfig(width=160, height=120)
        with FrameRecorder(tmp_path, camera_config=cam) as rec:
            rec.record(_fake_frame(), 1, _fake_state())
        # Manifest written on __exit__
        assert (tmp_path / "manifest.json").exists()
