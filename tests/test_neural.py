"""Tests for the neural gate detector."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drone_racing.perception.neural import (
    NeuralDetector,
    NeuralDetectorConfig,
    _letterbox,
    _nms,
)


def _mock_ort_module(mock_session: MagicMock) -> types.ModuleType:
    """Create a fake onnxruntime module that returns mock_session."""
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = MagicMock(return_value=mock_session)  # type: ignore[attr-defined]
    return ort


class TestLetterbox:
    """Tests for letterbox preprocessing."""

    def test_square_image_to_square_target(self) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result, scale, (pad_x, pad_y) = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert pad_x == pytest.approx(0.0, abs=1.0)
        assert pad_y == pytest.approx(0.0, abs=1.0)

    def test_wide_image_gets_vertical_padding(self) -> None:
        img = np.zeros((360, 1280, 3), dtype=np.uint8)
        result, scale, (pad_x, pad_y) = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        # Width fills 640, so scale = 640/1280 = 0.5
        assert scale == pytest.approx(0.5, abs=0.01)
        # Height becomes 180, vertical padding = (640-180)/2 = 230
        assert pad_y > 100

    def test_tall_image_gets_horizontal_padding(self) -> None:
        img = np.zeros((1280, 360, 3), dtype=np.uint8)
        result, scale, (pad_x, pad_y) = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert pad_x > 100


class TestNMS:
    """Tests for non-maximum suppression."""

    def test_no_boxes(self) -> None:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        assert _nms(boxes, scores, 0.5) == []

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        assert _nms(boxes, scores, 0.5) == [0]

    def test_non_overlapping_boxes_kept(self) -> None:
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [100, 100, 150, 150],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, 0.5)
        assert len(keep) == 2

    def test_overlapping_boxes_suppressed(self) -> None:
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [12, 12, 52, 52],  # high overlap
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, 0.3)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept


class TestNeuralDetector:
    """Tests for the NeuralDetector class."""

    def test_detect_without_load_raises(self) -> None:
        detector = NeuralDetector()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(image, timestamp_ns=0)

    def test_load_missing_model_raises(self) -> None:
        config = NeuralDetectorConfig(model_path="nonexistent.onnx")
        detector = NeuralDetector(config)
        mock_session = MagicMock()
        ort_mod = _mock_ort_module(mock_session)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod}), pytest.raises(FileNotFoundError):
            detector.load()

    def test_detect_with_mock_onnx(self) -> None:
        """Test full detection pipeline with a mocked ONNX session."""
        config = NeuralDetectorConfig(
            conf_threshold=0.3,
            num_keypoints=8,
        )
        detector = NeuralDetector(config)

        # Create mock ONNX output: 1 detection with 8 keypoints
        n_kp = 8
        n_cols = 5 + n_kp * 3  # 29
        n_preds = 1

        # Single detection centred at (320, 320) with size 100x100
        pred = np.zeros((1, n_cols, n_preds), dtype=np.float32)
        pred[0, 0, 0] = 320.0  # cx
        pred[0, 1, 0] = 320.0  # cy
        pred[0, 2, 0] = 100.0  # w
        pred[0, 3, 0] = 100.0  # h
        pred[0, 4, 0] = 0.9  # confidence

        # Set 8 keypoints around the centre with high visibility
        kp_positions = [
            (280, 280),
            (360, 280),
            (360, 360),
            (280, 360),  # outer
            (300, 300),
            (340, 300),
            (340, 340),
            (300, 340),  # inner
        ]
        for i, (kx, ky) in enumerate(kp_positions):
            base = 5 + i * 3
            pred[0, base, 0] = kx
            pred[0, base + 1, 0] = ky
            pred[0, base + 2, 0] = 0.9  # keypoint confidence

        mock_session = MagicMock()
        mock_session.run.return_value = [pred]
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.get_outputs.return_value = [MagicMock(name="output0")]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        ort_mod = _mock_ort_module(mock_session)
        with (
            patch.dict(sys.modules, {"onnxruntime": ort_mod}),
            patch("drone_racing.perception.neural.Path.exists", return_value=True),
        ):
            detector.load()

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(image, timestamp_ns=1000)

        assert len(detections) == 1
        det = detections[0]
        assert det.timestamp_ns == 1000
        assert det.confidence >= 0.9
        assert det.corners_2d.shape[0] == 8  # all 8 keypoints visible
        assert det.corners_2d.shape[1] == 2

    def test_low_confidence_filtered(self) -> None:
        """Test that low-confidence detections are filtered out."""
        config = NeuralDetectorConfig(conf_threshold=0.5)
        detector = NeuralDetector(config)

        pred = np.zeros((1, 29, 1), dtype=np.float32)
        pred[0, 4, 0] = 0.3  # below threshold

        mock_session = MagicMock()
        mock_session.run.return_value = [pred]
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.get_outputs.return_value = [MagicMock(name="output0")]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        ort_mod = _mock_ort_module(mock_session)
        with (
            patch.dict(sys.modules, {"onnxruntime": ort_mod}),
            patch("drone_racing.perception.neural.Path.exists", return_value=True),
        ):
            detector.load()

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(image, timestamp_ns=0)
        assert len(detections) == 0
