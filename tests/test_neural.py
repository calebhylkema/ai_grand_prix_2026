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
    _clockwise_from_topleft,
    _letterbox,
    _nms,
    order_corners,
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


class TestClockwiseFromTopleft:
    """Tests for clockwise corner sorting."""

    def test_already_ordered(self) -> None:
        corners = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float64)
        result = _clockwise_from_topleft(corners)
        np.testing.assert_array_equal(result, corners)

    def test_shuffled_square(self) -> None:
        expected = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float64)
        shuffled = expected[[2, 0, 3, 1]]
        result = _clockwise_from_topleft(shuffled)
        np.testing.assert_array_equal(result, expected)

    def test_reversed_order(self) -> None:
        expected = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float64)
        result = _clockwise_from_topleft(expected[::-1])
        np.testing.assert_array_equal(result, expected)

    def test_non_square_quad(self) -> None:
        corners = np.array([[80, 60], [320, 80], [300, 280], [100, 260]], dtype=np.float64)
        result = _clockwise_from_topleft(corners)
        # TL should have smallest x+y sum, BR should have largest
        assert result[0, 0] < result[1, 0]  # TL.x < TR.x
        assert result[0, 1] < result[3, 1]  # TL.y < BL.y
        assert result[2, 0] > result[3, 0]  # BR.x > BL.x

    def test_wrong_count_returns_as_is(self) -> None:
        corners = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        result = _clockwise_from_topleft(corners)
        np.testing.assert_array_equal(result, corners)


class TestOrderCorners:
    """Tests for the full corner ordering function."""

    def test_four_corners(self) -> None:
        expected = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float64)
        shuffled = expected[[3, 1, 0, 2]]
        result = order_corners(shuffled, 4)
        np.testing.assert_array_equal(result, expected)

    def test_eight_corners_outer_inner_separation(self) -> None:
        outer = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float64)
        inner = np.array([[150, 150], [250, 150], [250, 250], [150, 250]], dtype=np.float64)
        combined = np.vstack([outer, inner])
        shuffled = combined[[5, 2, 7, 0, 4, 1, 6, 3]]
        result = order_corners(shuffled, 8)
        np.testing.assert_array_equal(result[:4], outer)
        np.testing.assert_array_equal(result[4:], inner)

    def test_eight_corners_outer_farther_than_inner(self) -> None:
        outer = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float64)
        inner = np.array([[130, 130], [270, 130], [270, 270], [130, 270]], dtype=np.float64)
        combined = np.vstack([outer, inner])
        result = order_corners(combined[np.random.permutation(8)], 8)
        centroid = result.mean(axis=0)
        outer_d = np.linalg.norm(result[:4] - centroid, axis=1).mean()
        inner_d = np.linalg.norm(result[4:] - centroid, axis=1).mean()
        assert outer_d > inner_d

    def test_other_count_returned_as_is(self) -> None:
        corners = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float64)
        result = order_corners(corners, 8)
        np.testing.assert_array_equal(result, corners)
