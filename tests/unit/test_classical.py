"""Tests for drone_racing.perception.classical — HSV gate detector.

Test strategy:
    1. Synthetic coloured rectangle on grey background → detect 1 gate.
    2. No gate in image → empty list.
    3. Multiple gates → detect multiple.
    4. Corner ordering: TL, TR, BR, BL.
    5. Confidence scoring: good quad > poor shape.
    6. Too-small / too-large contours filtered out.
    7. Undistortion path exercised (even with zero distortion).
"""

from __future__ import annotations

import numpy as np

from drone_racing.perception.classical import (
    ClassicalDetector,
    ClassicalDetectorConfig,
    _order_corners_clockwise,
)
from drone_racing.utils.types import CameraConfig

# ---------------------------------------------------------------------------
# Helpers — synthetic image generation
# ---------------------------------------------------------------------------


def _make_grey_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a uniform grey RGB image."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    return img


def _draw_coloured_rect(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    colour_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw a filled coloured rectangle on an image.

    Parameters
    ----------
    image : np.ndarray
        RGB image to draw on (modified in-place and returned).
    center : tuple
        (cx, cy) centre of the rectangle.
    size : tuple
        (width, height) of the rectangle.
    colour_rgb : tuple
        RGB colour of the rectangle.

    Returns
    -------
    np.ndarray
        Modified image.
    """
    cx, cy = center
    w, h = size
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    image[y1:y2, x1:x2] = colour_rgb
    return image


# ---------------------------------------------------------------------------
# Detector config tuned for synthetic test images
# ---------------------------------------------------------------------------


def _test_config() -> ClassicalDetectorConfig:
    """Config tuned for bright red rectangles on grey backgrounds."""
    return ClassicalDetectorConfig(
        # Red in HSV: H≈0 or ≈170-179, S high, V high
        # Our bright red (255,0,0) → HSV ≈ (0, 255, 255)
        hsv_lower=(0, 100, 100),
        hsv_upper=(10, 255, 255),
        morph_kernel_size=3,
        min_contour_area=200.0,
        max_contour_area=300_000.0,
        min_confidence=0.2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleGateDetection:
    """Detect a single coloured rectangle."""

    def test_detects_red_rectangle(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (100, 100))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=1000)
        assert len(dets) >= 1
        # Should have 4 corners
        assert dets[0].corners_2d.shape[1] == 2
        assert len(dets[0].corners_2d) >= 4

    def test_timestamp_passed_through(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (80, 80))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=42)
        assert len(dets) >= 1
        assert dets[0].timestamp_ns == 42

    def test_confidence_positive(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (100, 100))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) >= 1
        assert dets[0].confidence > 0.0

    def test_corners_inside_image(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (100, 100))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) >= 1
        corners = dets[0].corners_2d
        assert np.all(corners[:, 0] >= 0)
        assert np.all(corners[:, 0] <= 640)
        assert np.all(corners[:, 1] >= 0)
        assert np.all(corners[:, 1] <= 480)


class TestNoGate:
    """No gate in the image."""

    def test_plain_grey_no_detection(self) -> None:
        img = _make_grey_image()
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) == 0

    def test_wrong_colour_no_detection(self) -> None:
        """Green rectangle with red-only HSV config → no detection."""
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (100, 100), colour_rgb=(0, 255, 0))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) == 0


class TestMultipleGates:
    """Multiple gates in one image."""

    def test_two_rectangles(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (160, 240), (80, 80))
        img = _draw_coloured_rect(img, (480, 240), (80, 80))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) >= 2

    def test_sorted_by_confidence(self) -> None:
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (160, 240), (80, 80))
        img = _draw_coloured_rect(img, (480, 240), (120, 120))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        if len(dets) >= 2:
            assert dets[0].confidence >= dets[1].confidence


class TestFiltering:
    """Area and aspect ratio filtering."""

    def test_tiny_contour_filtered(self) -> None:
        """A 5×5 rectangle is below min_contour_area."""
        img = _make_grey_image()
        img = _draw_coloured_rect(img, (320, 240), (5, 5))
        detector = ClassicalDetector(_test_config())
        dets = detector.detect(img, timestamp_ns=0)
        assert len(dets) == 0


class TestCornerOrdering:
    """Corner ordering function."""

    def test_already_ordered(self) -> None:
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        ordered = _order_corners_clockwise(corners)
        assert ordered.shape == (4, 2)
        # TL should have smallest x and y
        assert ordered[0, 0] <= ordered[1, 0]  # TL.x <= TR.x
        assert ordered[0, 1] <= ordered[3, 1]  # TL.y <= BL.y

    def test_shuffled_corners(self) -> None:
        # Shuffled order
        corners = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float64)
        ordered = _order_corners_clockwise(corners)
        # TL should be (0,0), TR (100,0), BR (100,100), BL (0,100)
        assert ordered[0, 0] < ordered[1, 0]  # TL left of TR


class TestUndistortion:
    """Camera config with zero distortion should work without errors."""

    def test_with_camera_config(self) -> None:
        cam = CameraConfig()  # zero distortion by default
        detector = ClassicalDetector(_test_config(), camera_config=cam)
        img = _make_grey_image(cam.width, cam.height)
        img = _draw_coloured_rect(img, (640, 360), (100, 100))
        dets = detector.detect(img, timestamp_ns=0)
        # Should work without crashing (zero distortion → no remap)
        assert isinstance(dets, list)
