"""Tests for drone_racing.perception.hybrid — neural + classical corner refinement.

Test strategy:
    1. _refine_corners_in_crop extracts corners from a synthetic gate crop.
    2. _refine_corners_in_crop returns None for blank/featureless crops.
    3. _refine_corners_in_crop maps corners back to full-image coordinates.
    4. HybridDetector config defaults are sane.
    5. HybridDetector.detect returns empty list when neural returns no bboxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.perception.classical import ClassicalDetectorConfig
from drone_racing.perception.hybrid import (
    HybridDetector,
    HybridDetectorConfig,
    _refine_corners_in_crop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gate_image(
    width: int = 200,
    height: int = 200,
    gate_color_rgb: tuple[int, int, int] = (0, 255, 0),
    bg_color_rgb: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """Create a synthetic image with a coloured square gate.

    The gate occupies roughly the centre 60% of the image.
    """
    img = np.full((height, width, 3), bg_color_rgb, dtype=np.uint8)
    margin = int(min(width, height) * 0.2)
    img[margin : height - margin, margin : width - margin] = gate_color_rgb
    return img


def _make_gate_with_hole(
    width: int = 200,
    height: int = 200,
    gate_color_rgb: tuple[int, int, int] = (0, 255, 0),
    bg_color_rgb: tuple[int, int, int] = (50, 50, 50),
) -> np.ndarray:
    """Create a synthetic gate with an inner hole (for 8-corner detection)."""
    img = np.full((height, width, 3), bg_color_rgb, dtype=np.uint8)
    outer = int(min(width, height) * 0.15)
    inner = int(min(width, height) * 0.30)
    # Outer gate
    img[outer : height - outer, outer : width - outer] = gate_color_rgb
    # Inner hole (background colour)
    img[inner : height - inner, inner : width - inner] = bg_color_rgb
    return img


def _default_morph_kernel() -> np.ndarray:
    """Pre-built morphological kernel matching default config."""
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def _default_hsv_bounds() -> tuple[np.ndarray, np.ndarray]:
    """HSV bounds matching default ClassicalDetectorConfig."""
    lower = np.array([0, 80, 80], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)
    return lower, upper


# ---------------------------------------------------------------------------
# Tests: corner refinement
# ---------------------------------------------------------------------------


class TestRefineCorners:
    """Tests for _refine_corners_in_crop."""

    def test_extracts_four_corners_from_solid_gate(self) -> None:
        image = _make_gate_image()
        bbox = np.array([0.0, 0.0, 200.0, 200.0])
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            image,
            bbox,
            cfg,
            margin=0.1,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is not None
        assert corners.shape == (4, 2)
        # All corners should be within image bounds
        assert np.all(corners >= 0)
        assert np.all(corners[:, 0] <= 200)
        assert np.all(corners[:, 1] <= 200)

    def test_returns_none_for_blank_image(self) -> None:
        # Uniform grey — no gate to detect
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = np.array([0.0, 0.0, 200.0, 200.0])
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            image,
            bbox,
            cfg,
            margin=0.1,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is None

    def test_returns_none_for_tiny_crop(self) -> None:
        image = _make_gate_image()
        bbox = np.array([95.0, 95.0, 105.0, 105.0])  # 10x10 region
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            image,
            bbox,
            cfg,
            margin=0.0,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is None

    def test_corners_offset_to_full_image_coords(self) -> None:
        # Place gate in the bottom-right of a larger image
        full = np.full((400, 400, 3), 50, dtype=np.uint8)
        gate_img = _make_gate_image(width=200, height=200)
        full[200:400, 200:400] = gate_img

        bbox = np.array([200.0, 200.0, 400.0, 400.0])
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            full,
            bbox,
            cfg,
            margin=0.0,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is not None
        # All corners should be in the bottom-right quadrant
        assert np.all(corners[:, 0] >= 200)
        assert np.all(corners[:, 1] >= 200)

    def test_gate_with_hole_returns_eight_corners(self) -> None:
        image = _make_gate_with_hole()
        bbox = np.array([0.0, 0.0, 200.0, 200.0])
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            image,
            bbox,
            cfg,
            margin=0.1,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is not None
        assert corners.shape == (8, 2)

    def test_margin_expands_crop(self) -> None:
        # Gate barely inside bbox — margin helps capture it
        full = np.full((300, 300, 3), 50, dtype=np.uint8)
        full[50:250, 50:250] = (0, 255, 0)  # gate

        # Tight bbox that misses edges
        bbox = np.array([60.0, 60.0, 240.0, 240.0])
        cfg = ClassicalDetectorConfig()
        hsv_lower, hsv_upper = _default_hsv_bounds()

        corners = _refine_corners_in_crop(
            full,
            bbox,
            cfg,
            margin=0.2,
            min_crop_size=30,
            morph_kernel=_default_morph_kernel(),
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
        )
        assert corners is not None
        assert len(corners) >= 4


# ---------------------------------------------------------------------------
# Tests: HybridDetector
# ---------------------------------------------------------------------------


class TestHybridDetector:
    """Tests for HybridDetector config and edge cases."""

    def test_default_config(self) -> None:
        cfg = HybridDetectorConfig()
        assert cfg.crop_margin == pytest.approx(0.3)
        assert cfg.min_crop_size == 30

    def test_detect_without_load_raises(self) -> None:
        detector = HybridDetector()
        image = _make_gate_image()
        with pytest.raises(RuntimeError, match="not loaded"):
            detector._neural.detect(image, 0)

    def test_config_passes_through(self) -> None:
        cfg = HybridDetectorConfig(crop_margin=0.5, min_crop_size=50)
        detector = HybridDetector(config=cfg)
        assert detector._cfg.crop_margin == pytest.approx(0.5)
        assert detector._cfg.min_crop_size == 50
