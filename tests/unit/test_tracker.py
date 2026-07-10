"""Tests for drone_racing.perception.tracker — gate tracking across frames.

Test strategy:
    1. Single detection across multiple frames → stable ID assigned.
    2. Two detections → two distinct IDs.
    3. Detection disappears → track ages out and is pruned.
    4. IoU matching: overlapping bboxes match, non-overlapping don't.
    5. min_hits threshold: new tracks don't get IDs until confirmed.
    6. Gate map ID assignment: tracks associate with closest projected gate.
    7. Reset clears all state.
    8. Empty frames handled gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_racing.perception.tracker import (
    GateTracker,
    GateTrackerConfig,
    _corners_to_bbox,
    _iou,
    _iou_matrix,
)
from drone_racing.utils.types import GateDetection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    cx: float,
    cy: float,
    size: float = 50.0,
    confidence: float = 0.8,
    timestamp_ns: int = 0,
) -> GateDetection:
    """Create a detection with a square bounding box centred at (cx, cy)."""
    half = size / 2.0
    corners = np.array(
        [
            [cx - half, cy - half],  # TL
            [cx + half, cy - half],  # TR
            [cx + half, cy + half],  # BR
            [cx - half, cy + half],  # BL
        ],
        dtype=np.float64,
    )
    return GateDetection(
        timestamp_ns=timestamp_ns,
        corners_2d=corners,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestIoU:
    """IoU computation correctness."""

    def test_identical_boxes(self) -> None:
        bbox = np.array([10, 20, 110, 120], dtype=np.float64)
        assert _iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = np.array([0, 0, 10, 10], dtype=np.float64)
        b = np.array([20, 20, 30, 30], dtype=np.float64)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = np.array([0, 0, 10, 10], dtype=np.float64)
        b = np.array([5, 5, 15, 15], dtype=np.float64)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        assert _iou(a, b) == pytest.approx(25.0 / 175.0, rel=1e-6)

    def test_contained_box(self) -> None:
        outer = np.array([0, 0, 100, 100], dtype=np.float64)
        inner = np.array([25, 25, 75, 75], dtype=np.float64)
        # Intersection = inner area = 50*50 = 2500, Union = 10000
        assert _iou(outer, inner) == pytest.approx(2500.0 / 10000.0, rel=1e-6)

    def test_zero_area_box(self) -> None:
        a = np.array([5, 5, 5, 5], dtype=np.float64)
        b = np.array([0, 0, 10, 10], dtype=np.float64)
        assert _iou(a, b) == pytest.approx(0.0)


class TestCornersToBox:
    """Bounding box from corner points."""

    def test_square_corners(self) -> None:
        corners = np.array([[10, 20], [50, 20], [50, 60], [10, 60]], dtype=np.float64)
        bbox = _corners_to_bbox(corners)
        np.testing.assert_array_almost_equal(bbox, [10, 20, 50, 60])

    def test_single_point(self) -> None:
        corners = np.array([[5, 10]], dtype=np.float64)
        bbox = _corners_to_bbox(corners)
        np.testing.assert_array_almost_equal(bbox, [5, 10, 5, 10])


class TestIoUMatrix:
    """IoU matrix between tracks and detections."""

    def test_shape(self) -> None:
        tracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        dets = [np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60]), np.array([20, 20, 30, 30])]
        mat = _iou_matrix(tracks, dets)
        assert mat.shape == (2, 3)

    def test_diagonal_matches(self) -> None:
        boxes = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        mat = _iou_matrix(boxes, boxes)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tracker tests
# ---------------------------------------------------------------------------


class TestTrackerBasic:
    """Basic tracking scenarios."""

    def test_single_detection_gets_id(self) -> None:
        """A detection seen for min_hits frames gets a gate_id."""
        tracker = GateTracker(GateTrackerConfig(min_hits=2))
        det = _make_detection(100, 100)

        # Frame 1: new track, not yet confirmed
        result = tracker.update([det])
        assert len(result) == 1
        assert result[0].gate_id is None  # Only 1 hit, need 2

        # Frame 2: same position, track confirmed
        det2 = _make_detection(100, 100, timestamp_ns=1)
        result = tracker.update([det2])
        assert len(result) == 1
        assert result[0].gate_id is not None

    def test_two_detections_get_distinct_ids(self) -> None:
        """Two simultaneous detections get different IDs."""
        tracker = GateTracker(GateTrackerConfig(min_hits=1))
        d1 = _make_detection(100, 100)
        d2 = _make_detection(400, 300)

        result = tracker.update([d1, d2])
        assert len(result) == 2
        ids = {r.gate_id for r in result}
        # Both should have IDs and they should be different
        assert None not in ids
        assert len(ids) == 2

    def test_consistent_id_across_frames(self) -> None:
        """Same gate at same location keeps the same ID across frames."""
        tracker = GateTracker(GateTrackerConfig(min_hits=1))

        result1 = tracker.update([_make_detection(200, 200)])
        id1 = result1[0].gate_id

        result2 = tracker.update([_make_detection(205, 200, timestamp_ns=1)])
        id2 = result2[0].gate_id

        result3 = tracker.update([_make_detection(210, 200, timestamp_ns=2)])
        id3 = result3[0].gate_id

        assert id1 == id2 == id3

    def test_empty_frame_no_crash(self) -> None:
        """Empty detection list doesn't crash."""
        tracker = GateTracker()
        result = tracker.update([])
        assert result == []

    def test_empty_then_detection(self) -> None:
        """Tracker handles transition from empty to populated frames."""
        tracker = GateTracker(GateTrackerConfig(min_hits=1))
        tracker.update([])
        tracker.update([])
        result = tracker.update([_make_detection(300, 300)])
        assert len(result) == 1


class TestTrackerAging:
    """Track aging and pruning."""

    def test_track_pruned_after_max_age(self) -> None:
        """Track that disappears is pruned after max_age frames."""
        tracker = GateTracker(GateTrackerConfig(max_age=3, min_hits=1))

        # Create a track
        tracker.update([_make_detection(100, 100)])
        assert tracker.active_track_count == 1

        # Detection disappears for max_age + 1 frames
        for _ in range(4):
            tracker.update([])

        assert tracker.active_track_count == 0

    def test_track_survives_short_gap(self) -> None:
        """Track survives a gap shorter than max_age."""
        tracker = GateTracker(GateTrackerConfig(max_age=5, min_hits=1))

        tracker.update([_make_detection(100, 100)])
        assert tracker.active_track_count == 1

        # Miss 3 frames
        for _ in range(3):
            tracker.update([])

        # Track should still be alive
        assert tracker.active_track_count == 1

        # Re-appear
        result = tracker.update([_make_detection(100, 100)])
        assert len(result) == 1
        assert tracker.active_track_count >= 1


class TestTrackerMinHits:
    """min_hits threshold suppresses transient detections."""

    def test_single_hit_not_reported(self) -> None:
        """With min_hits=3, a detection seen once gets no ID."""
        tracker = GateTracker(GateTrackerConfig(min_hits=3))
        result = tracker.update([_make_detection(100, 100)])
        assert result[0].gate_id is None

    def test_hits_accumulate(self) -> None:
        """Detection seen min_hits times gets an ID."""
        tracker = GateTracker(GateTrackerConfig(min_hits=3))

        for i in range(2):
            result = tracker.update([_make_detection(100, 100, timestamp_ns=i)])
            assert result[0].gate_id is None

        # 3rd hit → confirmed
        result = tracker.update([_make_detection(100, 100, timestamp_ns=2)])
        assert result[0].gate_id is not None


class TestTrackerReset:
    """Reset clears all state."""

    def test_reset_clears_tracks(self) -> None:
        tracker = GateTracker(GateTrackerConfig(min_hits=1))
        tracker.update([_make_detection(100, 100)])
        assert tracker.active_track_count == 1

        tracker.reset()
        assert tracker.active_track_count == 0

    def test_ids_restart_after_reset(self) -> None:
        tracker = GateTracker(GateTrackerConfig(min_hits=1))

        r1 = tracker.update([_make_detection(100, 100)])
        first_id = r1[0].gate_id

        tracker.reset()

        r2 = tracker.update([_make_detection(200, 200)])
        second_id = r2[0].gate_id

        # After reset, IDs restart from 0
        assert first_id == second_id == 0


class TestGateMapAssignment:
    """Gate map ID assignment via 2D centroid proximity."""

    def test_assign_closest_map_gate(self) -> None:
        """Tracks are assigned the closest projected gate map ID."""
        tracker = GateTracker(GateTrackerConfig(min_hits=1))

        # Create two tracks at different positions
        tracker.update([_make_detection(100, 100), _make_detection(500, 300)])
        tracker.update([_make_detection(100, 100), _make_detection(500, 300)])

        # Projected gate map centroids
        gate_positions_2d = {
            0: np.array([110.0, 110.0]),  # Close to first track
            1: np.array([490.0, 290.0]),  # Close to second track
        }

        tracker.assign_gate_map_ids(gate_positions_2d)

        # Now update to get IDs back
        result = tracker.update(
            [_make_detection(100, 100, timestamp_ns=2), _make_detection(500, 300, timestamp_ns=2)]
        )

        ids = sorted([r.gate_id for r in result if r.gate_id is not None])
        assert 0 in ids
        assert 1 in ids

    def test_empty_map_is_noop(self) -> None:
        """Empty gate map doesn't crash."""
        tracker = GateTracker(GateTrackerConfig(min_hits=1))
        tracker.update([_make_detection(100, 100)])
        tracker.assign_gate_map_ids({})
        # No crash, no map IDs assigned


class TestTrackerWithDetector:
    """Integration: tracker works with ClassicalDetector output format."""

    def test_detector_output_compatible(self) -> None:
        """GateDetection from ClassicalDetector is directly usable by tracker."""
        from drone_racing.perception.classical import ClassicalDetector, ClassicalDetectorConfig

        config = ClassicalDetectorConfig(
            hsv_lower=(0, 100, 100),
            hsv_upper=(10, 255, 255),
            morph_kernel_size=3,
            min_contour_area=200.0,
            max_contour_area=500_000.0,
            min_confidence=0.1,
        )
        detector = ClassicalDetector(config)

        # Make synthetic image with red rectangle
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        img[200:280, 260:380] = (255, 0, 0)

        dets = detector.detect(img, timestamp_ns=1000)
        assert len(dets) >= 1

        # Feed to tracker
        tracker = GateTracker(GateTrackerConfig(min_hits=1))
        tracked = tracker.update(dets)
        assert len(tracked) >= 1
        assert tracked[0].gate_id is not None
