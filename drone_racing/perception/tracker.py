"""Gate tracker — assigns consistent IDs to gate detections across frames.

Uses IoU (Intersection over Union) on bounding boxes and the Hungarian
algorithm (scipy ``linear_sum_assignment``) for optimal frame-to-frame
matching.  Unmatched detections are assigned new track IDs; tracks that
go unmatched for several frames are dropped.

Typical usage::

    tracker = GateTracker(config)
    for frame in video:
        raw_detections = detector.detect(frame, ts)
        tracked = tracker.update(raw_detections)
        # tracked detections now have .gate_id set

The tracker also supports optional gate-map matching: once a track has
been associated with a known gate in the course map (by proximity to
the expected image location), its ``gate_id`` is set to the map gate ID
rather than an arbitrary track ID.  This enables ``PnPSolver.solve_multiple``
to look up the correct world-frame gate pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from drone_racing.utils.types import GateDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GateTrackerConfig:
    """Tuneable parameters for the gate tracker.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider a detection–track pair a match (0–1).
    max_age : int
        Number of consecutive frames a track can miss before deletion.
    min_hits : int
        Minimum number of consecutive hits before a track is reported
        (suppresses transient false positives).
    """

    iou_threshold: float = 0.2
    max_age: int = 5
    min_hits: int = 2


# ---------------------------------------------------------------------------
# Internal track state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Track:
    """Internal state for a single tracked gate."""

    track_id: int
    """Unique ID assigned by the tracker."""

    bbox: np.ndarray
    """Last observed axis-aligned bounding box [x1, y1, x2, y2]."""

    corners_2d: np.ndarray
    """Last observed corner pixels (N, 2)."""

    confidence: float
    """Most recent detection confidence."""

    age: int = 0
    """Frames since this track was last matched (0 = matched this frame)."""

    hits: int = 1
    """Total number of frames this track has been matched."""

    gate_map_id: int | None = None
    """If matched to a gate in the course map, the map gate ID."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corners_to_bbox(corners: np.ndarray) -> np.ndarray:
    """Compute axis-aligned bounding box from corner points.

    Parameters
    ----------
    corners : np.ndarray
        (N, 2) pixel coordinates.

    Returns
    -------
    np.ndarray
        [x1, y1, x2, y2] bounding box.
    """
    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection over Union between two axis-aligned bounding boxes.

    Parameters
    ----------
    bbox_a : np.ndarray
        [x1, y1, x2, y2] first box.
    bbox_b : np.ndarray
        [x1, y1, x2, y2] second box.

    Returns
    -------
    float
        IoU in [0, 1].
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(0.0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    union = area_a + area_b - intersection

    if union < 1e-12:
        return 0.0
    return float(intersection / union)


def _iou_matrix(track_bboxes: list[np.ndarray], det_bboxes: list[np.ndarray]) -> np.ndarray:
    """Compute the IoU matrix between all tracks and detections.

    Parameters
    ----------
    track_bboxes : list[np.ndarray]
        List of track bounding boxes.
    det_bboxes : list[np.ndarray]
        List of detection bounding boxes.

    Returns
    -------
    np.ndarray
        (T, D) IoU matrix.
    """
    n_tracks = len(track_bboxes)
    n_dets = len(det_bboxes)
    matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
    for i in range(n_tracks):
        for j in range(n_dets):
            matrix[i, j] = _iou(track_bboxes[i], det_bboxes[j])
    return matrix


# ---------------------------------------------------------------------------
# Gate tracker
# ---------------------------------------------------------------------------


class GateTracker:
    """Multi-object tracker for gate detections using IoU + Hungarian matching.

    Parameters
    ----------
    config : GateTrackerConfig | None
        Tracker parameters.  Uses defaults if None.
    """

    def __init__(self, config: GateTrackerConfig | None = None) -> None:
        self._cfg = config or GateTrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id: int = 0

    @property
    def tracks(self) -> list[_Track]:
        """Current active tracks (read-only view)."""
        return list(self._tracks)

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 0

    def update(self, detections: list[GateDetection]) -> list[GateDetection]:
        """Match detections to existing tracks and assign gate IDs.

        Parameters
        ----------
        detections : list[GateDetection]
            Raw detections from the current frame (no ``gate_id`` set).

        Returns
        -------
        list[GateDetection]
            Detections with ``gate_id`` assigned for tracks that have
            met the ``min_hits`` threshold.  Detections that could not
            be reliably tracked are returned with ``gate_id = None``.
        """
        if len(detections) == 0:
            # Age all tracks, prune dead ones
            self._age_and_prune()
            return []

        # Compute bounding boxes for detections
        det_bboxes = [_corners_to_bbox(d.corners_2d) for d in detections]

        if len(self._tracks) == 0:
            # No existing tracks — create one per detection
            for i, det in enumerate(detections):
                self._create_track(det, det_bboxes[i])
            self._age_and_prune()
            return self._assign_ids(detections)

        # --- Build cost matrix (negative IoU for minimisation) ---
        track_bboxes = [t.bbox for t in self._tracks]
        iou_mat = _iou_matrix(track_bboxes, det_bboxes)

        # Hungarian assignment on cost = 1 - IoU (minimise cost)
        cost_matrix = 1.0 - iou_mat
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # --- Process matches ---
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for row, col in zip(row_indices, col_indices, strict=True):
            if iou_mat[row, col] >= self._cfg.iou_threshold:
                # Valid match: update track
                track = self._tracks[row]
                track.bbox = det_bboxes[col]
                track.corners_2d = detections[col].corners_2d.copy()
                track.confidence = detections[col].confidence
                track.age = 0
                track.hits += 1
                matched_tracks.add(row)
                matched_dets.add(col)

        # --- Handle unmatched detections: create new tracks ---
        for j in range(len(detections)):
            if j not in matched_dets:
                self._create_track(detections[j], det_bboxes[j])

        # --- Age and prune ---
        self._age_and_prune(matched_tracks)

        return self._assign_ids(detections)

    def assign_gate_map_ids(
        self,
        gate_positions_2d: dict[int, np.ndarray],
    ) -> None:
        """Associate tracks with known gates from the course map.

        For each active track, finds the closest gate map entry by
        2D centroid distance and assigns the map gate ID to the track.
        This enables ``PnPSolver.solve_multiple`` to look up the
        correct world-frame gate pose.

        Parameters
        ----------
        gate_positions_2d : dict[int, np.ndarray]
            Mapping of gate_map_id → expected 2D centroid position (pixels).
            Typically obtained by projecting gate world positions through
            the camera model using the current state estimate.
        """
        if not gate_positions_2d:
            return

        map_ids = list(gate_positions_2d.keys())
        map_centroids = np.array([gate_positions_2d[k] for k in map_ids], dtype=np.float64)

        for track in self._tracks:
            if track.hits < self._cfg.min_hits:
                continue
            # Track centroid from bounding box
            track_cx = (track.bbox[0] + track.bbox[2]) / 2.0
            track_cy = (track.bbox[1] + track.bbox[3]) / 2.0
            track_centroid = np.array([track_cx, track_cy])

            dists = np.linalg.norm(map_centroids - track_centroid[np.newaxis, :], axis=1)
            best_idx = int(np.argmin(dists))
            track.gate_map_id = map_ids[best_idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, detection: GateDetection, bbox: np.ndarray) -> None:
        """Create a new track from a detection."""
        track = _Track(
            track_id=self._next_id,
            bbox=bbox.copy(),
            corners_2d=detection.corners_2d.copy(),
            confidence=detection.confidence,
        )
        self._tracks.append(track)
        self._next_id += 1

    def _age_and_prune(self, matched_indices: set[int] | None = None) -> None:
        """Increment age of unmatched tracks and remove stale ones."""
        if matched_indices is None:
            matched_indices = set()

        for i, track in enumerate(self._tracks):
            if i not in matched_indices:
                track.age += 1

        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.age <= self._cfg.max_age]
        pruned = before - len(self._tracks)
        if pruned > 0:
            logger.debug("Tracker: pruned %d stale tracks", pruned)

    def _assign_ids(self, detections: list[GateDetection]) -> list[GateDetection]:
        """Assign gate_id to detections based on matching tracks.

        Only tracks with ``hits >= min_hits`` are considered reliable
        enough to report.

        Parameters
        ----------
        detections : list[GateDetection]
            Current frame's detections.

        Returns
        -------
        list[GateDetection]
            Same detections with ``gate_id`` set where a reliable
            track match exists.
        """
        # Build a lookup: bbox → track (using centroid proximity)
        for det in detections:
            det_bbox = _corners_to_bbox(det.corners_2d)
            det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
            det_cy = (det_bbox[1] + det_bbox[3]) / 2.0

            best_track: _Track | None = None
            best_dist = float("inf")

            for track in self._tracks:
                if track.hits < self._cfg.min_hits:
                    continue
                track_cx = (track.bbox[0] + track.bbox[2]) / 2.0
                track_cy = (track.bbox[1] + track.bbox[3]) / 2.0
                dist = ((det_cx - track_cx) ** 2 + (det_cy - track_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is not None and best_dist < 50.0:
                # Use gate_map_id if available, otherwise track_id
                det.gate_id = (
                    best_track.gate_map_id
                    if best_track.gate_map_id is not None
                    else best_track.track_id
                )

        return detections
