"""Classical gate detector for VQ1 (desaturated environment, highlighted gates).

Pipeline::

    RGB image
      → Convert to HSV
      → Threshold on gate colour range
      → Morphological cleanup (open → close)
      → Find contours
      → Filter by area and aspect ratio
      → Approximate contour to quadrilateral
      → Order corners (top-left, top-right, bottom-right, bottom-left)
      → Output GateDetection(s)

Target latency: <5 ms per frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.utils.types import GateDetection

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration for the classical detector
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ClassicalDetectorConfig:
    """Tuneable parameters for HSV-based gate detection.

    These will need adjustment for VQ1's specific gate colours and lighting.
    Defaults are for a bright-coloured gate on a desaturated background.
    """

    # HSV thresholds (OpenCV convention: H 0-179, S 0-255, V 0-255)
    hsv_lower: tuple[int, int, int] = (0, 80, 80)
    hsv_upper: tuple[int, int, int] = (179, 255, 255)

    # Morphological kernel size (pixels)
    morph_kernel_size: int = 5

    # Contour area thresholds (in pixels²)
    min_contour_area: float = 500.0
    max_contour_area: float = 500_000.0

    # Aspect ratio bounds (width/height) for gate candidates
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 3.0

    # Maximum corner count for polygon approximation
    # A gate should approximate to 4 corners
    max_approx_vertices: int = 8

    # Epsilon factor for contour approximation (fraction of perimeter)
    approx_epsilon_factor: float = 0.03

    # Minimum confidence for a detection to be emitted
    min_confidence: float = 0.3


# ---------------------------------------------------------------------------
# Corner ordering
# ---------------------------------------------------------------------------


def _order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as: top-left, top-right, bottom-right, bottom-left.

    Parameters
    ----------
    corners : np.ndarray
        (4, 2) unordered corner pixel coordinates.

    Returns
    -------
    np.ndarray
        (4, 2) ordered corners.
    """
    # Centroid
    cx, cy = corners.mean(axis=0)

    # Classify each corner relative to centroid
    top = corners[corners[:, 1] < cy]
    bottom = corners[corners[:, 1] >= cy]

    # Sort top corners by x (left then right)
    if len(top) == 2:
        top = top[np.argsort(top[:, 0])]
    elif len(top) < 2:
        # Fallback: sort all by angle from centroid
        return _order_by_angle(corners)

    # Sort bottom corners by x (right then left → BR, BL)
    if len(bottom) == 2:
        bottom = bottom[np.argsort(bottom[:, 0])]
    else:
        return _order_by_angle(corners)

    # TL, TR, BR, BL
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def _order_by_angle(corners: np.ndarray) -> np.ndarray:
    """Fallback: order corners by angle from centroid (clockwise from top-left).

    Parameters
    ----------
    corners : np.ndarray
        (N, 2) corner coordinates.

    Returns
    -------
    np.ndarray
        (N, 2) ordered corners.
    """
    cx, cy = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
    # Start from top-left (roughly -135° = -3π/4)
    # Shift so that top-left is the smallest angle
    angles_shifted = (angles + np.pi * 0.75) % (2 * np.pi)
    order = np.argsort(angles_shifted)
    return corners[order].astype(np.float64)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_confidence(
    contour: np.ndarray,
    approx: np.ndarray,
    image_area: float,
) -> float:
    """Compute a confidence score for a gate detection.

    Factors:
    - Number of vertices (4 is ideal)
    - Convexity
    - Area relative to image (not too small, not too large)
    - Aspect ratio closeness to 1.0 (square gate)

    Parameters
    ----------
    contour : np.ndarray
        Original contour.
    approx : np.ndarray
        Polygon approximation of the contour.
    image_area : float
        Total image area in pixels².

    Returns
    -------
    float
        Confidence in [0, 1].
    """
    scores: list[float] = []

    # Vertex count score: 4 is perfect, degrade with distance from 4
    n_verts = len(approx)
    vertex_score = max(0.0, 1.0 - abs(n_verts - 4) * 0.2)
    scores.append(vertex_score)

    # Convexity score
    convex = cv2.isContourConvex(approx)
    scores.append(1.0 if convex else 0.5)

    # Area score: prefer gates that fill 1-30% of image
    area = cv2.contourArea(contour)
    area_ratio = area / image_area
    if 0.01 < area_ratio < 0.30:
        area_score = 1.0
    elif 0.005 < area_ratio < 0.50:
        area_score = 0.6
    else:
        area_score = 0.3
    scores.append(area_score)

    # Aspect ratio score: closer to 1.0 is better
    rect = cv2.minAreaRect(approx)
    w, h = rect[1]
    if min(w, h) > 0:
        aspect = max(w, h) / min(w, h)
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.0) * 0.3)
    else:
        aspect_score = 0.0
    scores.append(aspect_score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class ClassicalDetector:
    """HSV-based gate detector for VQ1 environments.

    Parameters
    ----------
    config : ClassicalDetectorConfig
        Detection parameters.
    camera_config : CameraConfig | None
        Camera config for undistortion. If None, no undistortion is applied.
    """

    def __init__(
        self,
        config: ClassicalDetectorConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        self._cfg = config or ClassicalDetectorConfig()
        self._camera_cfg = camera_config

        # Pre-build morphological kernel
        ks = self._cfg.morph_kernel_size
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))

        # Pre-build undistortion maps if camera config provided
        self._undistort_map1: np.ndarray | None = None
        self._undistort_map2: np.ndarray | None = None
        if camera_config is not None:
            k = camera_config.intrinsic_matrix
            d = camera_config.distortion_coeffs
            if np.any(d != 0):
                h, w = camera_config.height, camera_config.width
                self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
                    k, d, None, k, (w, h), cv2.CV_32FC1
                )

    def detect(
        self,
        image: np.ndarray,
        timestamp_ns: int,
    ) -> list[GateDetection]:
        """Detect gates in a single RGB image.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        timestamp_ns : int
            Camera capture timestamp (ns).

        Returns
        -------
        list[GateDetection]
            Detected gates, ordered by confidence (highest first).
        """
        # --- Undistort if needed ---
        if self._undistort_map1 is not None and self._undistort_map2 is not None:
            image = cv2.remap(image, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)

        # --- Convert to HSV ---
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # --- Threshold ---
        lower = np.array(self._cfg.hsv_lower, dtype=np.uint8)
        upper = np.array(self._cfg.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # --- Morphological cleanup ---
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # --- Find contours (with hierarchy for inner corner detection) ---
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # --- Filter and extract gate candidates ---
        image_area = float(image.shape[0] * image.shape[1])
        detections: list[GateDetection] = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self._cfg.min_contour_area or area > self._cfg.max_contour_area:
                continue

            # Aspect ratio filter
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if min(w, h) < 1:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect < self._cfg.min_aspect_ratio or aspect > self._cfg.max_aspect_ratio:
                continue

            # Skip inner (child) contours — we process them via their parent
            if hierarchy is not None and hierarchy[0][idx][3] >= 0:
                continue

            # Polygon approximation
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self._cfg.approx_epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)

            n_verts = len(approx)
            if n_verts < 4 or n_verts > self._cfg.max_approx_vertices:
                continue

            # Confidence scoring
            confidence = _compute_confidence(contour, approx, image_area)
            if confidence < self._cfg.min_confidence:
                continue

            # Extract outer corners
            outer_corners = approx.reshape(-1, 2).astype(np.float64)
            if len(outer_corners) == 4:
                outer_corners = _order_corners_clockwise(outer_corners)

            # Try to find inner corners from child contour (gate hole)
            inner_corners = self._find_inner_corners(contours, hierarchy, idx, outer_corners)

            if inner_corners is not None:
                # Combine: 4 outer + 4 inner = 8 corners
                corners = np.vstack([outer_corners, inner_corners])
                # Boost confidence for 8-corner detections
                confidence = min(1.0, confidence + 0.15)
            else:
                corners = outer_corners

            detections.append(
                GateDetection(
                    timestamp_ns=timestamp_ns,
                    corners_2d=corners,
                    confidence=confidence,
                )
            )

        # Sort by confidence (highest first), prefer 8-corner detections
        detections.sort(key=lambda d: (len(d.corners_2d), d.confidence), reverse=True)

        logger.debug("Classical detector: found %d gate(s)", len(detections))
        return detections

    def _find_inner_corners(
        self,
        contours: list[np.ndarray],
        hierarchy: np.ndarray | None,
        parent_idx: int,
        outer_corners: np.ndarray,
    ) -> np.ndarray | None:
        """Extract 4 inner corners from a child contour (the gate hole).

        Parameters
        ----------
        contours : list[np.ndarray]
            All contours from findContours.
        hierarchy : np.ndarray | None
            Contour hierarchy from RETR_TREE.
        parent_idx : int
            Index of the outer (parent) contour.
        outer_corners : np.ndarray
            (4, 2) ordered outer corners for validation.

        Returns
        -------
        np.ndarray | None
            (4, 2) ordered inner corners, or None if not found.
        """
        if hierarchy is None:
            return None

        # Find the first child contour
        child_idx = hierarchy[0][parent_idx][2]
        if child_idx < 0:
            return None

        child = contours[child_idx]
        child_area = cv2.contourArea(child)
        parent_area = cv2.contourArea(contours[parent_idx])

        # Inner hole should be 30-80% of the outer area
        if parent_area < 1:
            return None
        area_ratio = child_area / parent_area
        if area_ratio < 0.2 or area_ratio > 0.85:
            return None

        # Approximate child to polygon
        perimeter = cv2.arcLength(child, closed=True)
        epsilon = self._cfg.approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(child, epsilon, closed=True)

        if len(approx) != 4:
            return None

        inner = approx.reshape(-1, 2).astype(np.float64)
        inner = _order_corners_clockwise(inner)

        # Validate: inner centroid should be close to outer centroid
        outer_c = outer_corners.mean(axis=0)
        inner_c = inner.mean(axis=0)
        if np.linalg.norm(outer_c - inner_c) > 50.0:
            return None

        return inner
