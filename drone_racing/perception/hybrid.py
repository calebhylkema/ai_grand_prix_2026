"""Hybrid gate detector — neural detection + classical corner refinement.

Uses the neural YOLO detector for robust gate finding (99%+ recall), then
crops each detected region and applies classical HSV-based corner extraction
for precise PnP-grade corners.  This gives MPCC-compatible pose accuracy
without requiring real sim data to retrain keypoints.

Pipeline::

    RGB image
      → Neural inference → bounding boxes (gate locations)
      → For each box:
          → Crop + pad from original image
          → HSV threshold on gate colour range
          → Morphological cleanup
          → Find contours → polygon approximation → 4 corners
          → Map corners back to full-image coordinates
      → Output GateDetection(s) with precise corners

Target latency: <22 ms per frame (17 ms neural + 5 ms classical refinement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from drone_racing.perception.classical import (
    ClassicalDetectorConfig,
    _order_corners_clockwise,
)
from drone_racing.perception.neural import NeuralDetector, NeuralDetectorConfig
from drone_racing.utils.types import GateDetection

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridDetectorConfig:
    """Configuration for the hybrid neural + classical detector.

    Parameters
    ----------
    neural : NeuralDetectorConfig
        Neural detector configuration.
    classical : ClassicalDetectorConfig
        Classical corner extraction configuration.
    crop_margin : float
        Fractional margin to add around the neural bounding box when
        cropping (0.2 = 20% extra on each side).
    min_crop_size : int
        Minimum crop dimension in pixels (skip if too small).
    """

    neural: NeuralDetectorConfig = field(default_factory=NeuralDetectorConfig)
    classical: ClassicalDetectorConfig = field(default_factory=ClassicalDetectorConfig)
    crop_margin: float = 0.2
    min_crop_size: int = 30


def _extract_bboxes(
    neural_detector: NeuralDetector,
    image: np.ndarray,
) -> list[np.ndarray]:
    """Run neural inference and return bounding boxes in original image coords.

    Parameters
    ----------
    neural_detector : NeuralDetector
        Loaded neural detector.
    image : np.ndarray
        RGB image (H, W, 3), uint8.

    Returns
    -------
    list[np.ndarray]
        Each entry is [x1, y1, x2, y2] in original image coordinates.
    """
    if not neural_detector._loaded or neural_detector._session is None:
        return []

    ih, iw = image.shape[:2]
    n_kp = neural_detector._cfg.num_keypoints

    from drone_racing.perception.neural import _letterbox, _nms

    letterboxed, scale, (pad_x, pad_y) = _letterbox(image, neural_detector._cfg.input_size)
    blob = letterboxed.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    outputs = neural_detector._session.run(
        neural_detector._output_names, {neural_detector._input_name: blob}
    )
    raw = outputs[0]

    expected_cols = 5 + n_kp * 3
    if raw.ndim == 3:
        if raw.shape[1] == expected_cols:
            raw = raw.transpose(0, 2, 1)
        preds = raw[0]
    else:
        preds = raw

    if len(preds) == 0 or preds.shape[1] < expected_cols:
        return []

    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]
    conf = preds[:, 4]

    mask = conf >= neural_detector._cfg.conf_threshold
    if not np.any(mask):
        return []

    cx, cy, w, h, conf = cx[mask], cy[mask], w[mask], h[mask], conf[mask]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes, conf, neural_detector._cfg.iou_threshold)
    if not keep:
        return []

    result: list[np.ndarray] = []
    for idx in keep:
        # Convert from letterboxed coords to original image coords
        bx1 = (boxes[idx, 0] - pad_x) / scale
        by1 = (boxes[idx, 1] - pad_y) / scale
        bx2 = (boxes[idx, 2] - pad_x) / scale
        by2 = (boxes[idx, 3] - pad_y) / scale

        bx1 = max(0.0, min(float(bx1), iw - 1))
        by1 = max(0.0, min(float(by1), ih - 1))
        bx2 = max(0.0, min(float(bx2), iw - 1))
        by2 = max(0.0, min(float(by2), ih - 1))

        result.append(np.array([bx1, by1, bx2, by2]))

    return result


def _refine_corners_in_crop(
    image: np.ndarray,
    bbox: np.ndarray,
    cfg: ClassicalDetectorConfig,
    margin: float,
    min_crop_size: int,
    morph_kernel: np.ndarray,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
) -> np.ndarray | None:
    """Extract precise gate corners from a cropped region of the image.

    Parameters
    ----------
    image : np.ndarray
        Full RGB image (H, W, 3), uint8.
    bbox : np.ndarray
        [x1, y1, x2, y2] bounding box in image coordinates.
    cfg : ClassicalDetectorConfig
        Classical detector parameters.
    margin : float
        Fractional margin to add around the bbox.
    min_crop_size : int
        Minimum crop dimension.
    morph_kernel : np.ndarray
        Pre-built morphological kernel.
    hsv_lower : np.ndarray
        HSV lower bound.
    hsv_upper : np.ndarray
        HSV upper bound.

    Returns
    -------
    np.ndarray | None
        (4, 2) or (8, 2) ordered corners in full-image coordinates,
        or None if corner extraction fails.
    """
    ih, iw = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add margin
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(iw, int(x2 + mx))
    cy2 = min(ih, int(y2 + my))

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w < min_crop_size or crop_h < min_crop_size:
        return None

    crop = image[cy1:cy2, cx1:cx2]

    # HSV threshold
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    # Find contours with hierarchy (for inner corners)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the best quad contour (largest area that approximates to 4 vertices)
    best_outer: np.ndarray | None = None
    best_area = 0.0
    best_parent_idx = -1
    crop_area = float(crop_w * crop_h)

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Scale area thresholds relative to crop (not full image)
        if area < crop_area * 0.05 or area > crop_area * 0.95:
            continue

        # Skip child contours
        if hierarchy is not None and hierarchy[0][idx][3] >= 0:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = cfg.approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) < 4 or len(approx) > cfg.max_approx_vertices:
            continue

        if area > best_area:
            best_area = area
            best_outer = approx.reshape(-1, 2).astype(np.float64)
            best_parent_idx = idx

    if best_outer is None or len(best_outer) != 4:
        return None

    outer = _order_corners_clockwise(best_outer)

    # Try to find inner corners from child contour
    inner: np.ndarray | None = None
    if hierarchy is not None and best_parent_idx >= 0:
        child_idx = hierarchy[0][best_parent_idx][2]
        if child_idx >= 0:
            child = contours[child_idx]
            child_area = cv2.contourArea(child)
            if best_area > 0:
                area_ratio = child_area / best_area
                if 0.2 <= area_ratio <= 0.85:
                    perimeter = cv2.arcLength(child, closed=True)
                    epsilon = cfg.approx_epsilon_factor * perimeter
                    child_approx = cv2.approxPolyDP(child, epsilon, closed=True)
                    if len(child_approx) == 4:
                        inner_pts = child_approx.reshape(-1, 2).astype(np.float64)
                        inner = _order_corners_clockwise(inner_pts)

    # Map corners back to full-image coordinates
    offset = np.array([cx1, cy1], dtype=np.float64)
    outer = outer + offset
    if inner is not None:
        inner = inner + offset
        return np.vstack([outer, inner])
    return outer


class HybridDetector:
    """Neural detection + classical corner refinement for precise PnP.

    Uses the neural YOLO model for robust gate localisation, then
    refines corner positions using classical HSV-based extraction
    on the cropped gate region.

    Parameters
    ----------
    config : HybridDetectorConfig | None
        Detection parameters. Uses defaults if None.
    camera_config : CameraConfig | None
        Camera config for undistortion. If None, no undistortion is applied.
    """

    def __init__(
        self,
        config: HybridDetectorConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        self._cfg = config or HybridDetectorConfig()
        self._camera_cfg = camera_config
        self._neural = NeuralDetector(self._cfg.neural)

        # Pre-build classical processing kernels
        ks = self._cfg.classical.morph_kernel_size
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        self._hsv_lower = np.array(self._cfg.classical.hsv_lower, dtype=np.uint8)
        self._hsv_upper = np.array(self._cfg.classical.hsv_upper, dtype=np.uint8)

        # Pre-build undistortion maps
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

    def load(self) -> None:
        """Load the neural model. Call once before detect()."""
        self._neural.load()

    def detect(
        self,
        image: np.ndarray,
        timestamp_ns: int,
    ) -> list[GateDetection]:
        """Detect gates using neural localisation + classical corner refinement.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        timestamp_ns : int
            Camera capture timestamp (ns).

        Returns
        -------
        list[GateDetection]
            Detected gates with precise corners, ordered by confidence.
        """
        # Undistort if needed
        if self._undistort_map1 is not None and self._undistort_map2 is not None:
            image = cv2.remap(image, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)

        # Step 1: Neural inference for bounding boxes
        bboxes = _extract_bboxes(self._neural, image)
        if not bboxes:
            logger.debug("Hybrid detector: no neural detections")
            return []

        # Step 2: Refine corners in each detected region
        detections: list[GateDetection] = []
        for bbox in bboxes:
            corners = _refine_corners_in_crop(
                image,
                bbox,
                self._cfg.classical,
                self._cfg.crop_margin,
                self._cfg.min_crop_size,
                self._morph_kernel,
                self._hsv_lower,
                self._hsv_upper,
            )
            if corners is None:
                logger.debug("Hybrid: classical refinement failed for bbox %s", bbox)
                continue

            # Confidence: base for hybrid detection, boosted for 8 corners
            confidence = 0.7
            if len(corners) >= 8:
                confidence = min(1.0, confidence + 0.15)

            detections.append(
                GateDetection(
                    timestamp_ns=timestamp_ns,
                    corners_2d=corners,
                    confidence=confidence,
                )
            )

        # Sort by corner count (prefer 8-corner), then confidence
        detections.sort(key=lambda d: (len(d.corners_2d), d.confidence), reverse=True)

        logger.debug(
            "Hybrid detector: %d neural bbox(es) → %d refined detection(s)",
            len(bboxes),
            len(detections),
        )
        return detections
