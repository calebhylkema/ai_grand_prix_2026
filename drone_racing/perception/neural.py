"""Neural gate detector using a YOLOv8-nano keypoint model via ONNX Runtime.

Pipeline::

    RGB image (H, W, 3)
      → Letterbox resize to model input size
      → Normalise to [0, 1] float32
      → ONNX Runtime inference
      → Decode YOLO keypoint outputs
      → NMS (Non-Maximum Suppression)
      → Extract 8 keypoints (4 outer + 4 inner corners)
      → Scale keypoints back to original image coordinates
      → Output GateDetection(s)

Target latency: <17 ms per frame on Jetson Orin NX.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from drone_racing.utils.types import GateDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NeuralDetectorConfig:
    """Parameters for the ONNX-based YOLOv8 keypoint gate detector.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    input_size : tuple[int, int]
        Model input size (width, height).
    conf_threshold : float
        Minimum detection confidence to keep.
    iou_threshold : float
        NMS IoU threshold.
    num_keypoints : int
        Number of keypoints per detection (4 outer + 4 inner = 8).
    providers : list[str]
        ONNX Runtime execution providers, in priority order.
    """

    model_path: str = "models/gate_keypoint.onnx"
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    num_keypoints: int = 8
    providers: list[str] = field(
        default_factory=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )


# ---------------------------------------------------------------------------
# Letterbox preprocessing
# ---------------------------------------------------------------------------


def _letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize image with letterboxing (pad to maintain aspect ratio).

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) input image.
    target_size : tuple[int, int]
        (width, height) target dimensions.

    Returns
    -------
    tuple[np.ndarray, float, tuple[float, float]]
        (letterboxed_image, scale, (pad_x, pad_y))
    """
    ih, iw = image.shape[:2]
    tw, th = target_size

    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    pad_x = (tw - nw) / 2.0
    pad_y = (th - nh) / 2.0

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # Ensure exact target size (rounding may be off by 1)
    if padded.shape[0] != th or padded.shape[1] != tw:
        padded = cv2.resize(padded, (tw, th))

    return padded, scale, (pad_x, pad_y)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Non-Maximum Suppression.

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4) bounding boxes [x1, y1, x2, y2].
    scores : np.ndarray
        (N,) confidence scores.
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    list[int]
        Indices of kept detections.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)

        order = rest[iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class NeuralDetector:
    """ONNX-based YOLOv8 keypoint gate detector.

    Parameters
    ----------
    config : NeuralDetectorConfig | None
        Detection parameters. Uses defaults if None.
    """

    def __init__(self, config: NeuralDetectorConfig | None = None) -> None:
        self._cfg = config or NeuralDetectorConfig()
        self._session = None
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._loaded = False

    def load(self) -> None:
        """Load the ONNX model. Call once before detect().

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ImportError
            If onnxruntime is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            msg = (
                "onnxruntime is required for NeuralDetector. Install with: pip install onnxruntime"
            )
            raise ImportError(msg) from e

        model_path = Path(self._cfg.model_path)
        if not model_path.exists():
            msg = f"ONNX model not found: {model_path}"
            raise FileNotFoundError(msg)

        self._session = ort.InferenceSession(
            str(model_path),
            providers=self._cfg.providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._loaded = True

        logger.info(
            "NeuralDetector loaded: %s (providers: %s)",
            model_path.name,
            self._session.get_providers(),
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

        Raises
        ------
        RuntimeError
            If the model has not been loaded via load().
        """
        if not self._loaded or self._session is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        ih, iw = image.shape[:2]
        n_kp = self._cfg.num_keypoints

        # --- Preprocess ---
        letterboxed, scale, (pad_x, pad_y) = _letterbox(image, self._cfg.input_size)

        # HWC RGB → CHW float32 [0, 1]
        blob = letterboxed.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        # --- Inference ---
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        raw = outputs[0]  # (1, num_predictions, 5 + n_kp*3)

        # --- Decode ---
        detections = self._decode_outputs(raw, scale, pad_x, pad_y, iw, ih, n_kp, timestamp_ns)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug("Neural detector: found %d gate(s)", len(detections))
        return detections

    def _decode_outputs(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        orig_w: int,
        orig_h: int,
        n_kp: int,
        timestamp_ns: int,
    ) -> list[GateDetection]:
        """Decode raw YOLO outputs into GateDetections.

        YOLOv8 pose output format: (1, 56, N) where 56 = 4 bbox + 1 conf + 8*3 kp + 3*kp_conf
        Actually YOLOv8 pose: (1, N, 56) after transpose, where:
        - [0:4] = cx, cy, w, h (in input coords)
        - [4] = objectness confidence
        - [5:5+n_kp*3] = kp_x, kp_y, kp_conf for each keypoint

        Parameters
        ----------
        raw : np.ndarray
            Raw model output.
        scale : float
            Letterbox scale factor.
        pad_x, pad_y : float
            Letterbox padding.
        orig_w, orig_h : int
            Original image dimensions.
        n_kp : int
            Number of keypoints.
        timestamp_ns : int
            Frame timestamp.

        Returns
        -------
        list[GateDetection]
            Decoded and NMS-filtered detections.
        """
        # YOLOv8 outputs (1, features, N_predictions) — transpose to (N, features)
        expected_cols = 5 + n_kp * 3
        if raw.ndim == 3:
            # If dim 1 matches expected features, transpose so predictions are rows
            if raw.shape[1] == expected_cols:
                raw = raw.transpose(0, 2, 1)
            preds = raw[0]
        else:
            preds = raw

        if len(preds) == 0:
            return []

        if preds.shape[1] < expected_cols:
            logger.warning(
                "Model output has %d columns, expected %d", preds.shape[1], expected_cols
            )
            return []

        # Extract bbox and confidence
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        conf = preds[:, 4]

        # Filter by confidence
        mask = conf >= self._cfg.conf_threshold
        if not np.any(mask):
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        conf = conf[mask]
        kp_data = preds[mask, 5 : 5 + n_kp * 3]

        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = _nms(boxes, conf, self._cfg.iou_threshold)
        if not keep:
            return []

        detections: list[GateDetection] = []
        for idx in keep:
            # Rescale bbox and keypoints from letterboxed coords to original image
            kp_raw = kp_data[idx].reshape(n_kp, 3)
            corners = np.zeros((n_kp, 2), dtype=np.float64)
            visible_count = 0

            for k in range(n_kp):
                kx = (kp_raw[k, 0] - pad_x) / scale
                ky = (kp_raw[k, 1] - pad_y) / scale
                kp_conf = kp_raw[k, 2]

                corners[k, 0] = np.clip(kx, 0, orig_w - 1)
                corners[k, 1] = np.clip(ky, 0, orig_h - 1)

                if kp_conf > 0.5:
                    visible_count += 1

            # Only emit if enough keypoints are visible
            if visible_count < 4:
                continue

            # Use only visible keypoints
            kp_confs = kp_raw[:, 2]
            visible_mask = kp_confs > 0.5
            visible_corners = corners[visible_mask]

            det_conf = float(conf[idx])
            # Boost confidence if all 8 keypoints are visible
            if visible_count >= 8:
                det_conf = min(1.0, det_conf + 0.1)

            detections.append(
                GateDetection(
                    timestamp_ns=timestamp_ns,
                    corners_2d=visible_corners,
                    confidence=det_conf,
                )
            )

        return detections
