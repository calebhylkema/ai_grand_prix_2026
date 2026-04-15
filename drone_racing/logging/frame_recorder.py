"""Frame + telemetry recorder for vision training-data capture.

Writes each camera frame to disk as a JPEG plus one line of JSON metadata
containing the matching state estimate and any gate detections.  The output
is designed for offline auto-labelling: given recorded ESKF poses and the
discovered gate map, gate corners can be back-projected to produce keypoint
ground truth for neural-detector retraining.

Output layout::

    <output_dir>/
        frames/
            000000.jpg
            000001.jpg
            ...
        meta.jsonl       # one JSON object per recorded frame
        manifest.json    # camera intrinsics + record summary

This module is deliberately synchronous — JPEG encoding at 60 Hz costs
~1-2 ms per frame, well inside the 10 ms control-loop budget.  Switch to a
background queue only if profiling shows the recorder is the bottleneck.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from drone_racing.utils.types import CameraConfig, GateDetection, State

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameRecorderConfig:
    """Recorder parameters.

    Parameters
    ----------
    jpeg_quality : int
        libjpeg quality 0-100; 85 balances size vs. training utility.
    stride : int
        Record every Nth frame (1 = every frame).  Useful to cap disk usage
        when the camera runs faster than training needs.
    max_frames : int | None
        Stop recording after this many frames.  ``None`` = unlimited.
    """

    jpeg_quality: int = 85
    stride: int = 1
    max_frames: int | None = None


class FrameRecorder:
    """Writes frames + synchronized telemetry to disk for offline training.

    Parameters
    ----------
    output_dir : str | Path
        Directory to create.  Overwrites existing ``meta.jsonl`` if present.
    camera_config : CameraConfig
        Used to populate the manifest (intrinsics, resolution).
    config : FrameRecorderConfig | None
        Recording parameters.
    """

    def __init__(
        self,
        output_dir: str | Path,
        camera_config: CameraConfig,
        config: FrameRecorderConfig | None = None,
    ) -> None:
        self._dir = Path(output_dir)
        self._frames_dir = self._dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = config or FrameRecorderConfig()
        self._camera = camera_config

        self._meta_path = self._dir / "meta.jsonl"
        # Truncate so two runs to the same dir don't produce a mixed log.
        self._meta_fh = self._meta_path.open("w", encoding="utf-8")
        self._frame_idx = 0
        self._seen = 0
        self._closed = False

    def record(
        self,
        frame: np.ndarray,
        timestamp_ns: int,
        state: State,
        detections: list[GateDetection] | None = None,
    ) -> bool:
        """Record a frame with its matching state + detections.

        Parameters
        ----------
        frame : np.ndarray
            (H, W, 3) uint8 BGR image (OpenCV convention).
        timestamp_ns : int
            Camera capture timestamp.
        state : State
            ESKF state at (or closest to) the frame timestamp.
        detections : list[GateDetection] | None
            Detected gates in this frame, pre-tracker or post-tracker.

        Returns
        -------
        bool
            True if the frame was written, False if skipped (stride / cap).
        """
        if self._closed:
            return False

        self._seen += 1
        if self._cfg.stride > 1 and (self._seen - 1) % self._cfg.stride != 0:
            return False
        if self._cfg.max_frames is not None and self._frame_idx >= self._cfg.max_frames:
            return False

        name = f"{self._frame_idx:06d}.jpg"
        frame_path = self._frames_dir / name
        ok = cv2.imwrite(
            str(frame_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._cfg.jpeg_quality)],
        )
        if not ok:
            logger.warning("FrameRecorder: cv2.imwrite failed for %s", frame_path)
            return False

        meta = {
            "idx": self._frame_idx,
            "frame": f"frames/{name}",
            "timestamp_ns": int(timestamp_ns),
            "state": {
                "position": state.position.tolist(),
                "velocity": state.velocity.tolist(),
                "orientation": state.orientation.tolist(),
            },
            "detections": [_detection_to_dict(d) for d in (detections or [])],
        }
        self._meta_fh.write(json.dumps(meta, separators=(",", ":")) + "\n")
        self._meta_fh.flush()
        self._frame_idx += 1
        return True

    def close(self) -> None:
        """Flush metadata and write the manifest."""
        if self._closed:
            return
        self._closed = True
        self._meta_fh.close()

        manifest = {
            "frames_written": self._frame_idx,
            "frames_seen": self._seen,
            "stride": self._cfg.stride,
            "jpeg_quality": self._cfg.jpeg_quality,
            "camera": {
                "width": int(self._camera.width),
                "height": int(self._camera.height),
                "fx": float(self._camera.fx),
                "fy": float(self._camera.fy),
                "cx": float(self._camera.cx),
                "cy": float(self._camera.cy),
                "distortion": [float(x) for x in np.asarray(self._camera.distortion).ravel()],
            },
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(
            "FrameRecorder: wrote %d frames (of %d seen) to %s",
            self._frame_idx,
            self._seen,
            self._dir,
        )

    def __enter__(self) -> FrameRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        """Number of frames actually persisted to disk."""
        return self._frame_idx


def _detection_to_dict(det: GateDetection) -> dict[str, object]:
    """Serialize a :class:`GateDetection` to a JSON-safe dict."""
    return {
        "corners_2d": det.corners_2d.tolist(),
        "confidence": float(det.confidence),
        "gate_id": None if det.gate_id is None else int(det.gate_id),
        "timestamp_ns": int(det.timestamp_ns),
    }
