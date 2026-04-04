# Perception Module — Gate Detection & Corner Extraction

## Purpose

Detect racing gates in monocular RGB frames and extract precise corner coordinates for PnP pose estimation. This is the entry point of the autonomy pipeline — its latency and accuracy directly bound overall system performance.

## Architecture

Two detection backends, selected by config:

1. **ClassicalDetector** (VQ1): HSV thresholding → morphological cleanup → contour detection → convex hull → quad fitting. Target: <5ms per frame.
2. **NeuralDetector** (VQ2+): YOLOv8-nano keypoint model detecting 4–8 gate corners directly. Target: <17ms per frame on Jetson Orin.

Both backends output a `GateDetection` dataclass:
```python
@dataclass
class GateDetection:
    corners_2d: np.ndarray    # (N, 2) pixel coordinates, ordered clockwise from top-left
    confidence: float         # 0.0–1.0
    gate_id: int | None       # Assigned by tracker, None if unmatched
    timestamp_ns: int         # Camera capture timestamp (NOT processing time)
```

## Key Constraints

- Gate physical dimensions are in `config/gates.toml` — NEVER hardcode
- Standard gate: 1.5m × 1.5m square with 4 outer + 4 inner corners = 8 potential keypoints
- PnP needs minimum 4 non-collinear corners; system degrades gracefully with fewer
- Camera intrinsics (fx, fy, cx, cy, distortion coeffs) in `config/camera.toml`
- All image coordinates use OpenCV convention: (0,0) at top-left, x-right, y-down

## Implementation Notes

- Use ONNX Runtime (not raw PyTorch) for inference in production — 2-3x faster on Jetson via TensorRT
- Pre-allocate output buffers to avoid per-frame allocation
- Gate tracker assigns consistent IDs across frames using Hungarian matching on IoU
- Log raw detections + inference time to `logging/` for offline analysis
- Test fixtures in `tests/fixtures/perception/` contain annotated gate images with known corners

## Latency Budget

| Stage | Target | Hard Limit |
|-------|--------|------------|
| Image preprocessing | <1ms | 2ms |
| Neural inference | <15ms | 20ms |
| Corner refinement | <1ms | 2ms |
| **Total perception** | **<17ms** | **24ms** |
