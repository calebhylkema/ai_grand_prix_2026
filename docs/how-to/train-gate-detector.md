# Train the gate detector

The active VQ2 detector is classical (glow-resistant ring test). A YOLOv8-pose
keypoint detector is the planned upgrade if dynamic lighting / clutter defeats
classical CV (the spec warns it will; see the roadmap).

## Generate training data

```bash
uv run aigp-train-detector generate --output .local/datasets/gate_keypoints --num-train 5000
```

Synthetic gate renders with domain randomization. For VQ2-real training data, use
captured frames from the trace archives instead (they are the actual distribution).

## Train

```bash
uv run aigp-train-detector --data .local/datasets/gate_keypoints/dataset.yaml --epochs 100
```

Trains YOLOv8-nano keypoint (4–8 gate corners directly). Outputs to
`.local/runs/pose/`.

## Export for inference

```bash
uv run aigp-train-detector --export-only --weights .local/runs/pose/train/weights/best.pt
```

Exports ONNX — production inference uses ONNX Runtime (2–3× faster than raw
PyTorch on Jetson via TensorRT). Latency budget: <17 ms per frame.
