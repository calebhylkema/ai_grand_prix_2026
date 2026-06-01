#!/usr/bin/env python3
"""Benchmark trained RL policy inference latency.

Loads a PPO model from disk and measures how long a single forward pass takes.
This is the RL controller's analogue of the MPCC solve time — the key number
for deciding whether RL is viable as the primary controller.

Usage::

    uv run python scripts/bench_rl_inference.py --model runs/rl_smoke/final_model
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="RL policy inference benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/rl_smoke/final_model",
        help="Path to PPO model (without .zip extension)",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model_path = Path(args.model)
    print(f"Loading model from {model_path}...")
    model = PPO.load(str(model_path))

    # Build a dummy observation (14-D: pos, vel, euler, relative_gate, progress, speed)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, 14).astype(np.float32)

    # Warmup
    for _ in range(args.warmup):
        _action, _state = model.predict(obs, deterministic=True)

    # Timed
    det_times = []
    stoch_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        det_times.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(args.iterations):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=False)
        stoch_times.append((time.perf_counter() - t0) * 1000.0)

    def _stats(label: str, times: list[float]) -> None:
        mn = min(times)
        p50 = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        mx = max(times)
        print(
            f"  {label:25s}  min={mn:6.3f}  p50={p50:6.3f}  p95={p95:6.3f}  "
            f"p99={p99:6.3f}  max={mx:6.3f} ms"
        )

    print("\n" + "=" * 90)
    print(f"RL Policy Inference Benchmark ({args.iterations} iterations)")
    print("=" * 90)
    _stats("Deterministic predict", det_times)
    _stats("Stochastic predict", stoch_times)
    print("=" * 90)

    # Compare to MPCC
    print("\nContext: MPCC p95 is ~13-22ms (from scripts/tune_mpcc.py)")
    p95_det = sorted(det_times)[int(len(det_times) * 0.95)]
    speedup = 22.3 / p95_det if p95_det > 0 else float("inf")
    print(f"RL speedup vs MPCC baseline: ~{speedup:.0f}x")
    print(f"Action sample: {action}")


if __name__ == "__main__":
    main()
