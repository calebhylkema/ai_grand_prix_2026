"""Perception module: gate detection, PnP pose estimation, and gate tracking."""

from drone_racing.perception.classical import ClassicalDetector, ClassicalDetectorConfig
from drone_racing.perception.neural import NeuralDetector, NeuralDetectorConfig
from drone_racing.perception.pnp import PnPSolver
from drone_racing.perception.tracker import GateTracker, GateTrackerConfig

__all__ = [
    "ClassicalDetector",
    "ClassicalDetectorConfig",
    "GateTracker",
    "GateTrackerConfig",
    "NeuralDetector",
    "NeuralDetectorConfig",
    "PnPSolver",
]
