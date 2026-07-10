"""``aigp-train-detector`` — gate-detector data generation + training dispatcher.

Subcommands::

    aigp-train-detector generate   # render synthetic YOLO keypoint training data
    aigp-train-detector train      # train (or --export-only) the YOLOv8 keypoint model

Each subcommand forwards its remaining arguments to the underlying training
module; run ``aigp-train-detector <subcommand> --help`` for its options.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: dispatch to data generation or training."""
    args = list(argv) if argv is not None else sys.argv[1:]

    # Dispatch on the first positional so `<subcommand> --help` reaches the
    # underlying module's parser instead of being consumed here.
    if args and args[0] == "generate":
        from drone_racing.training.data_generation import main as run

        run(args[1:])
        return
    if args and args[0] == "train":
        from drone_racing.training.gate_detector import main as run

        run(args[1:])
        return

    parser = argparse.ArgumentParser(
        prog="aigp-train-detector",
        description="Gate keypoint detector: synthetic data generation and YOLOv8 training.",
    )
    parser.add_argument(
        "stage",
        choices=["generate", "train"],
        help="Pipeline stage: generate synthetic data, or train/export the model",
    )
    parser.parse_args(args)


if __name__ == "__main__":
    main()
