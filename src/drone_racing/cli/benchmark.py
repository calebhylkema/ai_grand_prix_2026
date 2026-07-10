"""``aigp-benchmark`` — latency benchmark dispatcher.

Subcommands::

    aigp-benchmark pipeline   # perception-to-control pipeline latency (<40ms target)
    aigp-benchmark rl         # trained RL policy inference latency
    aigp-benchmark mpcc       # MPCC solver parameter sweep

Each subcommand forwards its remaining arguments to the underlying benchmark
module; run ``aigp-benchmark <subcommand> --help`` for its options.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: dispatch to the selected benchmark."""
    args = list(argv) if argv is not None else sys.argv[1:]

    # Dispatch on the first positional so `<subcommand> --help` reaches the
    # underlying module's parser instead of being consumed here.
    if args and args[0] == "pipeline":
        from drone_racing.cli.bench_pipeline import main as run

        run(args[1:])
        return
    if args and args[0] == "rl":
        from drone_racing.cli.bench_rl_inference import main as run

        run(args[1:])
        return
    if args and args[0] == "mpcc":
        from drone_racing.cli.tune_mpcc import main as run

        run(args[1:])
        return

    parser = argparse.ArgumentParser(
        prog="aigp-benchmark",
        description="Latency benchmarks for the racing stack.",
    )
    parser.add_argument(
        "benchmark",
        choices=["pipeline", "rl", "mpcc"],
        help="Which benchmark to run (pipeline latency, RL inference, MPCC sweep)",
    )
    parser.parse_args(args)


if __name__ == "__main__":
    main()
