"""
Persistent training manager CLI.

Usage (from the project root):

    python -m scripts.manage_training init
    python -m scripts.manage_training start --all
    python -m scripts.manage_training start --backbone resnet50
    python -m scripts.manage_training stop
    python -m scripts.manage_training status
    python -m scripts.manage_training watch --interval 10
    python -m scripts.manage_training resume          # alias for start --all
    python -m scripts.manage_training clear           # clear stop flags
"""

from __future__ import annotations

import argparse
import logging
import sys

from training.training_manager import (
    BASELINE_BACKBONES,
    STATE_FILE,
    clear_stop_flags,
    init_state,
    print_status,
    run_all,
    stop_all,
    watch,
)
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage BrainTumorDiagnosis training runs.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "init",
        help="Create the state file and detect existing checkpoints.",
    )

    start_parser = subparsers.add_parser(
        "start",
        help="Start (or resume) training.",
    )

    start_group = start_parser.add_mutually_exclusive_group()

    start_group.add_argument(
        "--all",
        action="store_true",
        help="Train every incomplete backbone sequentially.",
    )

    start_group.add_argument(
        "--backbone",
        type=str,
        choices=BASELINE_BACKBONES,
        default=None,
        help="Train a single backbone.",
    )

    subparsers.add_parser(
        "resume",
        help="Alias for 'start --all': resume any interrupted training.",
    )

    subparsers.add_parser(
        "stop",
        help="Gracefully stop running trainings at the next epoch boundary.",
    )

    subparsers.add_parser(
        "status",
        help="Print the project and per-backbone progress.",
    )

    subparsers.add_parser(
        "clear",
        help="Clear stop flags so paused trainings can resume.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Print progress periodically while training runs.",
    )

    watch_parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Report interval in minutes (default: 10).",
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    setup_logging(
        level=logging.INFO,
        log_file=None,
    )

    if args.command == "init":
        state = init_state()

        print_status(state)

        return 0

    if args.command == "start":
        if args.backbone is not None:
            from training.training_manager import run_backbone, load_state

            state = load_state()

            run_backbone(args.backbone, state, STATE_FILE)

            print_status(state)
        else:
            run_all()

        return 0

    if args.command == "resume":
        run_all()

        return 0

    if args.command == "stop":
        stop_all()

        return 0

    if args.command == "status":
        from training.training_manager import load_state

        print_status(load_state())

        return 0

    if args.command == "clear":
        clear_stop_flags()

        return 0

    if args.command == "watch":
        watch(interval_minutes=args.interval)

        return 0

    parser.print_help()

    return 1


if __name__ == "__main__":
    sys.exit(main())