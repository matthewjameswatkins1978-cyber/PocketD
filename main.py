"""Pocket Drummer — entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from midi_out import find_output_port, list_output_ports
from scheduler import run_hardcoded_groove


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pocket Drummer — listen, lock, groove.",
    )
    parser.add_argument(
        "--milestone",
        type=int,
        default=1,
        choices=[1],
        help="Development milestone to run (currently only 1)",
    )
    parser.add_argument(
        "--midi-port",
        type=str,
        default="",
        help="MIDI output port name (substring match)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available MIDI output ports and exit",
    )
    parser.add_argument(
        "--groove",
        type=str,
        default="simple_rock",
        help="Groove id from data/grooves.yaml (default: simple_rock)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Tempo in BPM (default: 120)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Number of bars to play (default: loop until Ctrl+C)",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Complexity level 1-5 (default: 5 = full groove)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pocket_drummer")

    if args.list_ports:
        ports = list_output_ports()
        if not ports:
            print("No MIDI output ports found.")
            print("Create one in loopMIDI, then retry.")
            return 1
        print("MIDI output ports:")
        for i, name in enumerate(ports):
            print(f"  [{i}] {name}")
        return 0

    if not args.midi_port:
        log.error("Specify --midi-port or use --list-ports to see options.")
        return 1

    try:
        port = find_output_port(args.midi_port)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("Milestone %d — hard-coded groove → MIDI", args.milestone)
    log.info("Port: %s | Groove: %s | BPM: %.1f", port, args.groove, args.bpm)
    log.info("Press Ctrl+C to stop.")

    if args.milestone == 1:
        run_hardcoded_groove(
            midi_port=port,
            groove_id=args.groove,
            bpm=args.bpm,
            bars=args.bars,
            complexity_level=args.complexity,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
