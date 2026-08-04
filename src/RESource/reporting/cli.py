"""CLI entrypoint for building a country solar report from existing results.

Usage:
    resource-report COUNTRY [--regions R1 R2 ...] [--scenarios NAME ...] [--out DIR]

Examples:
    resource-report CAN --regions BC
    resource-report CAN --scenarios baseline no_buffers --out reports
"""

from __future__ import annotations

import argparse
import sys

from RESource import utility as utils
from RESource.reporting.builder import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a country solar report from existing RESource pipeline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("country", help="Country config code, e.g. CAN")
    parser.add_argument(
        "--regions",
        "-r",
        nargs="*",
        metavar="CODE",
        help="Region codes to include (default: all regions declared in the scenario configs)",
    )
    parser.add_argument(
        "--scenarios",
        "-s",
        nargs="*",
        metavar="NAME",
        help="Scenario file stems to include, e.g. baseline no_buffers (default: all)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="reports",
        metavar="DIR",
        help="Output directory root (default: reports)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        out_path = build_report(
            args.country,
            regions=args.regions,
            scenarios=args.scenarios,
            out_dir=args.out,
        )
    except Exception as exc:
        utils.print_error(f"Failed to build report: {exc}")
        return 1
    utils.print_update(message=f"Report ready: {out_path}")
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
