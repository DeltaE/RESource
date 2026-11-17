"""CLI entrypoint for building the standalone RESource dashboard.

Usage:
    resource-dash [--out RESource_dash.html]

Writes a single, self-contained HTML file with no baked-in data: open it in
a browser, pick a ``.h5`` store file (e.g. from ``results/...``) with the
file picker, and it's parsed and charted entirely client-side via Pyodide —
no server, no PyTables (which can't run in a browser) required.
"""

from __future__ import annotations

import argparse
import sys

from RESource import utility as utils
from RESource.dash.build import DEFAULT_OUTPUT, build_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the static, Pyodide-backed RESource dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out",
        "-o",
        default=str(DEFAULT_OUTPUT),
        metavar="FILE",
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        out_path = build_dashboard(args.out)
    except Exception as exc:
        utils.print_error(f"Failed to build dashboard: {exc}")
        return 1
    utils.print_update(message=f"Dashboard ready: {out_path}")
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
