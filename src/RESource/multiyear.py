#!/usr/bin/env python3
"""Run the RESource pipeline across a range of weather years.

Each year is executed as an independent subprocess so that a failure in one
year does not abort subsequent years.  Results and logs accumulate normally
in the standard output directories.

Usage
-----
    resource-multiyear CONFIG --start YYYY --end YYYY [--regions R1 R2 ...]

Examples
--------
    resource-multiyear config/CAN_baseline.yaml --start 2014 --end 2024 -r BC
    resource-multiyear config/CAN_baseline.yaml --start 2014 --end 2024 -r BC AB ON
    resource-multiyear config/WB6_baseline.yaml --start 2019 --end 2023 -r AL MK RS
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Colours (graceful fallback if colorama absent) ────────────────────────────
try:
    from colorama import Fore, Style, init

    init(autoreset=True)

    def _c(col, msg):
        return f"{col}{Style.BRIGHT}{msg}{Style.RESET_ALL}"
except ImportError:

    def _c(col, msg):
        return msg  # no-op

    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = ""


def ok(msg: str) -> None:
    """Print a success message."""
    print(_c(Fore.GREEN, msg))


def err(msg: str) -> None:
    """Print an error message."""
    print(_c(Fore.RED, msg))


def warn(msg: str) -> None:
    """Print a warning message."""
    print(_c(Fore.YELLOW, msg))


def info(msg: str) -> None:
    """Print an informational message."""
    print(_c(Fore.CYAN, msg))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def run_year(config: str, year: int, regions: list[str]) -> bool:
    """Invoke the installed RESource CLI for a single year.

    Args:
        config: Path to a RESource YAML configuration.
        year: Weather year to process.
        regions: Optional region codes to process.

    Returns:
        True when the subprocess exits successfully; otherwise, False.
    """
    cmd = [sys.executable, "-m", "RESource.cli", config, "--year", str(year)]
    if regions:
        cmd += ["--regions"] + regions

    info(f"\n{'─' * 65}")
    info(f"  Year {year}  |  cmd: {' '.join(cmd)}")
    info(f"{'─' * 65}")

    t0 = datetime.now()
    result = subprocess.run(cmd)  # inherits stdout/stderr → live output
    elapsed = (datetime.now() - t0).total_seconds()

    if result.returncode == 0:
        ok(f"  ✓ Year {year} completed  ({_hms(elapsed)})")
        return True
    else:
        err(f"  ✗ Year {year} FAILED (exit {result.returncode})  ({_hms(elapsed)})")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-year command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run RESource across a range of weather years.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        required=True,
        metavar="YYYY",
        help="First year of the range (inclusive)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        required=True,
        metavar="YYYY",
        help="Last year of the range (inclusive)",
    )
    parser.add_argument(
        "--regions",
        "-r",
        nargs="*",
        metavar="CODE",
        help="Region codes (default: all regions in config)",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────


def entrypoint() -> None:
    """Run all requested weather years and return a process-style exit code."""
    args = build_parser().parse_args()

    if args.start > args.end:
        err(f"--start ({args.start}) must be ≤ --end ({args.end})")
        sys.exit(1)

    if not Path(args.config).exists():
        err(f"Config not found: {args.config}")
        sys.exit(1)

    years = list(range(args.start, args.end + 1))
    regions = [r.upper() for r in args.regions] if args.regions else []

    print(f"\n{'═' * 65}")
    info("  RESource multi-year run")
    info(f"  Config  : {args.config}")
    info(f"  Years   : {args.start} – {args.end}  ({len(years)} years)")
    info(f"  Regions : {', '.join(regions) if regions else 'all in config'}")
    print(f"{'═' * 65}\n")

    wall_start = datetime.now()
    results: dict[int, bool] = {}

    for year in years:
        results[year] = run_year(args.config, year, regions)

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_elapsed = (datetime.now() - wall_start).total_seconds()
    passed = [y for y, s in results.items() if s]
    failed = [y for y, s in results.items() if not s]

    print(f"\n{'═' * 65}")
    info(f"  Multi-year summary  ({_hms(wall_elapsed)} total)")
    print(f"{'─' * 65}")
    ok(f"  Succeeded ({len(passed)}) : {', '.join(map(str, passed)) or '—'}")
    if failed:
        err(f"  Failed    ({len(failed)}) : {', '.join(map(str, failed))}")
        warn("  Re-run failed years individually to investigate.")
    print(f"{'═' * 65}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    entrypoint()
