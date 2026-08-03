#!/usr/bin/env python3
"""
RESource — Renewable Energy Resource Analysis Pipeline

Usage:
    resource CONFIG --year YYYY [--regions R1 R2 ...]

Arguments:
    CONFIG           Path to YAML configuration file (required)
    --year, -y       Weather year to process (overrides 'weather_year' key in config)
    --regions, -r    Region codes to process (default: all regions in config)

Examples:
    resource config/CAN_baseline.yaml --year 2024
    resource config/CAN_baseline.yaml --year 2024 -r BC AB
    resource config/WB6_baseline.yaml --year 2023 -r AL BA XK
"""

import argparse
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:

    class _NoColor:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BRIGHT = RESET_ALL = ""

    Fore = Style = _NoColor()

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False

import RESource.RESources as RES
from RESource.utility import load_config

# ── Coloured output ───────────────────────────────────────────────────────────


def _c(color, msg):
    return f"{color}{Style.BRIGHT}{msg}{Style.RESET_ALL}"


def print_error(msg):
    print(_c(Fore.RED, msg))


def print_success(msg):
    print(_c(Fore.GREEN, msg))


def print_warning(msg):
    print(_c(Fore.YELLOW, msg))


def print_info(msg):
    print(_c(Fore.CYAN, msg))


def print_hint(msg):
    print(_c(Fore.MAGENTA, msg))


# ── Hardware snapshot ─────────────────────────────────────────────────────────


def hw_snapshot() -> dict:
    """
    Capture system and process state at a point in time.
    Returns an empty dict for fields unavailable without psutil.
    """
    snap = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu_logical": os.cpu_count(),  # stdlib fallback
        "cpu_physical": None,
        "ram_total_gb": None,
        "ram_avail_gb": None,
        "proc_rss_gb": None,
    }
    if _PSUTIL:
        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        snap.update(
            {
                "cpu_logical": psutil.cpu_count(logical=True),
                "cpu_physical": psutil.cpu_count(logical=False),
                "ram_total_gb": round(vm.total / 1e9, 1),
                "ram_avail_gb": round(vm.available / 1e9, 1),
                "proc_rss_gb": round(rss / 1e9, 2),
            }
        )
    return snap


# ── Runtime log ───────────────────────────────────────────────────────────────

LOG_FILE = Path("results/logs/runtime_log.txt")
W = 80  # line width


def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _fmt(label: str, value, width: int = 22) -> str:
    return f"  {label:<{width}}: {value}\n"


def _na(value, fmt=str) -> str:
    return fmt(value) if value is not None else "n/a (install psutil)"


def write_runtime_log(
    *,
    config_path: str,
    regions: list,
    weather_year,
    resource_types: list,
    status: str,
    start_dt: datetime,
    end_dt: datetime,
    hw_start: dict,
    hw_end: dict,
    region_log: list,  # list of dicts: region, resource, status, elapsed_s, error
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    runtime_s = (end_dt - start_dt).total_seconds()

    # Peak RSS: the larger of start/end (proxy; exact peak needs a sampling thread)
    rss_start = hw_start.get("proc_rss_gb")
    rss_end = hw_end.get("proc_rss_gb")
    peak_rss = max(filter(None, [rss_start, rss_end]), default=None)

    sep = "═" * W + "\n"
    thin = "─" * W + "\n"

    block = "\n"
    block += sep
    block += "  RESource Pipeline Run\n"
    block += f"  {start_dt:%Y-%m-%d %H:%M:%S}  →  {end_dt:%Y-%m-%d %H:%M:%S}  "
    block += f"({_hms(runtime_s)})\n"
    block += sep

    # ── Run summary ───────────────────────────────────────────────────────────
    block += _fmt("Status", status)
    block += _fmt("Config", config_path)
    block += _fmt("Weather year", weather_year)
    block += _fmt("Regions", ", ".join(str(r) for r in regions) or "—")
    block += _fmt("Resources", ", ".join(resource_types))
    block += thin

    # ── Per-region results ────────────────────────────────────────────────────
    if region_log:
        block += f"  {'Region':<6}  {'Resource':<8}  {'Status':<8}  {'Time':>10}  Error\n"
        block += f"  {'──────':<6}  {'────────':<8}  {'──────':<8}  {'──────':>10}  ─────\n"
        for entry in region_log:
            err = (entry.get("error") or "")[:50]
            block += (
                f"  {entry['region']:<6}  {entry['resource']:<8}  "
                f"{entry['status']:<8}  {_hms(entry['elapsed_s']):>10}  {err}\n"
            )
        block += thin

    # ── Hardware ──────────────────────────────────────────────────────────────
    cpu_info = (
        f"{_na(hw_start.get('cpu_logical'))} logical  /  "
        f"{_na(hw_start.get('cpu_physical'))} physical"
    )
    ram_avail_start = _na(hw_start.get("ram_avail_gb"), lambda v: f"{v:.1f} GB")
    ram_avail_end = _na(hw_end.get("ram_avail_gb"), lambda v: f"{v:.1f} GB")

    block += _fmt("CPU", cpu_info)
    block += _fmt("RAM total", _na(hw_start.get("ram_total_gb"), lambda v: f"{v:.1f} GB"))
    block += _fmt("RAM avail", f"{ram_avail_start}  →  {ram_avail_end}")
    block += _fmt(
        "Process RSS",
        f"{_na(rss_start, lambda v: f'{v:.2f} GB')}  →  "
        f"{_na(rss_end, lambda v: f'{v:.2f} GB')}  "
        f"(peak ≈ {_na(peak_rss, lambda v: f'{v:.2f} GB')})",
    )
    block += thin

    # ── Environment ───────────────────────────────────────────────────────────
    block += _fmt("Python", hw_start.get("python", "n/a"))
    block += _fmt("Platform", hw_start.get("platform", "n/a"))
    block += _fmt("psutil", "available" if _PSUTIL else "not installed — hw metrics unavailable")
    block += sep

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(block)


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Create the RESource command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RESource — Renewable Energy Resource Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=2024,
        metavar="YYYY",
        help="Weather year (overrides 'weather_year' in config)",
    )
    parser.add_argument(
        "--regions",
        "-r",
        nargs="*",
        metavar="CODE",
        help="Region codes to process (default: all in config)",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────


def main(start_dt: datetime | None = None) -> int:
    """Run configured wind and solar assessments.

    Args:
        start_dt: Optional start timestamp used for runtime reporting.

    Raises:
        SystemExit: If arguments or configuration are invalid.
    """

    if start_dt is None:
        start_dt = datetime.now()

    hw_start = hw_snapshot()
    args = build_parser().parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
        print_success(f"✓ Config loaded: {args.config}")
    except FileNotFoundError:
        print_error(f"✗ Config not found: {args.config}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"✗ Error loading config: {exc}")
        sys.exit(1)

    if "region_mapping" not in config:
        print_error("✗ 'region_mapping' missing from config.")
        sys.exit(1)

    available_regions = list(config["region_mapping"].keys())

    # ── Resolve weather year ──────────────────────────────────────────────────
    weather_year = args.year or config.get("weather_year")
    if weather_year is None:
        print_error("✗ No weather year specified.")
        print_hint("  Pass --year YYYY or add 'weather_year: YYYY' to your config.")
        sys.exit(1)
    weather_year = int(weather_year)

    # ── Resolve regions ───────────────────────────────────────────────────────
    if args.regions is None:
        regions = available_regions
    else:
        invalid = [r for r in args.regions if r not in available_regions]
        if invalid:
            print_error(f"✗ Unknown region(s): {invalid}")
            print_warning(f"  Available: {available_regions}")
            sys.exit(1)
        regions = args.regions

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print_info(f"  RESource  |  year={weather_year}  |  regions={regions}")
    print_info(f"  config={args.config}")
    if _PSUTIL:
        vm = psutil.virtual_memory()
        print_info(
            f"  CPU={hw_start['cpu_logical']} logical / {hw_start['cpu_physical']} physical  "
            f"|  RAM={hw_start['ram_total_gb']:.0f} GB total  "
            f"|  avail={vm.available / 1e9:.1f} GB"
        )
    print(f"{'=' * 65}\n")

    # ── Pipeline loop ─────────────────────────────────────────────────────────
    resource_types = ["wind", "solar"]
    region_log = []

    for region in regions:
        for resource_type in resource_types:
            print_info(f"→ {region} / {resource_type}")
            t0 = datetime.now()
            try:
                builder = RES.RESources_builder(
                    config_file_path=args.config,
                    region_short_code=region,
                    resource_type=resource_type,
                    weather_year=weather_year,
                )
                builder.build(
                    select_top_sites=True,
                    use_pypsa_buses=True,
                    use_grid_lines=True,
                    make_clusters=True,
                    clean_store=False,
                )
                elapsed = (datetime.now() - t0).total_seconds()
                region_log.append(
                    {
                        "region": region,
                        "resource": resource_type,
                        "status": "ok",
                        "elapsed_s": elapsed,
                        "error": None,
                    }
                )
                print_success(f"  ✓ {region} / {resource_type}  ({_hms(elapsed)})")

            except Exception as exc:
                elapsed = (datetime.now() - t0).total_seconds()
                region_log.append(
                    {
                        "region": region,
                        "resource": resource_type,
                        "status": "failed",
                        "elapsed_s": elapsed,
                        "error": str(exc),
                    }
                )
                print_error(f"  ✗ {region} / {resource_type}: {exc}")
                print_warning("    Continuing...")

    # ── Write log ─────────────────────────────────────────────────────────────
    end_dt = datetime.now()
    hw_end = hw_snapshot()
    any_fail = any(e["status"] != "ok" for e in region_log)
    status = "PARTIAL" if any_fail else "SUCCESS"

    print(f"\n{'=' * 65}")
    print_success(f"  Done — {status}  ({_hms((end_dt - start_dt).total_seconds())})")
    print_info(f"  Log → {LOG_FILE}")
    print(f"{'=' * 65}\n")

    write_runtime_log(
        config_path=args.config,
        regions=regions,
        weather_year=weather_year,
        resource_types=resource_types,
        status=status,
        start_dt=start_dt,
        end_dt=end_dt,
        hw_start=hw_start,
        hw_end=hw_end,
        region_log=region_log,
    )
    return 1 if any_fail else 0


# ── Entry point ───────────────────────────────────────────────────────────────


def entrypoint() -> None:
    """Run the CLI with logging and consistent process exit codes."""
    _start_dt = datetime.now()
    try:
        exit_code = main(_start_dt)
        if exit_code:
            sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n  Interrupted (Ctrl+C)")
        write_runtime_log(
            config_path="unknown",
            regions=[],
            weather_year="unknown",
            resource_types=[],
            status="INTERRUPTED",
            start_dt=_start_dt,
            end_dt=datetime.now(),
            hw_start=hw_snapshot(),
            hw_end={},
            region_log=[],
        )
        sys.exit(130)
    except Exception as exc:
        print_error(f"  Unexpected error: {exc}")
        write_runtime_log(
            config_path="unknown",
            regions=[],
            weather_year="unknown",
            resource_types=[],
            status=f"FAILED: {exc}",
            start_dt=_start_dt,
            end_dt=datetime.now(),
            hw_start=hw_snapshot(),
            hw_end={},
            region_log=[],
        )
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
