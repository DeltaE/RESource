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
import logging
import os
import platform
import re
import sys
import threading
import time
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import yaml

# Configure repository-backed process caches before importing geospatial modules.
_CLI_TEMP_DIRECTORY = Path("data/tmp/resource-cli").resolve()
_MATPLOTLIB_CACHE_DIRECTORY = Path("data/tmp/matplotlib").resolve()
_CLI_TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
_MATPLOTLIB_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_CLI_TEMP_DIRECTORY)
os.environ["MPLCONFIGDIR"] = str(_MATPLOTLIB_CACHE_DIRECTORY)

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
from RESource import utility as utils

# ── Coloured output ───────────────────────────────────────────────────────────


def _c(color, msg):
    return f"{color}{Style.BRIGHT}{msg}{Style.RESET_ALL}"


def print_error(msg):
    utils.print_error(msg)


def print_success(msg):
    utils.print_update(message=msg)


def print_warning(msg):
    utils.print_update(message=msg, alert=True)


def print_info(msg):
    utils.print_update(message=msg)


def print_hint(msg):
    utils.print_update(message=msg)


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

RUN_LOG_DIR = Path("results/logs/runs")
LATEST_DETAIL_LOG = Path("results/logs/resource.log")
LATEST_RUNTIME_LOG = Path("results/logs/runtime_log.txt")
W = 80  # line width


def _slugify(value: str) -> str:
    """Convert text into a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip()).strip("-").lower()
    return slug or "na"


def _build_run_log_paths(
    *,
    start_dt: datetime,
    config_path: str,
    weather_year: int | str,
    regions: list[str] | None,
) -> tuple[Path, Path]:
    """Generate unique detail/runtime log paths for a single CLI invocation."""
    timestamp = start_dt.strftime("%Y%m%d_%H%M%S_%f")
    pid_tag = os.getpid()
    config_tag = _slugify(Path(config_path).stem)
    year_tag = _slugify(weather_year)
    if regions is None:
        region_tag = "all"
    elif len(regions) == 0:
        region_tag = "none"
    else:
        region_tag = "-".join(_slugify(region) for region in regions)
    # Keep filenames manageable when many regions are requested.
    region_tag = region_tag[:80]
    run_base = f"{timestamp}_p{pid_tag}_{config_tag}_y{year_tag}_r{region_tag}"
    detail_log = RUN_LOG_DIR / f"{run_base}_detail.log"
    runtime_log = RUN_LOG_DIR / f"{run_base}_runtime.log"
    return detail_log, runtime_log


def _write_latest_log_pointer(pointer_file: Path, target_file: Path) -> None:
    """Write/update a small pointer file to the latest generated run log."""
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pointer_file, "w", encoding="utf-8") as fh:
        fh.write(f"Latest run log: {target_file.resolve()}\n")


class LiveStatus:
    """Small in-place CLI dashboard for long-running regional jobs."""

    STAGES_PER_PIPELINE = 7

    def __init__(self) -> None:
        self.total = 0
        self.done = 0
        self.failed = 0
        self.running = "initializing"
        self.current_stage = 0
        self.recent = deque(maxlen=2)
        self._lines_rendered = 0
        self._stream = sys.stderr
        self._enabled = self._stream.isatty()
        self._last_render = 0.0
        self._refresh_interval = 0.2
        self._job_started_at = None
        self._spinner_index = 0
        self._render_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_thread = None

    def configure(self, total: int) -> None:
        self.total = total
        if self._enabled and self._heartbeat_thread is None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat,
                name="resource-cli-status",
                daemon=True,
            )
            self._heartbeat_thread.start()
        self.render(force=True)

    def update(self, message: str) -> None:
        clean = " ".join(str(message).split())
        stage_match = re.match(r"Step\s+(\d+)", clean)
        if stage_match:
            self.current_stage = min(
                self.STAGES_PER_PIPELINE,
                max(self.current_stage, int(stage_match.group(1))),
            )
        if clean and (not self.recent or clean != self.recent[-1]):
            self.recent.append(clean)
        self.render()

    def start(self, region: str, resource: str) -> None:
        self.running = f"{region} / {resource}"
        self.current_stage = 0
        self._job_started_at = time.monotonic()
        self.update(f"Started {self.running}")
        self.render(force=True)

    def complete(self, region: str, resource: str, *, failed: bool = False) -> None:
        self.done += 1
        self.failed += int(failed)
        self.current_stage = 0
        self._job_started_at = None
        result = "failed" if failed else "completed"
        self.running = "waiting for next job"
        self.update(f"{region} / {resource} {result}")
        self.render(force=True)

    def render(self, *, force: bool = False) -> None:
        if not self._enabled:
            return
        with self._render_lock:
            now = time.monotonic()
            if not force and now - self._last_render < self._refresh_interval:
                return
            width = 30
            total_stages = self.total * self.STAGES_PER_PIPELINE
            finished_stages = min(
                total_stages,
                self.done * self.STAGES_PER_PIPELINE
                + (0 if self.done >= self.total else max(0, self.current_stage - 1)),
            )
            fraction = finished_stages / total_stages if total_stages else 0
            filled = min(width, int(width * fraction))
            bar = "█" * filled + "░" * (width - filled)
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._spinner_index % 10]
            self._spinner_index += 1
            elapsed = ""
            if self._job_started_at is not None:
                elapsed = f" | elapsed {_hms(now - self._job_started_at)}"
            lines = [
                (
                    f"Overall  [{bar}] {finished_stages}/{total_stages or '?'} stages "
                    f"| {self.done}/{self.total or '?'} pipelines"
                ),
                (
                    f"Running: {spinner} {self.running}"
                    + (
                        f" — step {self.current_stage}/{self.STAGES_PER_PIPELINE}"
                        if self.current_stage
                        else ""
                    )
                    + elapsed
                ),
                *(f"Status:  {status[:100]}" for status in self.recent),
            ]
            if self._lines_rendered:
                self._stream.write(f"\x1b[{self._lines_rendered}F")
            for line in lines:
                self._stream.write(f"\x1b[2K{line}\n")
            self._stream.flush()
            self._lines_rendered = len(lines)
            self._last_render = now

    def _heartbeat(self) -> None:
        """Refresh elapsed time without producing additional terminal lines."""
        while not self._stop_event.wait(0.5):
            self.render(force=True)

    def finish(self, status: str, log_path: Path) -> None:
        self.running = f"finished: {status}"
        self.update(f"Detailed log: {log_path}")
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)
        self.render(force=True)
        if self._enabled:
            self._stream.write("\n")
        else:
            print(f"RESource {status}: {self.done}/{self.total} jobs finished")
            print(f"Detailed log: {log_path}")


class LogStream:
    """Convert otherwise-unstructured stdout/stderr writes into log records."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, "captured terminal output | %s", line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, "captured terminal output | %s", self._buffer.rstrip())
        self._buffer = ""


def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _fmt(label: str, value, width: int = 22) -> str:
    return f"  {label:<{width}}: {value}\n"


def _na(value, fmt=str) -> str:
    return fmt(value) if value is not None else "n/a (install psutil)"


def write_runtime_log(
    log_file: Path,
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
    log_file.parent.mkdir(parents=True, exist_ok=True)

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

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(block)
    _write_latest_log_pointer(LATEST_RUNTIME_LOG, log_file)


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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed module-and-line logs in the terminal",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the fully resolved configuration and exit",
    )
    parser.add_argument(
        "--show-overrides",
        action="store_true",
        help="Print inherited source files and overridden paths, then exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve and validate configuration without running workflows",
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

    args = build_parser().parse_args()
    hw_start = hw_snapshot()
    live_status = LiveStatus()
    inspection_mode = args.show_config or args.show_overrides or args.validate_only
    detail_log_path, runtime_log_path = _build_run_log_paths(
        start_dt=start_dt,
        config_path=args.config,
        weather_year=args.year,
        regions=args.regions,
    )
    detail_log = utils.configure_runtime_logging(
        detail_log_path,
        verbose=args.verbose,
        status_sink=None if args.verbose or inspection_mode else live_status.update,
    )
    _write_latest_log_pointer(LATEST_DETAIL_LOG, detail_log)
    _write_latest_log_pointer(LATEST_RUNTIME_LOG, runtime_log_path)
    logger = logging.getLogger("RESource")
    logger.info(
        "Pipeline invocation: config=%s year=%s regions=%s", args.config, args.year, args.regions
    )

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        config, config_provenance = utils.resolve_config(args.config)
        utils.print_update(message=f"Config loaded: {args.config}")
    except FileNotFoundError:
        print_error(f"✗ Config not found: {args.config}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"✗ Error loading config: {exc}")
        sys.exit(1)

    if "region_mapping" not in config:
        print_error("✗ 'region_mapping' missing from config.")
        sys.exit(1)

    if args.show_config:
        print(yaml.safe_dump(config, sort_keys=False), end="")
    if args.show_overrides:
        print(yaml.safe_dump(config_provenance, sort_keys=False), end="")
    if args.validate_only:
        print(f"Configuration valid: {Path(args.config).resolve()}")
    if args.show_config or args.show_overrides or args.validate_only:
        return 0

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

    # ── Pipeline loop ─────────────────────────────────────────────────────────
    resource_types = ["wind", "solar"]
    region_log = []
    live_status.configure(len(regions) * len(resource_types))

    for region in regions:
        for resource_type in resource_types:
            live_status.start(region, resource_type)
            logger.info("Starting job region=%s resource=%s", region, resource_type)
            t0 = datetime.now()
            builder = None
            try:
                capture_context = redirect_stdout(LogStream(logger, logging.INFO))
                error_context = redirect_stderr(LogStream(logger, logging.WARNING))
                with capture_context, error_context:
                    builder = RES.RESources_builder(
                        config_file_path=args.config,
                        region_short_code=region,
                        resource_type=resource_type,
                        weather_year=weather_year,
                    )
                    builder.build(
                        select_top_sites=True,
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
                logger.info(
                    "Completed job region=%s resource=%s elapsed=%s",
                    region,
                    resource_type,
                    _hms(elapsed),
                )
                live_status.complete(region, resource_type)

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
                logger.exception("Job failed region=%s resource=%s", region, resource_type)
                live_status.complete(region, resource_type, failed=True)
            finally:
                builder = None
                cleanup = utils.release_process_memory()
                logger.info(
                    "Post-job memory cleanup region=%s resource=%s collected=%s",
                    region,
                    resource_type,
                    cleanup["unreachable_objects_collected"],
                )

    # ── Write log ─────────────────────────────────────────────────────────────
    end_dt = datetime.now()
    hw_end = hw_snapshot()
    any_fail = any(e["status"] != "ok" for e in region_log)
    status = "PARTIAL" if any_fail else "SUCCESS"

    live_status.finish(status, detail_log)
    live_status.update(f"Runtime summary: {runtime_log_path}")

    write_runtime_log(
        runtime_log_path,
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
    _, fallback_runtime_log = _build_run_log_paths(
        start_dt=_start_dt,
        config_path="unknown",
        weather_year="unknown",
        regions=[],
    )
    try:
        exit_code = main(_start_dt)
        if exit_code:
            sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n  Interrupted (Ctrl+C)")
        write_runtime_log(
            fallback_runtime_log,
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
            fallback_runtime_log,
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
