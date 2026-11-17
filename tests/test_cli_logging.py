"""Tests for compact CLI status and structured runtime logging."""

import logging
from datetime import datetime
from pathlib import Path

from RESource import utility as utils
from RESource.cli import LiveStatus, build_parser, write_runtime_log


def _emit_test_status() -> None:
    utils.print_update(message="test pipeline status")


def test_verbose_flag_is_available() -> None:
    """The short verbose flag enables detailed terminal logging."""
    args = build_parser().parse_args(["config/test.yaml", "-v"])

    assert args.verbose is True


def test_config_inspection_flags_are_available() -> None:
    """Configuration resolution can be inspected without running a workflow."""
    args = build_parser().parse_args(
        ["config/test.yaml", "--show-config", "--show-overrides", "--validate-only"]
    )

    assert args.show_config is True
    assert args.show_overrides is True
    assert args.validate_only is True


def test_status_is_logged_with_source_location(tmp_path: Path) -> None:
    """Status records identify the originating module, function, and line."""
    messages: list[str] = []
    log_path = utils.configure_runtime_logging(
        tmp_path / "resource.log",
        status_sink=messages.append,
    )
    try:
        _emit_test_status()
        for handler in logging.getLogger("RESource").handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "test_cli_logging:" in contents
        assert "_emit_test_status" in contents
        assert "test pipeline status" in contents
        assert messages == ["test pipeline status"]
    finally:
        logger = logging.getLogger("RESource")
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        utils._OUTPUT_CONFIGURED = False
        utils._COMPACT_OUTPUT = False
        utils._STATUS_SINK = None


def test_runtime_log_keeps_full_error_message(tmp_path: Path) -> None:
    """Runtime logs should preserve the complete error text for failed jobs."""
    log_file = tmp_path / "runtime.log"
    long_error = "This is a deliberately long error message that should stay intact in the runtime log"
    write_runtime_log(
        log_file,
        config_path="config/test.yaml",
        regions=["BC"],
        weather_year=2024,
        resource_types=["wind"],
        status="PARTIAL",
        start_dt=datetime(2024, 1, 1, 12, 0, 0),
        end_dt=datetime(2024, 1, 1, 12, 0, 2),
        hw_start={},
        hw_end={},
        region_log=[
            {
                "region": "BC",
                "resource": "wind",
                "status": "failed",
                "elapsed_s": 2,
                "error": long_error,
            }
        ],
    )

    contents = log_file.read_text(encoding="utf-8")
    assert long_error in contents
