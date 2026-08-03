"""Tests for default transient CDS queue-limit handling."""

import inspect

import pytest

from RESource.era5_cutout import (
    DEFAULT_CDS_BASE_DELAY_SECONDS,
    DEFAULT_CDS_MAX_ATTEMPTS,
    DEFAULT_CDS_MAX_DELAY_SECONDS,
    prepare_cutout_with_retry,
)


class FakeCutout:
    """Minimal cutout that returns configured outcomes from prepare()."""

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def prepare(self, **kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome


def test_prepare_retries_temporary_cds_queue_limit():
    cutout = FakeCutout(
        [
            RuntimeError(
                "The job has been rejected: Number queued requests for this dataset "
                "is temporarily limited."
            ),
            None,
        ]
    )
    delays = []

    prepare_cutout_with_retry(
        cutout,
        max_attempts=3,
        base_delay_seconds=2,
        max_delay_seconds=10,
        sleep=delays.append,
    )

    assert cutout.calls == 2
    assert delays == [2]


def test_prepare_does_not_retry_invalid_cds_request():
    cutout = FakeCutout([RuntimeError("400 Bad Request: invalid variable")])

    with pytest.raises(RuntimeError, match="invalid variable"):
        prepare_cutout_with_retry(cutout, sleep=lambda _: None)

    assert cutout.calls == 1


def test_retry_policy_is_a_package_default():
    """Every scenario inherits the same retry policy without YAML configuration."""
    parameters = inspect.signature(prepare_cutout_with_retry).parameters

    assert parameters["max_attempts"].default == DEFAULT_CDS_MAX_ATTEMPTS == 6
    assert parameters["base_delay_seconds"].default == DEFAULT_CDS_BASE_DELAY_SECONDS == 60.0
    assert parameters["max_delay_seconds"].default == DEFAULT_CDS_MAX_DELAY_SECONDS == 900.0
