"""Versioned, read-only data assets distributed with RESource."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def legend_file(filename: str) -> Traversable:
    """Return a packaged plotting-legend resource.

    Args:
        filename: CSV basename, for example ``CLC_2018_legend.csv``.

    Raises:
        ValueError: If *filename* contains directory components.
        FileNotFoundError: If the requested legend is not distributed.
    """
    if filename != filename.rsplit("/", maxsplit=1)[-1] or "\\" in filename:
        raise ValueError("filename must be a CSV basename")

    resource = files("RESource").joinpath("assets", "legends", filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown RESource legend: {filename}")
    return resource


def mapping_file(filename: str) -> Traversable:
    """Return a packaged regional-mapping resource by basename."""
    if filename != filename.rsplit("/", maxsplit=1)[-1] or "\\" in filename:
        raise ValueError("filename must be a CSV basename")

    resource = files("RESource").joinpath("assets", "mappings", filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown RESource mapping: {filename}")
    return resource
