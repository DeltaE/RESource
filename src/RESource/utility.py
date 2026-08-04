"""
Utility functions and helper methods for RESource renewable energy assessment framework.

This module provides common functionality used across the RESource workflow including
configuration management, data I/O operations, coordinate transformations
utilities, and validation functions. It serves as a central repository for shared
code that supports the modular architecture of the assessment framework.

Key Functions:
    - Configuration parsing and validation
    - File I/O operations (YAML, JSON, pickle, geospatial formats)
    - Coordinate system transformations and spatial utilities
    - Data validation and error handling
    - URL downloading and caching mechanisms
    - String formatting and output styling

Author: Md Eliasinul Islam
Affiliation: Delta E+ lab, Simon Fraser University
Version: 1.0
Development Year: 2024-2025
"""

import datetime
import gc

"""
Utility functions and helper methods for RESource renewable energy assessment framework.

This module provides common functionality used across the RESource workflow including
configuration management, data I/O operations, coordinate transformations
utilities, and validation functions. It serves as a central repository for shared
code that supports the modular architecture of the assessment framework.

Key Functions:
    - Configuration parsing and validation
    - File I/O operations (YAML, JSON, pickle, geospatial formats)
    - Coordinate system transformations and spatial utilities
    - Data validation and error handling
    - URL downloading and caching mechanisms
    - String formatting and output styling

Author: Md Eliasinul Islam
Affiliation: Delta E+ lab, Simon Fraser University  
Version: 1.0
Development Year: 2024-2025
"""

import hashlib
import json
import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path

import geojson as gj
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm

_STATUS_SINK = None
_OUTPUT_CONFIGURED = False
_COMPACT_OUTPUT = False

KNOWN_CONFIG_TOP_LEVEL_KEYS = {
    "Affiliation",
    "Developer",
    "Release_Year",
    "Scenario",
    "Title",
    "admin_boundary",  # GADM
    "country",
    "custom_land_layers",
    "default_CRS",
    "demand_indicators",  # WorldPop, Gov (Population, CEEI)
    "description",
    "economic_parameters",
    "filters",  # per-resource siting exclusion buffers (vector_buffers)
    "infrastructure",  # OSM, CODERS, transmission
    "lands",  # GAEZ, CORINE, EU_DEM, and any raster-processing defaults
    "multi_country_flag",
    "region_mapping",
    "technology",  # annual_technology_baseline (NREL ATB), resource_specs (per-resource cost/sizing model)
    "version",
    "weather",  # cutout, GWA
    "weather_year",
}


def configure_runtime_logging(
    log_path: str | Path,
    *,
    verbose: bool = False,
    status_sink=None,
) -> Path:
    """Configure detailed file logging and optional compact CLI status output."""
    global _COMPACT_OUTPUT, _OUTPUT_CONFIGURED, _STATUS_SINK

    destination = Path(log_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RESource")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        terminal_handler = logging.StreamHandler()
        terminal_handler.setLevel(logging.DEBUG)
        terminal_handler.setFormatter(formatter)
        logger.addHandler(terminal_handler)

    _STATUS_SINK = status_sink
    _COMPACT_OUTPUT = status_sink is not None and not verbose
    _OUTPUT_CONFIGURED = True
    return destination


def compact_output_enabled() -> bool:
    """Return whether the CLI is rendering the compact live status display."""
    return _COMPACT_OUTPUT


def _record_status(message: str, *, level: int = logging.INFO, stacklevel: int = 2) -> None:
    """Write a structured record and forward it to the compact status display."""
    logging.getLogger("RESource").log(level, message, stacklevel=stacklevel + 1)
    if _STATUS_SINK is not None:
        _STATUS_SINK(message)


def release_process_memory() -> dict[str, int]:
    """Release unreachable Python objects before a memory-intensive job.

    Returns:
        Counts from a generation-2 garbage collection cycle. This helper never
        deletes disk caches, temporary files, downloaded data, or cutouts.
    """
    generation_counts_before = gc.get_count()
    unreachable_objects = gc.collect(2)
    generation_counts_after = gc.get_count()
    return {
        "unreachable_objects_collected": unreachable_objects,
        "generation_0_before": generation_counts_before[0],
        "generation_1_before": generation_counts_before[1],
        "generation_2_before": generation_counts_before[2],
        "generation_0_after": generation_counts_after[0],
        "generation_1_after": generation_counts_after[1],
        "generation_2_after": generation_counts_after[2],
    }


def get_gdrive_public_file(file_id: str, output_path: str) -> None:
    """
    Downloads a public-file from Google Drive using its file ID.

    Args:
        file_id (str): The unique identifier of the Google Drive file.
        output_path (str): The local path where the downloaded file will be saved.

    Note: The file must be publicly accessible (ANYONE WITH THE LINK). If the file is private, this function will not work.

    Returns:
        None
    """
    import gdown

    output_path = Path(output_path)
    if output_path.exists():
        print(
            f"Skipping downloading from Google Drive. Expected file found locally at: {output_path}"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(file_id, str) or not file_id:
        raise ValueError("Invalid file_id provided. It must be a non-empty string.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)


def repository_temp_directory(name: str) -> Path:
    """Return an ignored, repository-backed scratch directory.

    Args:
        name: Subdirectory name below ``data/tmp``.

    Returns:
        Absolute path to the created scratch directory.

    Raises:
        ValueError: If ``name`` could escape the repository scratch root.
    """
    relative_name = Path(name)
    if relative_name.is_absolute() or ".." in relative_name.parts:
        raise ValueError("repository temporary directory name must be relative")
    scratch_directory = (Path("data/tmp") / relative_name).resolve()
    scratch_directory.mkdir(parents=True, exist_ok=True)
    return scratch_directory


def fetch_file_if_missing(
    source: str,
    destination: str | Path,
    *,
    description: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream a configured remote file into place when it is not available locally.

    The response is written under repository-backed temporary storage and moved
    atomically to the requested destination after a complete, non-empty download.

    Args:
        source: Direct HTTP(S) URL from the workflow configuration.
        destination: Durable local file path defined by the configuration.
        description: Human-readable dataset label for status output.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        Path to the existing or newly downloaded file.

    Raises:
        ValueError: If the source URL or chunk size is invalid.
        IOError: If the downloaded response is empty or incomplete.
        requests.HTTPError: If the server returns an unsuccessful response.
    """
    destination_path = Path(destination)
    if destination_path.is_file() and destination_path.stat().st_size > 0:
        return destination_path
    if not isinstance(source, str) or not source.startswith(("https://", "http://")):
        raise ValueError(f"A direct HTTP(S) source is required for {destination_path}")
    if chunk_size <= 0:
        raise ValueError("download chunk_size must be positive")

    label = description or destination_path.name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_directory = repository_temp_directory("resource-downloads")
    print_update(message=f"Fetching missing dataset '{label}' from configured source...")

    with tempfile.TemporaryDirectory(prefix="run-", dir=scratch_directory) as workspace:
        temporary_path = Path(workspace) / destination_path.name
        with requests.get(source, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0
            next_report = 5
            with (
                temporary_path.open("wb") as output,
                tqdm(
                    total=total_bytes or None,
                    desc=label,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                    disable=compact_output_enabled(),
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    progress.update(len(chunk))
                    if compact_output_enabled() and total_bytes:
                        percent = int(downloaded_bytes * 100 / total_bytes)
                        if percent >= next_report:
                            print_update(message=f"Downloading '{label}': {percent}% complete")
                            next_report += 5

        if downloaded_bytes == 0:
            raise OSError(f"Configured source returned an empty file for {label}")
        if total_bytes and downloaded_bytes != total_bytes:
            raise OSError(
                f"Incomplete download for {label}: received {downloaded_bytes} "
                f"of {total_bytes} bytes"
            )
        temporary_path.replace(destination_path)

    print_update(message=f"Dataset '{label}' saved to {destination_path}")
    return destination_path


def configure_repository_temp(path: str | Path = "data/tmp/resource-cds") -> Path:
    """Route temporary processing to repository-backed storage.

    Args:
        path: Scratch directory on the repository filesystem.

    Returns:
        Absolute scratch-directory path.

    Notes:
        This changes the current process and its child processes only. The scratch
        directory is ignored by Git and is not treated as a durable workflow output.
    """
    scratch_directory = Path(path).resolve()
    scratch_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch_directory)
    tempfile.tempdir = str(scratch_directory)
    return scratch_directory


import pandas as pd
import rasterio as rio
import requests
import rioxarray as rxr
import xarray as xr
import yaml
from colorama import Fore, Style

now = datetime.datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")


def print_update(level: int = None, message: str = "--", alert: bool | None = False):
    if _OUTPUT_CONFIGURED:
        _record_status(
            message,
            level=logging.ERROR if alert else logging.INFO,
            stacklevel=2,
        )
        return
    if alert:
        level = level or 2
        color = Fore.RED
        prefix = " └ ❌ "
    elif level is not None:
        if level == 1:
            color = Fore.YELLOW
            prefix = "└"
        elif level == 2:
            color = Fore.CYAN
            prefix = " └"
        elif level == 3:
            color = Fore.LIGHTWHITE_EX + Style.DIM
            prefix = "  └"
        elif level > 3:
            color = Fore.LIGHTBLACK_EX + Style.DIM
            prefix = "  └─"
    else:
        color = Fore.LIGHTMAGENTA_EX + Style.DIM
        prefix = " ─"

    print(f"{color}{prefix}> {message}{Style.RESET_ALL}")


def print_error(message):
    if _OUTPUT_CONFIGURED:
        _record_status(message, level=logging.ERROR, stacklevel=2)
        return
    print(f"{Fore.RED} └ ❌ > {message}{Style.RESET_ALL}")


def print_module_title(text, Length_Char_inLine=60):
    if _OUTPUT_CONFIGURED:
        _record_status(text, stacklevel=2)
        return
    print(
        f"{Fore.LIGHTCYAN_EX}{Length_Char_inLine * '_'}{Style.RESET_ALL}\n"
        f"{Fore.LIGHTGREEN_EX}{5 * ' '}{text}{Style.RESET_ALL}\n"
        f"{Fore.LIGHTCYAN_EX}{Length_Char_inLine * '_'}{Style.RESET_ALL}"
    )


def print_banner(message: str):
    if _OUTPUT_CONFIGURED:
        _record_status(message, stacklevel=2)
        return
    line = "*" * len(message)
    print(f"{Fore.GREEN}{Style.BRIGHT}{line}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{line}{Style.RESET_ALL}")


def print_info(info: str):
    print(f"{Fore.LIGHTBLACK_EX}{Style.BRIGHT}ℹ️  {info}{Style.RESET_ALL}")


def print_warning(info: str):
    print(f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}⚠️  {info}{Style.RESET_ALL}")


def extract_from_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    extracted_folders = [f for f in Path(extract_dir).iterdir() if f.is_dir()]
    print_update(level=2, message=f"Extracted folders: {extracted_folders}")
    return extracted_folders


def load_geojson_file(geojson_file_path: str | Path) -> list:
    """
    Loads a GeoJSON file and extracts the coordinates from its geometry.

    Args:
        geojson_file_path (str | Path): The file path to the GeoJSON file.

    Returns:
        list: A list of coordinates extracted from the GeoJSON file's geometry.

    Raises:
        FileNotFoundError: If the specified GeoJSON file does not exist.
        JSONDecodeError: If the file is not a valid GeoJSON format.
        KeyError: If the 'geometry' or 'coordinates' keys are missing in the GeoJSON data.
    """
    with open(geojson_file_path) as f:
        coords_list = gj.load(f)["geometry"]["coordinates"]
        f.close()
        return coords_list


# Function to generate a unique index from region name and coordinates
def assign_cell_id(
    cells: gpd.GeoDataFrame, source_column: str = None, index_name: str = "cell"
) -> gpd.GeoDataFrame:
    """
    Assigns unique cell IDs to each region in the specified GeoDataFrame.

    Parameters:
    cells (gpd.GeoDataFrame): Input GeoDataFrame containing spatial data with 'x' and 'y' coordinates.
    source_column (str): Sub-national unit named column to be used for generating unique IDs (e.g. Region, Municipality). To be configured in the config file under 'GADM' key
    source_column (str): Sub-national unit named column to be used for generating unique IDs (e.g. Region, Municipality). To be configured in the config file under 'GADM' key
    index_name (str): Name for the new index column to be created.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with a new column of unique cell IDs for each region.
    """
    # Ensure the source column exists
    if source_column not in cells.columns:
        raise ValueError(
            f"'{source_column}' does not exist in the GeoDataFrame. Try reconfiguring the 'sub-national_unit_tag' in 'GADM' section in the config file."
        )
        raise ValueError(
            f"'{source_column}' does not exist in the GeoDataFrame. Try reconfiguring the 'sub-national_unit_tag' in 'GADM' section in the config file."
        )

    # Remove spaces in the region names for consistency
    cells[source_column] = cells[source_column].str.replace(" ", "", regex=False)

    # Check if 'x' and 'y' coordinates exist
    if "x" not in cells.columns or "y" not in cells.columns:
        raise ValueError("Columns 'x' and 'y' must exist in the GeoDataFrame.")

    # Generate unique cell IDs using a combination of the region name and coordinates
    cells[index_name] = cells.apply(
        lambda row: f"{row[source_column]}_{row['x']}_{row['y']}", axis=1
    )

    # Set the index to the newly created column
    cells.set_index(index_name, inplace=True)

    # Remove duplicated index values, keeping the first occurrence
    cells = cells[~cells.index.duplicated(keep="first")]

    # Remove duplicated index values, keeping the first occurrence
    cells = cells[~cells.index.duplicated(keep="first")]

    return cells


def get_available_column(dataframe: list, alternatives: list):
    """Return the first column name that exists in the dataframe"""
    for col in alternatives:
        if col in dataframe.columns:
            return col
    return None


def ensure_path(save_to: str | Path, is_file: bool = False) -> Path:
    """
    Ensures that the given argument is a Path object and creates the required directory path.

    ## Args:
        - save_to (str | Path): The path input, either as a string or a Path object.
        - is_file (bool): If True, create only the parent directory and return the file path.
            If False, create the directory represented by save_to.

    ## Returns:
    - Path: The input converted (if necessary) to a Path object.
    """
    if not isinstance(save_to, Path):
        save_to = Path(save_to)

    if is_file:
        save_to.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_to.mkdir(parents=True, exist_ok=True)

    return save_to


# Function to Generate Cell Index from Region name
def assign_regional_cell_ids(cells_dataframe, Source_Column, index_name):
    unique_values = cells_dataframe[Source_Column].unique()

    # If there's only one unique value, return the original DataFrame
    if len(unique_values) == 1:
        return cells_dataframe

    region_dfs = []

    for x in unique_values:
        _mask = cells_dataframe[Source_Column] == x
        _x_df_ = cells_dataframe[_mask].reset_index(drop=True)

        # Create a new column with given index name, for each region
        _x_df_[index_name] = [f"{x}_{index + 1}" for index in range(len(_x_df_))]

        region_dfs.append(_x_df_)

    # Concatenate all site-specific DataFrames into one DataFrame
    dataframe_with_cell_ids = pd.concat(region_dfs, ignore_index=True)

    # Set the index to the newly created column if it exists
    if index_name in dataframe_with_cell_ids.columns:
        dataframe_with_cell_ids.set_index(index_name, inplace=True)

    return dataframe_with_cell_ids


# print_update(level=2, message=f"{__name__}| ❌ ")


def dict_to_pickle(my_dictionary: dict, save_to_path: str):
    """
    Takes dictionary file and saves to given local path as pickle file. Returns a NONE as
    """
    with open(save_to_path, "wb") as file:
        pickle.dump(my_dictionary, file)
    # return None


def pickle_to_dict(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        my_dictionary = pickle.load(file)
    return my_dictionary


def create_blank_yaml(file_path):
    with open(file_path, "w"):
        pass


def save_dict_datafile(dictionary, save_to):
    with open(save_to, "w") as json_file:
        json.dump(dictionary, json_file)
        return print_update(level=2, message=f"{__name__}| Dictionary datafile saved as '{save_to}")


def load_dict_datafile(json_file_path: str) -> dict:
    with open(json_file_path) as json_file:
        dictionary_: dict = json.load(json_file)
        return dictionary_


def save_to_yaml(data: dict, file_path: str | Path, default_name: str = "config.yaml"):
    """
    Saves a dictionary to a YAML file.

    If file_path is a directory, saves to default_name inside that directory.

    Parameters:
        data (dict): The dictionary to save.
        file_path (str|Path): Path to the YAML file or directory.
        default_name (str): Default filename if a directory is given.
    """
    file_path = Path(file_path)

    # If path is a directory, append default filename
    if file_path.exists() and file_path.is_dir():
        file_path = file_path / default_name

    # Ensure parent directories exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save YAML
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print_info(f"{__name__}| A copy of the dictionary saved to : '{file_path}'")


def check_LocalCopy_and_run_function(
    directory_path: str, function_to_run: str, force_update: bool = False
) -> bool:
    """
    Check if a directory exists. If it does, execute the provided function.

    Parameters:
    - directory_path (str): The path to the directory.
    - function_to_run (callable): The function to execute if the directory exists.

    Returns:
    - bool: True if the directory exists and the function is executed, False otherwise.
    """
    if force_update:
        output = function_to_run()
        print_update(
            level=2,
            message=f"{__name__}| Forcefully ran '{function_to_run.__name__}' on '{directory_path}'.",
        )
        return output
    else:
        if not os.path.exists(directory_path):
            output = function_to_run()
            print_update(level=2, message=f"{__name__}| Directory '{directory_path}' created.")
            return output
        else:
            print_update(
                level=2, message=f"{__name__}| Directory '{directory_path}' found locally."
            )


def _merge_config(base: dict, override: dict, path: tuple[str, ...] = ()) -> dict:
    """Deep-merge mappings while replacing scalar and list values."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        current_path = path + (str(key),)
        if not path and key not in KNOWN_CONFIG_TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown top-level configuration key: {key}")
        if key in base and isinstance(base[key], list) and isinstance(value, dict):
            if set(value) != {"$append"} or not isinstance(value["$append"], list):
                dotted = ".".join(current_path)
                raise TypeError(f"List override at {dotted} must be a list or {{$append: [...]}}")
            merged[key] = [*base[key], *value["$append"]]
            continue
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(base[key], value, current_path)
            continue
        if key in base and base[key] is not None and value is not None:
            base_is_mapping = isinstance(base[key], dict)
            value_is_mapping = isinstance(value, dict)
            if base_is_mapping != value_is_mapping:
                dotted = ".".join(current_path)
                raise TypeError(f"Configuration type mismatch at {dotted}")
        merged[key] = value
    return merged


def _override_paths(value, path: tuple[str, ...] = ()) -> list[str]:
    """Return dotted leaf paths represented by an override document."""
    paths = []
    for key, child in value.items():
        if key == "extends":
            continue
        child_path = path + (str(key),)
        if isinstance(child, dict) and child:
            paths.extend(_override_paths(child, child_path))
        else:
            paths.append(".".join(child_path))
    return paths


def resolve_config(
    file_path: str | Path,
    *,
    _chain: tuple[Path, ...] = (),
) -> tuple[dict, dict]:
    """Resolve a YAML configuration and return its provenance.

    ``extends`` paths are resolved relative to the file declaring them. Mappings
    merge recursively; lists and scalar values replace their base values.
    """
    path = Path(file_path).resolve()
    if path in _chain:
        cycle = " -> ".join(str(item) for item in (*_chain, path))
        raise ValueError(f"Circular configuration inheritance: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(raw_text) or {}
    if not isinstance(document, dict):
        raise TypeError(f"Configuration must be a YAML mapping: {path}")

    source_record = {
        "path": str(path),
        "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    parent_reference = document.get("extends")
    if parent_reference is None:
        resolved = dict(document)
        sources = [source_record]
        override_paths = []
    else:
        if not isinstance(parent_reference, str) or not parent_reference.strip():
            raise ValueError(f"extends must be a non-empty relative path in {path}")
        parent_path = (path.parent / parent_reference).resolve()
        parent, parent_provenance = resolve_config(parent_path, _chain=(*_chain, path))
        resolved = _merge_config(parent, document)
        sources = [*parent_provenance["sources"], source_record]
        override_paths = [
            *parent_provenance.get("override_paths", []),
            *_override_paths(document),
        ]

    provenance = {
        "requested_config": str(path),
        "sources": sources,
        "override_paths": list(dict.fromkeys(override_paths)),
    }
    return resolved, provenance


def load_config(file_path: str | Path) -> dict:
    """Load a full or inherited YAML workflow configuration."""
    config, _provenance = resolve_config(file_path)
    return config


def download_data(source_URL: str, file_path: str) -> str:
    """
    Downloads a file from a given URL and saves it to the specified file path.

    Parameters:
        source_URL (str): URL of the file to download.
        file_path (str): Path where the downloaded file will be saved.

    Returns:
        str: The file path if download is successful; otherwise, an instruction message.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send HTTP GET request
        response = requests.get(source_URL, headers=headers, timeout=30)
        ensure_path(Path(file_path).parent)
        print_update(level=3, message=f"{__name__}| ⏬ Downloading data from {source_URL} ...")
        ensure_path(Path(file_path).parent)
        print_update(level=3, message=f"{__name__}| ⏬ Downloading data from {source_URL} ...")

        # Check if the request was successful
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print_update(
                level=3,
                message=f"{__name__}| ✔ File downloaded successfully and saved as {file_path}",
            )
            return file_path
        else:
            print_update(
                level=2,
                message=f"{__name__}| ❌ Failed to download the file. Status code: {response.status_code}",
            )
            return print_update(
                level=2,
                message=f"{__name__}| Please download the data manually from {source_URL} and save it to {file_path}",
            )
    except requests.RequestException as e:
        print_update(
            level=3, message=f"{__name__}| ❌ An error occurred while downloading the file: {e}"
        )
        return print_update(
            level=2,
            message=f"{__name__}| Please download the data manually from {source_URL} and save it to {file_path}",
        )


def load_raster_file(raster_path: str | Path, band: int = 1) -> np.ndarray:
    """
    Loads a GeoTIFF raster as a NumPy array (single band).
    Designed for lightweight array processing.

    Parameters
    ----------
    raster_path : str or Path
        Path to the GeoTIFF raster.
    band : int, default=1
        Raster band to read.

    Returns
    -------
    np.ndarray
        2D array of raster values, or None on failure.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        print(f"❌ File not found: {raster_path}")
        return None

    if raster_path.suffix.lower() not in [".tif", ".tiff"]:
        print(f"❌ Invalid raster format: {raster_path.suffix} (expected .tif or .tiff)")
        return None

    try:
        with rio.open(raster_path) as src:
            data = src.read(band)
            print(f"✅ Loaded raster ({src.width}×{src.height}), res: {src.res[0]:.2f} m")
        return data
    except Exception as e:
        print(f"⚠️ Failed to read raster {raster_path.name}: {e}")
        return None


def get_raster_da(raster_path: str | Path, masked: bool = True):
    """
    Loads a GeoTIFF raster as a rioxarray DataArray with CRS, transform, and resolution metadata.
    Drops all-NaN rows/cols for cleaner data structure.

    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file.
    masked : bool, default=True
        Whether to mask nodata values.

    Returns
    -------
    xarray.DataArray
        Geospatial raster object with CRS and transform metadata.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        print(f"❌ Raster does not exist: {raster_path}")
        return None

    if raster_path.suffix.lower() not in [".tif", ".tiff"]:
        print(f"❌ Invalid raster file: {raster_path}")
        return None

    try:
        da = (
            rxr.open_rasterio(raster_path, masked=masked)
            .squeeze(drop=True)
            .dropna(dim="x", how="all")
            .dropna(dim="y", how="all")
        )

        print(f"✅ Loaded DataArray: {raster_path.name}")
        print(f"   ├─ shape: {da.shape}")
        print(f"   ├─ CRS: {da.rio.crs}")
        print(f"   └─ res: {da.rio.resolution()}")
        return da

    except Exception as e:
        print(f"⚠️ Failed to load DataArray from {raster_path.name}: {e}")
        loading_via_rioxarray_fails = True

        if loading_via_rioxarray_fails:
            fallback = load_raster_file(raster_path)
            return fallback


def check_raster_classes(
    source_da: xr.DataArray, clipped_da: xr.DataArray, boundary_gdf: gpd.GeoDataFrame
):
    """
    Scientifically validates if class loss is due to spatial absence
    or clipping artifacts.
    """

    # 1. Clean data: Remove NoData/NaN to focus on thematic classes
    def get_clean_unique(da):
        vals = da.values.flatten()
        return np.unique(vals[~np.isnan(vals) & (vals != 0)])

    clipped_classes = get_clean_unique(clipped_da)

    # 2. Critical Step: Find what SHOULD be there based on the geometry
    # We use the bounding box of the boundary to slice the source first
    bbox = boundary_gdf.total_bounds
    source_subset = source_da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
    expected_classes = get_clean_unique(source_subset)

    # 3. Identify actual artifacts (Lost despite being in the geographic extent)
    lost_artifacts = np.setdiff1d(expected_classes, clipped_classes)

    print("--- Raster Integrity Report ---")
    print(f"Unique classes in Geographic Extent: {len(expected_classes)}")
    print(f"Unique classes in Clipped Result:    {len(clipped_classes)}")

    if len(lost_artifacts) > 0:
        print(f"❌ Artifact Alert: {len(lost_artifacts)} classes lost due to clipping logic.")
        print(f"Missing IDs: {lost_artifacts}")
    else:
        print("✅ Rigor Check Passed: No classes were lost during the geometric clip.")


def standardize_tags(name: str) -> str:
    """Remove all whitespace from a string for use in IDs, file names, etc."""
    if not isinstance(name, str):
        return name
    return name.replace(" ", "")


# check_raster_classes(CLC_da, CLC_raster_WB6_da, WB6_boundary_dissolved_reproj)
