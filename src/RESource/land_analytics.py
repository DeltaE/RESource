"""Store regional land-layer visualizations without storing raster data."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds

from RESource import utility as utils
from RESource.hdf5_handler import DataHandler
from RESource.step_cache import StepCache

_LEGENDS = Path(__file__).parent / "assets" / "legends"
_BUILTIN_LEGENDS = {
    ("exclusion_areas", "exclusion_2017.tif"): "gaez_exclusion_legend.csv",
    ("terrain_resources", "slpmed05.tif"): "gaez_terrains_legend.csv",
    ("terrain_resources", "slpmed30s.tif"): "gaez_terrains_legend.csv",
    ("CORINE_land_cover", "U2018_CLC2018_V2020_20u1.tif"): "CLC_2018_legend.csv",
}


def land_analytics_version(config: dict, region: str) -> str:
    """Identify the land configuration used to create images in a regional store."""
    relevant = {
        key: config.get(key)
        for key in ("lands", "custom_land_layers", "filters", "region_mapping", "admin_boundary")
    }
    relevant["region"] = region
    encoded = json.dumps(StepCache._stringify_keys(relevant), sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _image_exists(handler: DataHandler, key: str) -> bool:
    with h5py.File(handler.store, "a") as store:
        return f"LandAnalytics/images/{key}" in store


def _store_figure(handler: DataHandler, key: str, fig, **attrs) -> None:
    with io.BytesIO() as output:
        fig.savefig(output, format="png", dpi=150, bbox_inches="tight")
        handler.save_image(output.getvalue(), key=key, **attrs)


def _legend(layer: dict, classes: list[int]) -> dict[int, tuple[str, str]]:
    legend_path = layer.get("legends")
    builtin = _BUILTIN_LEGENDS.get((layer["name"], layer.get("raster")))
    if not legend_path and builtin:
        legend_path = _LEGENDS / builtin
    entries = {}
    if legend_path:
        legend = pd.read_csv(legend_path)
        for code, description, color in legend[["class", "description", "color"]].itertuples(
            index=False, name=None
        ):
            entries[int(code)] = (str(description), str(color))
    cmap = plt.get_cmap("tab20")
    for index, code in enumerate(classes):
        if code not in entries:
            entries[code] = (f"Class {code}", cmap(index % 20))
    return entries


def store_land_layer_images(
    handler: DataHandler, layers: list[dict], boundary: gpd.GeoDataFrame, version: str
) -> None:
    """Store class distributions and overview maps of configured categorical rasters.

    Counts use all valid pixel centres within the region; maps are downsampled.
    Shares are pixel counts, not area-weighted measurements.
    """
    if boundary.crs is None or boundary.empty:
        raise ValueError("Land analytics require a non-empty boundary with a CRS")
    for layer in layers:
        path = Path(layer["filepath"])
        name = layer["name"]
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        base = f"pipeline_{slug}_{identity}_{version}"
        distribution_key, map_key = f"{base}_distribution", f"{base}_map"
        if _image_exists(handler, distribution_key) and _image_exists(handler, map_key):
            continue

        with rasterio.open(path) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError(f"Expected a single-band raster with a CRS: {path}")
            if not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                utils.print_warning(f"Skipping non-categorical floating-point land raster: {path}")
                continue
            region = boundary.to_crs(src.crs)
            geometries = list(region.geometry)
            counts: Counter[int] = Counter()
            for _, window in src.block_windows(1):
                data = src.read(1, window=window, masked=True)
                inside = geometry_mask(
                    geometries,
                    out_shape=data.shape,
                    transform=src.window_transform(window),
                    invert=True,
                )
                values = data.data[inside & ~np.ma.getmaskarray(data)]
                codes, frequencies = np.unique(values, return_counts=True)
                counts.update({int(k): int(v) for k, v in zip(codes, frequencies, strict=True)})
                if len(counts) > 256:
                    break
            if not counts:
                utils.print_warning(f"No valid land-class pixels inside the boundary: {path}")
                continue
            if len(counts) > 256:
                utils.print_warning(f"Skipping land raster with more than 256 classes: {path}")
                continue
            classes = sorted(counts)
            legend = _legend(layer, classes)
            scale = min(1.0, 1200 / max(src.height, src.width))
            height, width = max(1, round(src.height * scale)), max(1, round(src.width * scale))
            sample = src.read(1, out_shape=(height, width), masked=True)
            transform = src.transform * src.transform.scale(src.width / width, src.height / height)
            inside = geometry_mask(
                geometries, out_shape=(height, width), transform=transform, invert=True
            )
            sampled = np.ma.array(sample.data, mask=np.ma.getmaskarray(sample) | ~inside)
            bounds = array_bounds(height, width, transform)

        metadata = {"source": str(path), "layer": name, "config_version": version}
        if not _image_exists(handler, distribution_key):
            fig, ax = plt.subplots(figsize=(9, max(3, len(classes) * 0.3)))
            try:
                ordered = sorted(classes, key=lambda code: counts[code])
                labels = [f"{code}: {legend[code][0]}" for code in ordered]
                shares = [100 * counts[code] / sum(counts.values()) for code in ordered]
                ax.barh(labels, shares, color=[legend[code][1] for code in ordered])
                ax.set_xlabel("Share of valid pixels (%)")
                ax.set_title(f"{name}: class distribution")
                fig.tight_layout()
                _store_figure(
                    handler, distribution_key, fig, title=f"{name}: class distribution", **metadata
                )
            finally:
                plt.close(fig)

        if not _image_exists(handler, map_key):
            fig, ax = plt.subplots(figsize=(9, 7))
            try:
                indices = np.searchsorted(classes, sampled.data)
                valid_indices = np.minimum(indices, len(classes) - 1)
                unmatched = np.asarray(classes)[valid_indices] != sampled.data
                indexed = np.ma.array(indices, mask=np.ma.getmaskarray(sampled) | unmatched)
                cmap = ListedColormap([legend[code][1] for code in classes])
                ax.imshow(
                    indexed,
                    cmap=cmap,
                    vmin=-0.5,
                    vmax=len(classes) - 0.5,
                    extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                    origin="upper",
                )
                region.boundary.plot(ax=ax, color="black", linewidth=0.5)
                ax.legend(
                    handles=[
                        Patch(color=legend[code][1], label=f"{code}: {legend[code][0]}")
                        for code in classes
                    ],
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    fontsize=7,
                )
                ax.set_title(f"{name}: land classes")
                ax.set_axis_off()
                _store_figure(handler, map_key, fig, title=f"{name}: land classes", **metadata)
            finally:
                plt.close(fig)


def store_land_impact_images(
    handler: DataHandler, plot_dir: Path, resource_type: str, version: str
) -> None:
    """Store freshly generated cumulative and individual layer impact plots."""
    for kind in ("stepwise", "individual_layers_impact"):
        filename = (
            f"stepwise_land_availability_plot_{resource_type}.png"
            if kind == "stepwise"
            else f"individual_layers_impact_land_availability_plot_{resource_type}.png"
        )
        key = f"pipeline_{kind}_{resource_type}_{version}"
        if not _image_exists(handler, key):
            handler.save_image(
                plot_dir / filename,
                key=key,
                title=f"{kind.replace('_', ' ').title()} land availability: {resource_type}",
                resource_type=resource_type,
                config_version=version,
            )


def land_impact_images_exist(handler: DataHandler, resource_type: str, version: str) -> bool:
    """Check that both resource-specific impact figures are present."""
    return all(
        _image_exists(handler, f"pipeline_{kind}_{resource_type}_{version}")
        for kind in ("stepwise", "individual_layers_impact")
    )


def run_land_analytics(
    handler: DataHandler, land_container, resource_type: str, version: str, capacity_cached: bool
) -> None:
    """Generate missing land images, including enabled impact plots on cached runs."""
    if land_container.plot_layer_impacts:
        if not land_impact_images_exist(handler, resource_type, version):
            if capacity_cached:
                land_container.set_excluder()
            store_land_impact_images(
                handler, land_container.vis_root / "lands", resource_type, version
            )
    store_land_layer_images(
        handler,
        land_container.get_layers(raster_only=True),
        land_container.region_boundary,
        version,
    )
