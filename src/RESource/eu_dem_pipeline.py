#!/usr/bin/env python3
"""
Copernicus DEM (GLO-30) / EU-DEM pipeline for WB6 (streaming mosaic):
- Auto-generate Copernicus DEM GLO-30 tile URLs from WB6 boundary (or fallback bbox), and/or merge with a URLs file
- Robust downloads (skip 403/404), optional tile cleanup
- **Streaming mosaic** via gdalbuildvrt + gdal_translate (fallback to rasterio.merge if GDAL CLI unavailable)
- **Reproject to EPSG:3035 with GDAL (progress) or rasterio fallback**
- **Compute slope (%) with GDAL (progress) or NumPy fallback**
- Optional clip to WB6 boundary
- Optional aggregate slope to 100 m (for CLC alignment)
- Reclass slope (%) into 9 GAEZ-like bins
- Write outputs to your repo’s data/ structure

"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from shutil import which

# modest performance + remote read hints
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")

import numpy as np
import rasterio
import requests
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

try:
    import geopandas as gpd
except Exception:
    gpd = None
    warnings.warn(
        "geopandas not found. Clip-to-boundary and boundary auto-build will be disabled.",
        RuntimeWarning,
        stacklevel=2,
    )

# ----------------------------
# Data paths. RESOURCE_DATA_DIR lets installed applications keep large data
# outside the Python environment; the repository's ./data remains the default.
# ----------------------------
DATA_ROOT = Path(os.environ.get("RESOURCE_DATA_DIR", "data")).expanduser()
DL_ROOT = DATA_ROOT / "downloaded_data" / "EU_DEM"
PROC_ROOT = DATA_ROOT / "processed_data" / "EU_DEM"

TILES_DIR = DL_ROOT / "tiles"
MOSAIC_RAW = DL_ROOT / "WB6_EUDEM_raw_mosaic.vrt.tif"  # GeoTIFF mosaic (name hints it's a mosaic)
SLOPE_25 = PROC_ROOT / "WB6_slope_pct_25m.tif"
SLOPE_100 = PROC_ROOT / "WB6_slope_pct_100m.tif"
SLOPE_CLASS = PROC_ROOT / "WB6_slope_class_uint8.tif"


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    for p in [DL_ROOT, PROC_ROOT, TILES_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _is_zip(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def _safe_name(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if not name:
        name = "tile.tif"
    return name


def copdem30_tile_url(lat_deg: int, lon_deg: int) -> str:
    """
    Build the public AWS URL for Copernicus DEM GLO-30 COG tiles.
    NOTE: GLO-30 uses '10' in the token (1-arcsec), bucket 'copernicus-dem-30m.s3.amazonaws.com'.
    """
    NS = f"N{lat_deg:02d}_00" if lat_deg >= 0 else f"S{abs(lat_deg):02d}_00"
    EW = f"E{lon_deg:03d}_00" if lon_deg >= 0 else f"W{abs(lon_deg):03d}_00"
    base = f"Copernicus_DSM_COG_10_{NS}_{EW}_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{base}/{base}.tif"


def copdem30_urls_from_bbox(lat_min: int, lat_max: int, lon_min: int, lon_max: int) -> list[str]:
    """Inclusive lower bounds, exclusive upper bounds (Python range semantics)."""
    urls: list[str] = []
    for la in range(lat_min, lat_max):
        for lo in range(lon_min, lon_max):
            urls.append(copdem30_tile_url(la, lo))
    return urls


def copdem30_urls_from_boundary(boundary_file: Path) -> list[str]:
    """
    Intersect a boundary polygon with a 1x1-degree grid to get only intersecting tiles.
    Requires geopandas/shapely; falls back to coarse bbox if not available.
    """
    if gpd is None:
        return copdem30_urls_from_bbox(lat_min=40, lat_max=47, lon_min=15, lon_max=24)

    gdf = gpd.read_file(boundary_file)
    gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    lon_min = int(math.floor(minx))
    lon_max = int(math.ceil(maxx))
    lat_min = int(math.floor(miny))
    lat_max = int(math.ceil(maxy))

    from shapely.geometry import box

    wb_union = gdf.unary_union
    urls: list[str] = []
    for la in range(lat_min, lat_max):
        for lo in range(lon_min, lon_max):
            tile_poly = box(lo, la, lo + 1, la + 1)
            if tile_poly.intersects(wb_union):
                urls.append(copdem30_tile_url(la, lo))
    return urls


def download_tiles(urls: list[str], out_dir: Path) -> list[Path]:
    """Download tiles; skip 403/404 (e.g., ocean-only tiles); continue on errors."""
    paths: list[Path] = []
    for url in urls:
        fname = _safe_name(url)
        out_path = out_dir / fname
        if out_path.exists():
            paths.append(out_path)
            continue
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code in (403, 404):
                    print(f"[WARN] Skipping missing tile: {url} (HTTP {r.status_code})")
                    continue
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with (
                    open(out_path, "wb") as f,
                    tqdm(
                        total=total, unit="B", unit_scale=True, desc=f"Downloading {fname}"
                    ) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            paths.append(out_path)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Error downloading {url}: {e}. Skipping.")
            continue
    return paths


def extract_if_zip(files: list[Path], out_dir: Path) -> list[Path]:
    """
    Accept both .zip and .tif inputs. Returns a list of .tif paths ready for mosaic.
    For Copernicus GLO-30, tiles are COG .tif (no zip) — this still works.
    """
    tifs: list[Path] = []
    for p in files:
        if _is_zip(p):
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(out_dir)
                for member in z.namelist():
                    if member.lower().endswith(".tif"):
                        tifs.append(out_dir / member)
        else:
            if p.suffix.lower() == ".tif":
                tifs.append(p)
    return tifs


# ---------- Streaming mosaic (preferred) ----------
def _have_gdal_cli() -> bool:
    return which("gdalbuildvrt") is not None and which("gdal_translate") is not None


def gdal_mosaic(tifs: list[Path], out_path: Path) -> Path:
    """Streaming mosaic using gdalbuildvrt + gdal_translate (fast, low RAM)."""
    vrt_path = out_path.with_suffix(".vrt")
    tif_list = [str(t) for t in tifs]
    # Build VRT
    subprocess.run(
        ["gdalbuildvrt", str(vrt_path)] + tif_list,
        check=True,
        capture_output=True,
    )
    # Translate to compressed GeoTIFF
    subprocess.run(
        [
            "gdal_translate",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "BIGTIFF=YES",
            str(vrt_path),
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    try:
        Path(vrt_path).unlink()  # clean up
    except Exception:
        pass
    return out_path


# ---------- Fallback in-memory mosaic ----------
def rasterio_mosaic(tifs: list[Path], out_path: Path) -> Path:
    """In-memory mosaic with rasterio.merge (slower, high RAM)."""
    srcs = [rasterio.open(str(t)) for t in tifs]
    mosaic_data, mosaic_transform = rio_merge(srcs)
    profile = srcs[0].profile.copy()
    for s in srcs:
        s.close()
    profile.update(
        height=mosaic_data.shape[1],
        width=mosaic_data.shape[2],
        transform=mosaic_transform,
        compress="LZW",
        BIGTIFF="YES",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic_data)
    return out_path


# ---------- Reproject (GDAL preferred) ----------
def gdal_reproject_3035(in_path: Path, out_path: Path, target_res_m: float) -> Path:
    """Reproject with gdalwarp (shows progress; multi-threaded)."""
    cmd = [
        "gdalwarp",
        "-t_srs",
        "EPSG:3035",
        "-tr",
        str(target_res_m),
        str(target_res_m),
        "-r",
        "bilinear",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-co",
        "COMPRESS=LZW",
        "-co",
        "BIGTIFF=YES",
        "-overwrite",
        str(in_path),
        str(out_path),
    ]
    print("[INFO] gdalwarp:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def reproject_to_3035(in_path: Path, out_path: Path, target_res_m: float = 30.0) -> Path:
    """Rasterio/NumPy fallback reproject (single-threaded, no progress)."""
    with rasterio.open(in_path) as src:
        dst_crs = "EPSG:3035"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_res_m
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress="LZW",
            BIGTIFF="YES",
        )
        data = np.empty((src.count, height, width), dtype=src.dtypes[0])
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                num_threads=2,
            )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return out_path


# ---------- Clip ----------
def clip_to_boundary(in_path: Path, boundary_path: Path | None, out_path: Path) -> Path:
    """Clip DEM to polygon boundary if provided; otherwise pass-through (copy)."""
    if boundary_path is None:
        if in_path != out_path:
            shutil.copyfile(in_path, out_path)
        return out_path

    if gpd is None:
        warnings.warn("geopandas not installed; skipping clip.", RuntimeWarning, stacklevel=2)
        if in_path != out_path:
            shutil.copyfile(in_path, out_path)
        return out_path

    with rasterio.open(in_path) as src:
        gdf = gpd.read_file(boundary_path)
        if gdf.crs is None or gdf.crs.to_string() != src.crs.to_string():
            gdf = gdf.to_crs(src.crs)
        shapes = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
        out_img, out_transform = rio_mask(src, shapes=shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "compress": "LZW",
                "BIGTIFF": "YES",
            }
        )
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_img)
    return out_path


# ---------- Slope (GDAL preferred) ----------
def gdal_slope_percent(in_path: Path, out_path: Path) -> Path:
    """Compute slope (%) with gdaldem (shows progress; multi-threaded where possible)."""
    cmd = [
        "gdaldem",
        "slope",
        str(in_path),
        str(out_path),
        "-p",
        "-compute_edges",
        "-of",
        "GTiff",
        "-co",
        "COMPRESS=LZW",
        "-co",
        "BIGTIFF=YES",
    ]
    print("[INFO] gdaldem:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def compute_slope_percent_from_dem(in_path: Path, out_path: Path) -> Path:
    """NumPy fallback slope (%) using Horn kernel on a single-band DEM in EPSG:3035."""
    with rasterio.open(in_path) as src:
        dem = src.read(1, masked=True).astype("float32")
        transform = src.transform
        px = transform.a
        py = -transform.e
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32") / (8.0 * px)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float32") / (8.0 * py)

        dem_filled = np.where(dem.mask, np.nan, dem.filled(np.nan))

        def conv2(a, k):
            pad_y, pad_x = k.shape[0] // 2, k.shape[1] // 2
            ap = np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
            out = np.empty_like(a, dtype="float32")
            for i in range(out.shape[0]):
                ai = i + pad_y
                for j in range(out.shape[1]):
                    aj = j + pad_x
                    window = ap[ai - pad_y : ai + pad_y + 1, aj - pad_x : aj + pad_x + 1]
                    if np.any(np.isnan(window)):
                        out[i, j] = np.nan
                    else:
                        out[i, j] = np.sum(window * k, dtype="float32")
            return out

        dzdx = conv2(dem_filled, kx)
        dzdy = conv2(dem_filled, ky)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_pct = np.tan(slope_rad) * 100.0
        slope_pct = np.where(np.isnan(slope_pct), np.nan, slope_pct).astype("float32")

        profile = src.profile.copy()
        profile.update(dtype="float32", compress="LZW", BIGTIFF="YES")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(slope_pct, 1)
    return out_path


# ---------- Resample average ----------
def resample_average(in_path: Path, out_path: Path, target_res_m: float) -> Path:
    with rasterio.open(in_path) as src:
        dst_crs = src.crs
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_res_m
        )
        profile = src.profile.copy()
        profile.update(
            transform=transform, width=width, height=height, compress="LZW", BIGTIFF="YES"
        )
        dst = np.empty((1, height, width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            num_threads=2,
        )
        with rasterio.open(out_path, "w", **profile) as d:
            d.write(dst[0], 1)  # write a 2-D band

    return out_path


# ---------- Reclass ----------
def reclass_slope_to_bins(
    in_path: Path,
    out_path: Path,
    bins: tuple[float, ...] = (0.5, 2, 5, 8, 16, 30, 45, 1e9),
    water_mask_path: Path | None = None,
) -> Path:
    """
    bins define upper edges for classes 1..8 (last is sentinel).
    Class 9 can be assigned to water if water_mask_path is given (values==1).
    NoData=255 for classes.
    """
    with rasterio.open(in_path) as src:
        slope = src.read(1, masked=True).astype("float32")
        arr = slope.filled(np.nan)
        classes = np.digitize(arr, bins, right=True).astype("uint8") + 1
        classes[~np.isfinite(arr) | slope.mask] = 255

        if water_mask_path:
            with rasterio.open(water_mask_path) as wm:
                water_on_grid = np.zeros((src.height, src.width), dtype="float32")
                reproject(
                    source=rasterio.band(wm, 1),
                    destination=water_on_grid,
                    src_transform=wm.transform,
                    src_crs=wm.crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )
                water_bool = water_on_grid == 1
                classes[water_bool] = 9

        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, nodata=255, compress="LZW", BIGTIFF="YES")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(classes, 1)
    return out_path


# ---------- Build WB6 boundary ----------
def make_wb6_boundary(
    regions_dir: Path,
    pattern: str = "gadm41_Western Balkan Regions_L1_*.geojson",
    out_name: str = "WB6_laea3035.geojson",
    force: bool = False,
) -> Path | None:
    """
    Dissolve AL/BA/XK/ME/MK/RS L1 GeoJSONs into a single WB6 boundary in EPSG:3035.
    Returns the path to the output file if created/found, else None.
    """
    if gpd is None:
        print("[WARN] geopandas not available; cannot auto-build WB6 boundary.", file=sys.stderr)
        return None

    regions_dir = regions_dir.resolve()
    out_path = regions_dir / out_name

    if out_path.exists() and not force:
        return out_path

    files = sorted(regions_dir.glob(pattern))
    if not files:
        print(
            f"[WARN] No inputs matching '{pattern}' in {regions_dir} — cannot auto-build boundary.",
            file=sys.stderr,
        )
        return None

    try:
        import pandas as pd  # local import to avoid hard dep if not needed elsewhere

        gdfs = [gpd.read_file(f) for f in files]
        g = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        wb6 = g.dissolve().to_crs("EPSG:3035")
        regions_dir.mkdir(parents=True, exist_ok=True)
        wb6.to_file(out_path, driver="GeoJSON")
        print(f"[INFO] Built WB6 boundary → {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARN] Failed to auto-build WB6 boundary: {e}", file=sys.stderr)
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WB6 slope pipeline (EPSG:3035) using Copernicus DEM GLO-30 tiles with streaming mosaic"
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Text file with DEM tile URLs (one per line). If omitted, only auto-generated URLs (if requested) are used.",
    )
    parser.add_argument(
        "--use-copdem30-wb6",
        action="store_true",
        help="Auto-generate Copernicus DEM GLO-30 tile URLs for the WB6 area.",
    )
    parser.add_argument(
        "--boundary-file",
        type=str,
        default=None,
        help="Optional polygon file (GeoJSON/Shapefile). If present, used for URL gen & clipping.",
    )
    parser.add_argument(
        "--auto-boundary-from",
        type=str,
        default=str(DATA_ROOT / "processed_data" / "regions"),
        help="Directory with per-country WB6 GeoJSONs to auto-build a dissolved boundary (if needed).",
    )
    parser.add_argument(
        "--auto-boundary-pattern",
        type=str,
        default="gadm41_Western Balkan Regions_L1_*.geojson",
        help="Glob pattern for input country GeoJSONs.",
    )
    parser.add_argument(
        "--auto-boundary-out",
        type=str,
        default="WB6_laea3035.geojson",
        help="Output filename for dissolved WB6 boundary (EPSG:3035).",
    )
    parser.add_argument(
        "--force-make-boundary",
        action="store_true",
        help="Force rebuild of the dissolved boundary even if it exists.",
    )
    parser.add_argument(
        "--target-res",
        type=float,
        default=30.0,
        help="Target metric resolution for DEM reprojection (meters). Default 30 (GLO-30).",
    )
    parser.add_argument("--make-100m", action="store_true", help="Also create a 100 m slope (%).")
    parser.add_argument(
        "--water-mask",
        type=str,
        default=None,
        help="Optional water mask raster (1=water) to set class 9.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading even if URLs are provided or auto-generated.",
    )
    parser.add_argument(
        "--clean-tiles",
        action="store_true",
        help="Delete existing .tif in tiles/ before downloading new ones.",
    )

    # NEW: prefer/force GDAL for reproject & slope + debug
    parser.add_argument(
        "--force-gdal",
        action="store_true",
        help="Force use of GDAL CLI (gdalwarp/gdaldem) for reprojection and slope if available.",
    )
    parser.add_argument(
        "--debug-gdal", action="store_true", help="Print which GDAL executables are detected."
    )

    args = parser.parse_args()

    if args.debug_gdal:
        print("[DEBUG] PATH:", os.environ.get("PATH", ""))
        print("[DEBUG] gdalbuildvrt:", which("gdalbuildvrt"))
        print("[DEBUG] gdal_translate:", which("gdal_translate"))
        print("[DEBUG] gdalwarp:", which("gdalwarp"))
        print("[DEBUG] gdaldem:", which("gdaldem"))

    # Resolve/auto-build boundary
    boundary_path: Path | None = None
    if args.boundary_file:
        p = Path(args.boundary_file)
        if not p.is_absolute():
            p_try = p
            p_repo = Path.cwd() / p
            if p_try.exists():
                boundary_path = p_try
            elif p_repo.exists():
                boundary_path = p_repo
        else:
            if p.exists():
                boundary_path = p

    if boundary_path is None:
        regions_dir = Path(args.auto_boundary_from)
        maybe = make_wb6_boundary(
            regions_dir=regions_dir,
            pattern=args.auto_boundary_pattern,
            out_name=args.auto_boundary_out,
            force=args.force_make_boundary,
        )
        if maybe and maybe.exists():
            boundary_path = maybe
        else:
            print(
                "[WARN] Proceeding without a WB6 boundary; will use coarse bbox and skip clipping.",
                file=sys.stderr,
            )

    ensure_dirs()

    # Optional: clean tiles dir
    if args.clean_tiles:
        for old in TILES_DIR.glob("*.tif"):
            try:
                old.unlink()
            except Exception as e:
                print(f"[WARN] Could not remove {old}: {e}", file=sys.stderr)

    # Build URL list
    urls: list[str] = []
    if args.use_copdem30_wb6:
        if boundary_path and boundary_path.exists():
            urls.extend(copdem30_urls_from_boundary(boundary_path))
        else:
            urls.extend(
                copdem30_urls_from_bbox(lat_min=39, lat_max=47, lon_min=15, lon_max=24)
            )  # slightly expanded south

    if args.urls_file:
        with open(args.urls_file) as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    urls.append(u)

    # De-duplicate while preserving order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    # Only mosaic tiles we expect this run (avoid stale leftovers)
    expected_names = {_safe_name(u) for u in urls} if urls else set()

    # Download tiles
    tif_candidates: list[Path] = []
    if urls and not args.skip_download:
        print(f"[INFO] Downloading {len(urls)} tile(s) to {TILES_DIR} ...")
        downloaded = download_tiles(urls, TILES_DIR)
        tif_candidates.extend(downloaded)

    # Collect/filter tiles for mosaic
    def _select_for_mosaic(candidates: list[Path], expected_names: set[str]) -> list[Path]:
        if not expected_names:
            return candidates  # no filter; use all found
        return [p for p in candidates if p.name in expected_names]

    candidates = list(set((tif_candidates or []) + list(TILES_DIR.glob("*.tif"))))
    tifs_all = extract_if_zip(candidates, TILES_DIR)
    tifs = _select_for_mosaic(tifs_all, expected_names)

    if not tifs:
        print(
            f"[ERROR] No GeoTIFF tiles selected in {TILES_DIR}. Provide tiles or URLs.",
            file=sys.stderr,
        )
        sys.exit(2)
    else:
        print(f"[INFO] Selected {len(tifs)} tile(s) for mosaic (of {len(tifs_all)} present).")

    # Mosaic (streaming if possible)
    print("[INFO] Mosaicking tiles…")
    if _have_gdal_cli():
        print("[INFO] Using gdalbuildvrt + gdal_translate (streaming).")
        mosaic_path = gdal_mosaic(tifs, MOSAIC_RAW)
    else:
        print(
            "[WARN] GDAL CLI not found; falling back to in-memory rasterio.merge (slower, high RAM)."
        )
        mosaic_path = rasterio_mosaic(tifs, MOSAIC_RAW)
    print("[INFO] Mosaic written.")

    # Reproject to 3035; dynamic filenames by res
    dem_res_str = f"{int(args.target_res)}m"
    dem_out_path = DL_ROOT / f"WB6_EUDEM_{dem_res_str}_laea3035.tif"
    dem_clip_path = DL_ROOT / f"WB6_EUDEM_{dem_res_str}_laea3035_clip.tif"

    print(f"[INFO] Reprojecting mosaic to EPSG:3035 at {int(args.target_res)} m …")
    if (args.force_gdal and which("gdalwarp")) or (not args.force_gdal and which("gdalwarp")):
        reproj_path = gdal_reproject_3035(mosaic_path, dem_out_path, target_res_m=args.target_res)
    else:
        reproj_path = reproject_to_3035(mosaic_path, dem_out_path, target_res_m=args.target_res)

    # Optional clip
    if boundary_path and boundary_path.exists():
        print("[INFO] Clipping DEM to WB6 boundary …")
        clip_path = clip_to_boundary(reproj_path, boundary_path, dem_clip_path)
        dem_for_slope = clip_path
    else:
        dem_for_slope = reproj_path

    # Slope
    slope_native_path = (
        SLOPE_25
        if args.target_res <= 25.0
        else PROC_ROOT / f"WB6_slope_pct_{int(args.target_res)}m.tif"
    )
    print("[INFO] Computing slope (%) …")
    if (args.force_gdal and which("gdaldem")) or (not args.force_gdal and which("gdaldem")):
        gdal_slope_percent(dem_for_slope, slope_native_path)
    else:
        compute_slope_percent_from_dem(dem_for_slope, slope_native_path)

    # Optional 100 m average (for CLC alignment)
    if args.make_100m:
        print("[INFO] Averaging slope to 100 m …")
        resample_average(slope_native_path, SLOPE_100, target_res_m=100.0)
        slope_for_class = SLOPE_100
    else:
        slope_for_class = slope_native_path

    # Reclass to 1..8 (optionally 9=water)
    print("[INFO] Reclassing slope to GAEZ-like bins …")
    water_mask_path = Path(args.water_mask) if args.water_mask else None
    reclass_slope_to_bins(slope_for_class, SLOPE_CLASS, water_mask_path=water_mask_path)

    print("\n✅ Done.")
    print(f"DEM (3035): {dem_out_path}")
    if boundary_path and boundary_path.exists():
        print(f"DEM (3035, clipped): {dem_clip_path}")
    print(f"Slope % ({int(args.target_res)}m): {slope_native_path}")
    if args.make_100m:
        print(f"Slope % (100m): {SLOPE_100}")
    print(f"Slope classes (uint8): {SLOPE_CLASS}  [1..8 (by bins), optional 9=water, 255=NoData]")


if __name__ == "__main__":
    main()
