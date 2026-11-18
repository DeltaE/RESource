"""Turn one RESource store into the JSON payload the dash renders.

Everything here is derived from the store. Nothing is invented: a table or column
that is absent yields an absent block, and the dash says so.
Depends on store_reader (same namespace when inlined into the dash).
"""

from __future__ import annotations

import base64
import json
import re

import h5py
import numpy as np
import pandas as pd

try:  # native import; inside the dash both modules share one namespace
    from store_reader import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    from .store_reader import *  # noqa: F401,F403

TECHS = ("solar", "wind")
LCOE_SENTINEL = 999_000.0  # RESource writes 999999 where a cell has no viable LCOE
NAME_RE = re.compile(r"^resources_([^_]+)_([^_]+)_(.+)_(\d{4})_(\d{8})\.h5$")
REGION_COLS = ("Region", "Municipality", "Province", "Country")
MAX_LINE_FEATURES = 8000


def parse_store_name(filename):
    m = NAME_RE.match(filename)
    if not m:
        return {"file": filename}
    c, r, s, y, d = m.groups()
    return {"file": filename, "country": c, "region": r, "scenario": s, "year": int(y),
            "run_date": f"{d[:4]}-{d[4:6]}-{d[6:]}"}


def _num(a, sig=5):
    """Float array -> JSON-safe list, rounded to ``sig`` significant figures, NaN/inf -> None."""
    a = np.asarray(a, dtype=float)
    out = []
    for v in a:
        out.append(None if not np.isfinite(v) else float(f"{v:.{sig}g}"))
    return out


def _f(v, sig=5):
    return None if v is None or not np.isfinite(v) else float(f"{float(v):.{sig}g}")


def _region_col(df):
    return next((c for c in REGION_COLS if c in df.columns), None)


def _wmean(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else None


# ------------------------------------------------------------------ blocks

def _outline(wkts):
    """Outer outline of all boundary units (the store's level-1 region), or None without shapely."""
    try:
        from shapely import wkt as swkt
        from shapely.ops import unary_union
    except ImportError:
        return None
    try:
        geom = unary_union([swkt.loads(g).buffer(0) for g in wkts if isinstance(g, str) and g])
        geom = geom.buffer(0.01).buffer(-0.01)            # close slivers between neighbouring units
        parts = list(getattr(geom, "geoms", [geom]))
        big = max(g.area for g in parts)
        geom = unary_union([g for g in parts if g.area >= 0.002 * big])   # drop specks, keep real islands
        geom = geom.simplify(0.01, preserve_topology=True)
        return wkt_parts(geom.wkt, max_pts=400, max_parts=60)
    except Exception:  # an invalid geometry must not hide the boundary itself
        return None


def _boundary(f):
    df = read_fixed_frame(f, "boundary")
    rc = _region_col(df)
    feats = [wkt_parts(g, max_pts=300, max_parts=40) for g in df["geometry"]]
    return {"name_col": rc, "names": [str(v) for v in df[rc]] if rc else [str(i) for i in df.index],
            "rings": feats, "n": int(len(df)), "outline": _outline(list(df["geometry"])),
            "note": "Outlines thinned to 300 vertices per ring, 40 largest rings per region. Display only."}


def _cells(f, adjustments=None):
    df = read_fixed_frame(f, "cells")
    if adjustments is not None:
        for tech in TECHS:
            column = f"potential_capacity_{tech}"
            df[f"{column}_unadjusted"] = df[column]
            df[column] = adjustments["cells"][f"{column}_adjusted"].to_numpy()
    rc = _region_col(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sentinel = {}
    cols = {}
    for c in numeric:
        v = df[c].to_numpy(dtype=float)
        if c.startswith("lcoe"):
            bad = v >= LCOE_SENTINEL
            sentinel[c] = int(bad.sum())
            v = np.where(bad, np.nan, v)
        cols[c] = _num(v)
    return df, {
        "ids": [str(i) for i in df.index], "region_col": rc,
        "region": [str(v) for v in df[rc]] if rc else None,
        "rings": [wkt_parts(g, max_pts=40, max_parts=6) for g in df["geometry"]] if "geometry" in df else None,
        "cols": cols, "numeric": numeric, "sentinel": sentinel, "n": int(len(df)),
    }


def _lines(f):
    df = read_fixed_frame(f, "lines")
    volt = df["voltage"].to_numpy() if "voltage" in df else np.zeros(len(df))
    kind = df["power"].astype(str).to_numpy() if "power" in df else np.array(["line"] * len(df))
    order = np.argsort(-volt, kind="stable")
    dropped = max(0, len(df) - MAX_LINE_FEATURES)
    lines, points = [], []
    geoms = df["geometry"].to_numpy()
    for i in order[:MAX_LINE_FEATURES]:
        g = geoms[i]
        k = wkt_kind(g)
        parts = wkt_parts(g, max_pts=60, max_parts=4)
        if not parts:
            continue
        if "LINE" in k:
            for p in parts:
                lines.append({"v": int(volt[i]), "k": kind[i], "p": p})
        else:  # substations, plants, generators: polygon or point -> one marker
            a = np.asarray(parts[0])
            points.append({"v": int(volt[i]), "k": kind[i], "x": round(float(a[:, 0].mean()), 4),
                           "y": round(float(a[:, 1].mean()), 4)})
    vc = pd.Series(volt[np.char.find(kind.astype(str), "line") >= 0]).value_counts()
    return {"lines": lines, "points": points, "n": int(len(df)), "dropped": int(dropped),
            "kinds": {str(k): int(v) for k, v in pd.Series(kind).value_counts().items()},
            "voltages": {str(int(k)): int(v) for k, v in vc.sort_index(ascending=False).items()}}


NODE_TABLES = ("substations", "grid_nodes", "nodes")   # uploaded grid nodes, used when a run has no OSM lines


def _nodes(f, key):
    """Grid nodes from an uploaded substations table: one marker each, with name, type and owner as stored."""
    df = read_fixed_frame(f, key)
    cols = {c.lower(): c for c in map(str, df.columns)}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    lat, lon = pick("latitude", "lat", "y"), pick("longitude", "lon", "lng", "x")
    name, typ, own = pick("node_name", "name"), pick("node_type", "type", "power"), pick("owner", "operator")
    volt = pick("voltage", "voltage_kv", "kv", "volt", "nominal_voltage")
    pts = []
    for i in range(len(df)):
        if lat and lon and np.isfinite(pd.to_numeric(df[lat].iloc[i], errors="coerce")) and np.isfinite(pd.to_numeric(df[lon].iloc[i], errors="coerce")):
            x, y = float(df[lon].iloc[i]), float(df[lat].iloc[i])
        elif "geometry" in df:
            parts = wkt_parts(df["geometry"].iloc[i], max_pts=60, max_parts=1)
            if not parts:
                continue
            a = np.asarray(parts[0]); x, y = float(a[:, 0].mean()), float(a[:, 1].mean())
        else:
            continue
        v = pd.to_numeric(df[volt].iloc[i], errors="coerce") if volt else np.nan
        if np.isfinite(v) and v < 1000:   # a column in kV, stored as volts elsewhere in the dash
            v = v * 1000
        pts.append({"x": round(x, 4), "y": round(y, 4), "name": str(df[name].iloc[i]) if name else str(df.index[i]),
                    "type": str(df[typ].iloc[i]) if typ else "node", "owner": str(df[own].iloc[i]) if own else None,
                    "v": int(v) if np.isfinite(v) else None})
    types = pd.Series([p["type"] for p in pts]).value_counts()
    return {"key": key, "points": pts, "n": int(len(df)), "types": {str(k): int(v) for k, v in types.items()}}


def _clusters(f, tech, adjustments=None):
    df = read_fixed_frame(f, f"clusters/{tech}")
    if adjustments is not None:
        df["potential_capacity_unadjusted"] = df["potential_capacity"]
        df["potential_capacity"] = adjustments[f"clusters/{tech}"]["potential_capacity_adjusted"].to_numpy()
    rc = _region_col(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cols = {}
    for c in numeric:
        v = df[c].to_numpy(dtype=float)
        if c.startswith("lcoe"):
            v = np.where(v >= LCOE_SENTINEL, np.nan, v)
        cols[c] = _num(v)
    return df, {"ids": [str(i) for i in df.index], "region": [str(v) for v in df[rc]] if rc else None,
                "rings": [wkt_parts(g, max_pts=80, max_parts=12) for g in df["geometry"]] if "geometry" in df else None,
                "cols": cols, "numeric": numeric, "n": int(len(df))}


def profile_stats(ts, weights=None):
    """Monthly, diurnal and month x hour means of an hourly frame (columns = sites)."""
    idx = time_index(ts.index)
    month, hour = idx.month.to_numpy(), idx.hour.to_numpy()
    vals = ts.to_numpy(dtype=float)
    if weights is None:
        w = np.ones(vals.shape[1])
    else:
        w = np.nan_to_num(np.asarray(weights, float), nan=0.0)
        if w.sum() <= 0:
            w = np.ones(vals.shape[1])
    fleet = np.nansum(vals * w, axis=1) / w.sum()
    s = pd.Series(fleet)
    mh = s.groupby([month, hour]).mean().unstack()
    return {
        "monthly": _num(s.groupby(month).mean().reindex(range(1, 13)).to_numpy(), 4),
        "diurnal": _num(s.groupby(hour).mean().reindex(range(24)).to_numpy(), 4),
        "mh": [_num(mh.reindex(index=range(1, 13), columns=range(24)).to_numpy()[i], 3) for i in range(12)],
        "mean": _f(float(np.nanmean(fleet)), 4), "hours": int(len(s)),
        "start": str(idx[0]), "end": str(idx[-1]),
        # daily means in calendar order, and the hourly series itself for the day picker
        "daily": _num(s.groupby(idx.normalize()).mean().to_numpy(), 4),
        "day0": str(idx[0].date()), "hourly": _num(fleet, 3),
    }


def _profiles(f, tech, cl_df):
    ts = read_fixed_frame(f, f"timeseries/clusters/{tech}")
    cap = None
    if cl_df is not None and "potential_capacity" in cl_df:
        cap = cl_df["potential_capacity"].reindex([str(c) for c in ts.columns]).to_numpy()
    out = {"fleet": profile_stats(ts, cap), "weighted": cap is not None,
           "ids": [str(c) for c in ts.columns], "site": {}}
    idx = time_index(ts.index)
    month, hour = idx.month.to_numpy(), idx.hour.to_numpy()
    mon = ts.groupby(month).mean().reindex(range(1, 13))
    dia = ts.groupby(hour).mean().reindex(range(24))
    for c in ts.columns:
        out["site"][str(c)] = {"monthly": _num(mon[c].to_numpy(), 4), "diurnal": _num(dia[c].to_numpy(), 4)}
    return out


def _story(cells_df, cl, tech, sentinel):
    """Headline numbers for one technology, from the cells and clusters tables."""
    d = cells_df
    s = {}
    cap_c, cf_c, lcoe_c = f"potential_capacity_{tech}", f"{tech}_CF_mean", f"lcoe_{tech}"
    dev_c, la_c = f"Developable_area_{tech}", f"LandAvailability_ERA5_{tech}"
    if "geom_area_km2" in d:
        s["area_km2"] = _f(d["geom_area_km2"].sum())
    if dev_c in d:
        s["developable_km2"] = _f(d[dev_c].sum())
        if s.get("area_km2"):
            s["developable_share"] = _f(d[dev_c].sum() / d["geom_area_km2"].sum(), 4)
    if cap_c in d:
        cap = d[cap_c].to_numpy(float)
        s["capacity_mw"] = _f(np.nansum(cap))
        s["cells_with_capacity"] = int((cap > 0).sum())
        if cf_c in d:
            s["cf_wmean"] = _f(_wmean(d[cf_c], cap), 4)
            ok = cap > 0
            if ok.any():
                s["cf_min"], s["cf_max"] = _f(d[cf_c][ok].min(), 4), _f(d[cf_c][ok].max(), 4)
        if lcoe_c in d:
            l = d[lcoe_c].to_numpy(float)
            valid = np.isfinite(l) & (l < LCOE_SENTINEL) & (cap > 0)
            s["lcoe_cells_valid"] = int(valid.sum())
            s["lcoe_sentinel"] = int(sentinel.get(lcoe_c, 0))
            if valid.any():
                s["lcoe_min"], s["lcoe_wmean"] = _f(l[valid].min(), 4), _f(_wmean(l[valid], cap[valid]), 4)
                s["capacity_valid_mw"] = _f(cap[valid].sum())
    if "nearest_station_distance_km" in d:
        s["grid_km_mean"] = _f(d["nearest_station_distance_km"].mean(), 4)
        s["grid_km_max"] = _f(d["nearest_station_distance_km"].max(), 4)
    for c in (f"capex_{tech}", f"fom_{tech}", f"grid_connection_cost_per_km_{tech}", f"Operational_life_{tech}"):
        if c in d and len(d):
            s[c.replace(f"_{tech}", "")] = _f(d[c].iloc[0])
    if cl is not None and "potential_capacity" in cl:
        s["clusters"] = int(len(cl))
        s["cluster_capacity_mw"] = _f(cl["potential_capacity"].sum())
        l = cl["lcoe"].to_numpy(float) if "lcoe" in cl else None
        if l is not None and np.isfinite(l).any():
            s["cluster_lcoe_min"], s["cluster_lcoe_max"] = _f(np.nanmin(l), 4), _f(np.nanmax(l[l < LCOE_SENTINEL]) if (l < LCOE_SENTINEL).any() else np.nan, 4)
    return s


def is_landanalytics(key):
    k = key.lower().replace("_", "")
    return "landanalytics" in k


def _landanalytics(f, key, cells_df, cells_out, max_rows=400, max_cols=12):
    """One land analytics table, read without assuming a schema.

    Numeric columns are passed through with a label per row (first text column, else
    the index). If the rows are grid cells, the numeric columns are also attached to
    the cells payload so the map and layer stack can colour by them.
    """
    g = f[key]
    rows, ncols = int(g["axis1"].shape[0]), int(g["axis0"].shape[0])
    out = {"key": key, "rows": rows, "cols": ncols}
    if rows * ncols > 2_000_000:
        out["note"] = "Too large to summarise in the browser. Browse it in the Data tab."
        return out
    df = read_fixed_frame(f, key)
    df.columns = [str(c) for c in df.columns]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text = [c for c in df.columns if c not in numeric and c.lower() not in ("geometry", "centroid")]
    out["columns"] = list(df.columns)[:60]
    out["numeric"] = numeric[:max_cols]
    out["label_col"] = text[0] if text else "index"
    labels = df[text[0]].astype(str) if text else pd.Series([str(i) for i in df.index], index=df.index)
    out["totals"] = {c: _f(np.nansum(df[c].to_numpy(float))) for c in out["numeric"]}
    head = df.iloc[:max_rows]
    out["labels"] = labels.iloc[:max_rows].tolist()
    out["values"] = {c: _num(head[c].to_numpy(float)) for c in out["numeric"]}
    out["shown_rows"] = int(len(head))
    out["per_cell"] = False
    if cells_df is not None and len(df):
        idx = pd.Index([str(i) for i in df.index])
        hit = idx.isin(cells_df.index.astype(str)).mean()
        if hit >= 0.9 and idx.is_unique:
            out["per_cell"] = True
            al = df.set_axis(idx).reindex(cells_df.index.astype(str))
            short = key.rsplit("/", 1)[-1]
            out["cell_cols"] = []
            for c in out["numeric"]:
                name = f"{short}:{c}"
                cells_out["cols"][name] = _num(al[c].to_numpy(float))
                cells_out["numeric"].append(name)
                out["cell_cols"].append(name)
    return out


# ------------------------------------------------------------------ land analytics images
# RESource's HDF5 handler stores rendered figures as raw uint8 datasets under
# LandAnalytics/images (attr content_type), drawn with the class legends in
# RESource/assets/legends. The dash shows them as stored and does not redraw them.
# Rasters under LandAnalytics/rasters are listed only, so nothing in the store is hidden.

LA_IMAGE_MAX_BYTES = 8_000_000  # larger figures are listed but not embedded


def _la_attrs(obj):
    """Dataset attributes as JSON-safe values; JSON text is decoded, nothing renamed."""
    out = {}
    for k, v in obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, str) and v[:1] in ("{", "["):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[str(k)] = v
    return out


def _la_media(f, frame_keys):
    """Embedded images, plus a listing of every other non-table land analytics dataset."""
    images, other = [], []
    inside_frame = tuple(k + "/" for k in frame_keys)
    for top in [k for k in f.keys() if is_landanalytics(k)]:
        def visit(name, obj, top=top):
            full = f"{top}/{name}"
            if not isinstance(obj, h5py.Dataset) or full.startswith(inside_frame):
                return
            attrs = _la_attrs(obj)
            if obj.ndim == 1 and obj.dtype == np.uint8 and ("content_type" in attrs or "image" in full.lower()):
                ct, n = attrs.pop("content_type", None) or "image/png", int(obj.shape[0])
                im = {"key": full, "name": full.rsplit("/", 1)[-1], "title": attrs.pop("title", None) or full.rsplit("/", 1)[-1],
                      "content_type": ct, "bytes": n, "attrs": attrs}
                if n > LA_IMAGE_MAX_BYTES:
                    im["note"] = f"{n / 1e6:.1f} MB figure. Load it on demand to display it."
                else:
                    im["uri"] = f"data:{ct};base64," + base64.b64encode(obj[()].tobytes()).decode("ascii")
                images.append(im)
            else:
                other.append({"key": full, "shape": list(obj.shape), "dtype": str(obj.dtype), "units": attrs.get("units")})
        f[top].visititems(visit)
    return {"images": images, "other": other}


def land_image(path, key):
    """Read one land analytics image on demand as a JSON-encoded data URI."""
    if not is_landanalytics(key.split("/", 1)[0]):
        raise ValueError(f"Not a land analytics dataset: {key}")
    with h5py.File(path, "r") as store:
        dataset = store[key]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1 or dataset.dtype != np.uint8:
            raise ValueError(f"Not an image dataset: {key}")
        attrs = _la_attrs(dataset)
        if "content_type" not in attrs and "image" not in key.lower():
            raise ValueError(f"Not an image dataset: {key}")
        content_type = attrs.get("content_type") or "image/png"
        data = base64.b64encode(dataset[()].tobytes()).decode("ascii")
    return json.dumps({"uri": f"data:{content_type};base64,{data}"})


def _frame(block, extra=None):
    """Rebuild a DataFrame from a payload block ({ids, cols, region})."""
    df = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in block["cols"].items()})
    df.index = block["ids"]
    return df


def combine_summaries(payloads, label):
    """Merge the payloads of several regional stores into one country-level payload.

    ``payloads`` is a list of JSON strings from summarize_store. Cells, clusters,
    boundaries and grid features are concatenated; headline numbers are recomputed
    from the merged tables; fleet profiles are weighted by each store's clustered
    capacity. The grouping column becomes the store region.
    """
    P = [json.loads(p) if isinstance(p, str) else p for p in payloads]
    m0 = P[0]["meta"]
    out = {"meta": {"file": label, "country": m0.get("country"), "region": label, "scenario": m0.get("scenario"),
                    "year": m0.get("year"), "run_date": None},
           "combined": True, "sources": [p["meta"] for p in P], "keys": P[0]["keys"], "problems": {}, "story": {},
           "missing": sorted({k for p in P for k in p.get("missing", [])}), "units": P[0].get("units"),
           "cost": P[0].get("cost", {}), "clusters": {}, "profiles": {}, "landanalytics": [],
           "la_media": {"images": [], "other": []}}
    for i, p in enumerate(P):
        for k, v in p.get("problems", {}).items():
            out["problems"][f"{p['meta'].get('region')}:{k}"] = v
        for la in p.get("landanalytics", []):
            out["landanalytics"].append({**la, "source": p["meta"].get("region")})
        for kind in ("images", "other"):
            for x in (p.get("la_media") or {}).get(kind, []):
                out["la_media"][kind].append({**x, "source": p["meta"].get("region"), "source_index": i})

    def cat_block(blocks, regions, src):
        cols = sorted({c for b in blocks for c in b["cols"]}, key=lambda c: min(
            (b["numeric"].index(c) if c in b["numeric"] else 10**6) for b in blocks))
        o = {"ids": [], "region": [], "subregion": [], "src": [], "rings": [], "cols": {c: [] for c in cols},
             "numeric": cols, "region_col": "Region"}
        for b, r, si in zip(blocks, regions, src):
            n = len(b["ids"])
            o["ids"] += b["ids"]; o["region"] += [r] * n; o["src"] += [si] * n
            o["subregion"] += b.get("region") or [None] * n
            o["rings"] += b.get("rings") or [[] for _ in range(n)]
            for c in cols:
                o["cols"][c] += b["cols"].get(c) or [None] * n
        o["n"] = len(o["ids"])
        return o

    have = [(i, p) for i, p in enumerate(P) if p.get("cells")]
    cells_df = None
    if have:
        out["cells"] = cat_block([p["cells"] for _, p in have], [p["meta"].get("region") for _, p in have], [i for i, _ in have])
        out["cells"]["sentinel"] = {}
        for _, p in have:
            for k, v in p["cells"].get("sentinel", {}).items():
                out["cells"]["sentinel"][k] = out["cells"]["sentinel"].get(k, 0) + v
        cells_df = _frame(out["cells"])
    bs = [p["boundary"] for p in P if p.get("boundary")]
    out["regions"] = [{"name": p["meta"].get("region"), "rings": p["boundary"].get("outline")} for p in P if p.get("boundary") and p["boundary"].get("outline")]
    if bs:
        out["boundary"] = {"name_col": "Region", "names": [n for b in bs for n in b["names"]],
                           "rings": [r for b in bs for r in b["rings"]], "n": sum(b["n"] for b in bs), "note": bs[0]["note"]}
    ls = [p["lines"] for p in P if p.get("lines")]
    if ls:
        kinds, volts = {}, {}
        for l in ls:
            for k, v in l["kinds"].items():
                kinds[k] = kinds.get(k, 0) + v
            for k, v in l["voltages"].items():
                volts[k] = volts.get(k, 0) + v
        out["lines"] = {"lines": [x for l in ls for x in l["lines"]], "points": [x for l in ls for x in l["points"]],
                        "n": sum(l["n"] for l in ls), "dropped": sum(l["dropped"] for l in ls), "kinds": kinds, "voltages": volts}
    ns = [p["nodes"] for p in P if p.get("nodes")]
    if ns:
        types = {}
        for nd in ns:
            for k, v in nd["types"].items():
                types[k] = types.get(k, 0) + v
        out["nodes"] = {"key": ns[0]["key"], "points": [x for nd in ns for x in nd["points"]], "n": sum(nd["n"] for nd in ns), "types": types}
    for t in TECHS:
        hv = [(i, p) for i, p in enumerate(P) if p.get("clusters", {}).get(t)]
        cl_df = None
        if hv:
            out["clusters"][t] = cat_block([p["clusters"][t] for _, p in hv], [p["meta"].get("region") for _, p in hv], [i for i, _ in hv])
            cl_df = _frame(out["clusters"][t])
        pr = [(p["profiles"][t], sum(v or 0 for v in p["clusters"][t]["cols"].get("potential_capacity", [])) if p.get("clusters", {}).get(t) else 0.0)
              for p in P if p.get("profiles", {}).get(t)]
        if pr:
            w = np.array([x[1] for x in pr], float)
            weighted = bool((w > 0).all())
            if not weighted:
                w = np.ones(len(pr))

            def wavg(get):
                a = np.array([[np.nan if v is None else v for v in get(x[0]["fleet"])] for x in pr], float)
                return _num(np.nansum(a * w[:, None], axis=0) / w.sum(), 4)

            fleet = {"monthly": wavg(lambda fl: fl["monthly"]), "diurnal": wavg(lambda fl: fl["diurnal"]),
                     "mh": [wavg(lambda fl, r=r: fl["mh"][r]) for r in range(12)],
                     "mean": _f(float(np.sum([x[0]["fleet"]["mean"] * wi for x, wi in zip(pr, w)]) / w.sum()), 4),
                     "hours": pr[0][0]["fleet"]["hours"], "start": pr[0][0]["fleet"]["start"], "end": pr[0][0]["fleet"]["end"],
                     "starts": sorted({x[0]["fleet"]["start"] for x in pr})}
            for key in ("daily", "hourly"):   # only when every region covers the same hours
                if all(len(x[0]["fleet"].get(key) or []) == len(pr[0][0]["fleet"].get(key) or []) > 0 for x in pr) and len(fleet["starts"]) == 1:
                    fleet[key] = wavg(lambda fl, key=key: fl[key])
            fleet["day0"] = pr[0][0]["fleet"].get("day0")
            site = {}
            for x in pr:
                site.update(x[0]["site"])
            out["profiles"][t] = {"fleet": fleet, "weighted": weighted, "ids": [i for x in pr for i in x[0]["ids"]], "site": site}
        if cells_df is not None:
            out["story"][t] = _story(cells_df, cl_df, t, out["cells"]["sentinel"]) or {}
    return json.dumps(out)


_H5_INTERNAL = {"CLASS", "VERSION", "TITLE", "PYTABLES_FORMAT_VERSION"}


def _run_meta(f):
    """Attributes of the ``_meta`` group: the run record that newer RESource runs write.

    Returned as stored, one entry per attribute (run_id, runtime, config,
    config_provenance, ...). Values that hold JSON text are decoded; nothing is
    renamed or filled in. Stores written before ``_meta`` existed have no such group.
    """
    out = {}
    for key, v in f["_meta"].attrs.items():
        if key in _H5_INTERNAL:
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, str) and v[:1] in ("{", "["):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[str(key)] = v
    return out


def _capacity_adjustments(f, run):
    """Validate separately stored adjusted capacities against their source tables."""
    factors = run.get("capacity_haircut_applied")
    if factors is None:
        return None
    if not isinstance(factors, dict) or set(factors) != set(TECHS):
        raise ValueError("Invalid applied haircut factors.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in factors.values()):
        raise ValueError("Invalid applied haircut factors.")
    out = {}
    for key, columns in (
        ("cells", {tech: f"potential_capacity_{tech}" for tech in TECHS}),
        *((f"clusters/{tech}", {tech: "potential_capacity"}) for tech in TECHS),
    ):
        source = read_fixed_frame(f, key)
        adjusted = read_fixed_frame(f, f"capacity_haircuts/{key}")
        if not source.index.equals(adjusted.index) or not source.index.is_unique:
            raise ValueError(f"{key}: adjusted capacity indexes do not match.")
        for tech, col in columns.items():
            raw = source[col].to_numpy(dtype=float)
            saved_raw = adjusted[f"{col}_original"].to_numpy(dtype=float)
            saved_adjusted = adjusted[f"{col}_adjusted"].to_numpy(dtype=float)
            if not np.allclose(raw, saved_raw, equal_nan=True) or not np.allclose(saved_adjusted, raw * factors[tech], equal_nan=True):
                raise ValueError(f"{key}: adjusted capacity does not match the source and factor.")
        out[key] = adjusted
    return out


# ------------------------------------------------------------------ entry points

def summarize_store(path, filename=None):
    """Full payload for one store, as a JSON string. Errors come back as {"error": ...}."""
    meta = parse_store_name(filename or path.rsplit("/", 1)[-1])
    try:
        f = open_store(path)
    except StoreError as exc:
        return json.dumps({"meta": meta, "error": {"code": exc.code, "message": exc.message, **exc.extra}})
    out = {"meta": meta, "keys": [], "missing": [], "problems": {}, "story": {}}
    with f:
        out["keys"] = list_frame_keys(f)
        have = {k["key"] for k in out["keys"]}
        for k in out["keys"]:
            if k["cols"] <= 60:
                k["columns"] = [str(c) for c in frame_columns(f, k["key"])]

        def block(name, fn, *a):
            try:
                return fn(*a)
            except Exception as exc:  # one bad table must not hide the rest
                out["problems"][name] = f"{type(exc).__name__}: {exc}"
                return None

        out["run"] = block("_meta", _run_meta, f) if "_meta" in f else None
        adjustments = block("capacity haircuts", _capacity_adjustments, f, out["run"] or {})
        out["haircut_status"] = (
            "adjusted" if adjustments is not None else
            "invalid" if (out["run"] or {}).get("capacity_haircut_applied") is not None else
            "unadjusted"
        )
        cells_df = None
        if "boundary" in have:
            out["boundary"] = block("boundary", _boundary, f)
        if "cells" in have:
            r = block("cells", _cells, f, adjustments)
            if r:
                cells_df, out["cells"] = r
        if "lines" in have:
            out["lines"] = block("lines", _lines, f)
        nk = next((k for k in NODE_TABLES if k in have), None)
        if nk:
            out["nodes"] = block(nk, _nodes, f, nk)
        out["clusters"], out["profiles"], out["cost"] = {}, {}, {}
        for t in TECHS:
            cl_df = None
            if f"clusters/{t}" in have:
                r = block(f"clusters/{t}", _clusters, f, t, adjustments)
                if r:
                    cl_df, out["clusters"][t] = r
            if f"timeseries/clusters/{t}" in have:
                p = block(f"timeseries/clusters/{t}", _profiles, f, t, cl_df)
                if p:
                    out["profiles"][t] = p
            if f"cost/atb/{t}" in have:
                c = block(f"cost/atb/{t}", read_fixed_frame, f, f"cost/atb/{t}")
                if c is not None:
                    out["cost"][t] = json.loads(c.astype(object).where(c.notna(), None).to_json(orient="records"))
            if cells_df is not None:
                out["story"][t] = block(f"story/{t}", _story, cells_df, cl_df, t, out["cells"]["sentinel"]) or {}
        out["landanalytics"] = []
        for k in sorted(have):
            if is_landanalytics(k):
                la = block(k, _landanalytics, f, k, cells_df, out.get("cells"))
                if la:
                    out["landanalytics"].append(la)
        out["la_media"] = block("LandAnalytics images", _la_media, f, have) or {"images": [], "other": []}
        if "units" in have:
            u = block("units", read_fixed_frame, f, "units")
            if u is not None:
                out["units"] = {str(k): str(v) for k, v in u.iloc[:, 0].items()}
        expected = ["boundary", "cells", "lines", "units"] + [f"{p}/{t}" for t in TECHS for p in (
            "clusters", "timeseries/clusters", "timeseries", "cost/atb")]
        out["missing"] = [k for k in expected if k not in have]
    return json.dumps(out, default=str)


def cell_profile(path, tech, cell_id):
    """Monthly / diurnal / month x hour profile of one grid cell, read lazily."""
    with open_store(path) as f:
        ts = read_fixed_frame(f, f"timeseries/{tech}", columns=[cell_id])
    if cell_id not in ts.columns:
        return json.dumps({"error": f"{cell_id} not in timeseries/{tech}"})
    return json.dumps(profile_stats(ts[[cell_id]]))


def cluster_compare(path, tech, cluster_id, max_members=40):
    """Pair one cluster's member cells against its representative profile and LCOE.

    The "before" side is each member cell's own diurnal CF profile and LCOE
    (read lazily, so this stays cheap even for a cluster with hundreds of
    members); the "after" side is the cluster's own representative profile
    (from timeseries/clusters/{tech}) and LCOE (from clusters/{tech}). Lets a
    user see whether a cluster is actually grouping similar resources, or
    papering over a wide spread.

    Requires clusters/{tech}, dissolved_indices/{tech}, timeseries/{tech},
    timeseries/clusters/{tech} and cells to all be present (true once a region
    has run Step 6 clustering for that tech). Returns
    {"available": False, "missing": [...]} otherwise, rather than raising, so
    the dash can skip this cleanly.
    """
    needed = (f"clusters/{tech}", f"dissolved_indices/{tech}", f"timeseries/{tech}",
              f"timeseries/clusters/{tech}", "cells")
    with open_store(path) as f:
        have = {k["key"] for k in list_frame_keys(f)}
        missing = [k for k in needed if k not in have]
        if missing:
            return json.dumps({"available": False, "missing": missing})

        cl_df = read_fixed_frame(f, f"clusters/{tech}")
        cl_ids = [str(i) for i in cl_df.index]
        if cluster_id not in cl_ids:
            return json.dumps({"available": False, "missing": [f"cluster {cluster_id} in clusters/{tech}"]})
        row = cl_df.iloc[cl_ids.index(cluster_id)]
        rc = _region_col(cl_df)
        region, cluster_no = (row[rc] if rc else None), row.get("Cluster_No")

        dis = read_fixed_frame(f, f"dissolved_indices/{tech}")
        col = cluster_no if cluster_no in dis.columns else str(cluster_no)
        member_ids = None
        if region in dis.index and col in dis.columns:
            cell = dis.loc[region, col]
            if isinstance(cell, (list, tuple, np.ndarray)):
                member_ids = [str(c) for c in cell]
        if not member_ids:
            return json.dumps({"available": False, "missing": ["member cells for this cluster"]})
        member_ids = member_ids[:max_members]

        cell_ts = read_fixed_frame(f, f"timeseries/{tech}", columns=member_ids)
        hour = time_index(cell_ts.index).hour.to_numpy()
        member_diurnal = [
            _num(cell_ts[c].groupby(hour).mean().reindex(range(24)).to_numpy(), 4)
            for c in cell_ts.columns if c in member_ids
        ]

        rep_diurnal = None
        cl_ts = read_fixed_frame(f, f"timeseries/clusters/{tech}", columns=[cluster_id])
        if cluster_id in cl_ts.columns:
            rep_hour = time_index(cl_ts.index).hour.to_numpy()
            rep_diurnal = _num(cl_ts[cluster_id].groupby(rep_hour).mean().reindex(range(24)).to_numpy(), 4)

        lcoe_col = f"lcoe_{tech}"
        member_lcoe = None
        cells_df = read_fixed_frame(f, "cells", columns=[lcoe_col])
        if lcoe_col in cells_df.columns:
            v = cells_df.loc[cells_df.index.astype(str).isin(member_ids), lcoe_col].to_numpy(float)
            member_lcoe = _num(np.where(v >= LCOE_SENTINEL, np.nan, v), 4)

        # clusters/{tech} strips the "_{tech}" suffix (it's already tech-scoped), unlike cells.
        cluster_lcoe = row.get("lcoe")
        cluster_lcoe = None if cluster_lcoe is None or cluster_lcoe >= LCOE_SENTINEL else _f(cluster_lcoe, 4)
        cluster_capacity = _f(row.get("potential_capacity"), 4)

    return json.dumps({
        "available": True, "cluster_id": cluster_id, "region": None if region is None else str(region),
        "cluster_no": None if cluster_no is None else int(cluster_no), "member_ids": member_ids,
        "member_diurnal": member_diurnal, "rep_diurnal": rep_diurnal,
        "member_lcoe": member_lcoe, "cluster_lcoe": cluster_lcoe, "cluster_capacity": cluster_capacity,
    })


def table_preview(path, key, n=200):
    with open_store(path) as f:
        g = f[key]
        rows, ncols = int(g["axis1"].shape[0]), int(g["axis0"].shape[0])
        cols = frame_columns(f, key)
        keep = cols[:40]
        df = read_fixed_frame(f, key, columns=keep, rows=slice(0, n))
    if "timeseries" in key:
        df.index = time_index(df.index).astype(str)
    body = [[str(i)] + ["" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)[:120]) for v in row]
            for i, row in zip(df.index, df.astype(object).where(df.notna(), None).values.tolist())]
    return json.dumps({"key": key, "rows": rows, "cols": ncols, "columns": ["index"] + [str(c) for c in df.columns],
                       "body": body, "shown_rows": len(body), "shown_cols": len(df.columns)})


def table_csv(path, key, max_values=3_000_000):
    """Whole table as CSV text. Refuses very large tables instead of freezing the browser."""
    with open_store(path) as f:
        g = f[key]
        size = int(g["axis1"].shape[0]) * int(g["axis0"].shape[0])
        if size > max_values:
            return json.dumps({"error": f"{key} holds {size:,} values. Too large to export in the browser; "
                                        "read it with pandas.read_hdf instead."})
        df = read_fixed_frame(f, key)
    if "timeseries" in key:
        df.index = time_index(df.index)
    return json.dumps({"csv": df.to_csv()})
