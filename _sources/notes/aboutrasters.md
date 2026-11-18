- Is 100m Resolution is Optimal for Land Suitability Analysis
---

Quantitative Analysis at BC Latitudes (~50°N)
   > At approximately 50° North latitude (typical for British Columbia), a single 0.25° grid cell covers:
    - **North-South**: 0.25° ≈ **27.8 km**
    - **East-West**: 0.25° × cos(50°) ≈ **17.9 km**
    - **Total Area**: ≈ **496 km²**

### Pixel Density Comparison

| Resolution | Pixels per Cell | Total Pixels |
|------------|----------------|--------------|
| **100m** | 496 km² ÷ (0.1 km)² | **≈49,600 pixels** |
| **30m** | 496 km² ÷ (0.03 km)² | **≈552,000 pixels** |

### Why 100m is Sufficient

A binary developable land mask averaged to 0.25° resolution is based on **tens of thousands** of 100m pixels per cell. This provides:

- ✅ **Ample statistical sampling** for fraction estimation
- ✅ **Very low sampling/edge error**
- ✅ **Reliable threshold decisions**

The 30m resolution provides ~11× more pixels but:
- ❌ **Rarely changes** the cell-mean fraction significantly
- ❌ **Increases I/O and memory** requirements by ~11×
- ❌ **Minimal impact** on final suitability decisions

### When 30m Resolution Might Be Beneficial

30m resolution should only be considered when:

1. **Small buffer analysis**: Working with very narrow features (≤150m wide) where precise area calculations matter
2. **Micro-siting applications**: Planning exact turbine/panel placement rather than cell-level exclusions
3. **High-precision requirements**: Detailed local assessments requiring maximum spatial accuracy

### Recommended Workflow

For efficient and robust land suitability analysis:

1. **Vector Processing**: Keep buffers on vectors in projected CRS (meters)
2. **Rasterization**: Convert to 100m binary mask (1=allowed, 0=excluded)
3. **Aggregation**: Average binary mask to cutout resolution (yields fraction allowed)
4. **Threshold Application**: Apply minimum buildable-area threshold (e.g., ≥10-30% allowed)
5. **Integration**: Feed exclusion layer to Atlite for capacity calculations

> **Conclusion**: 100m resolution strikes the optimal balance between computational efficiency and analytical precision for regional renewable energy assessments.

__What resolution to use (while buffers are in flux)__

Rule of thumb: choose pixel size 𝑝  so that
p ≤ ½ × (smallest buffer distance)

This ensures narrow exclusions don’t disappear.

Quick calls:
- If your smallest buffer ≥ 200 m → 100 m raster is sufficient.
- If your smallest buffer 100–150 m → consider 50–75 m (or stick to 100 m and validate on a pilot tile).
- If your smallest buffer ≤ 60–80 m → 30 m pays off.