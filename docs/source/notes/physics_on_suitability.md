# Physics-Based Tree Height Suitability Mapping

## 1. Purpose
This document explains the physical rationale and raster processing steps used
to derive tree-height-based suitability masks for solar and wind siting.
The approach uses SCANFI canopy height data to quantify shading and turbulence
constraints caused by tall vegetation, producing binary maps for suitability assessment.

## 2. Physical Basis
Vegetation height influences solar shading and aerodynamic turbulence.
These effects can be simplified using geometric and empirical relationships
between canopy height (H), horizontal distance (d), and obstruction or turbulence angles.

### 2.1 Solar Shading Geometry
For solar systems, the obstruction angle θ between the panel plane and the tree top
is given by:

θ = arctan(H / d)

To ensure that shading remains below a tolerable limit θₘₐₓ (typically 5–10°),
the minimum clearance distance must satisfy:

- d ≥ H / tan(θₘₐₓ)

Example: For a 15 m tree and θₘₐₓ = 10°, d ≥ 85 m.
This ensures minimal annual energy loss from obstruction at low solar elevation angles.

### 2.2 Wind Turbulence Effects
For wind turbines, the impact of forest roughness extends approximately 8–10 times
the tree height downwind of the canopy edge.
Therefore, exclusion zones of radius ≥10H are used to maintain low turbulence
intensity and reliable inflow conditions.

## 3. Raster Processing Workflow
1. Mask no-data values from the SCANFI raster.
2. Classify pixels into tall forest (H ≥ H_forest_threshold) and low canopy (H ≤ H_inside_threshold).
3. Compute the Euclidean distance (m) to the nearest tall-forest pixel using a distance transform.
4. Apply the physical clearance condition: pixels closer than the safe distance are excluded.
5. For solar: d ≥ H / tan(θₘₐₓ); for wind: d ≥ 10H.
6. Combine distance and height conditions to form a binary suitability raster (1 = suitable, 0 = excluded).
7. Clip the raster to the boundary polygon and export as GeoTIFFs for further use in energy siting models.

## 4. Interpretation of Outputs
The process generates two rasters:

- **forestmask.tif** — pixels representing tall forest exceeding H_forest_threshold.
- **suitability.tif** — binary map where value 1 indicates acceptable vegetation height and distance from tall forest.

These outputs can be integrated into **RESource**, **PyPSA**, or **CLEWs-based** workflows
as siting or cost constraint layers.

## 5. Example Parameters

| Parameter | Typical Value | Physical Meaning |
|------------|----------------|------------------|
| θₘₐₓ | 10° | Max solar obstruction angle |
| H_inside_threshold | 2 m | Max allowable canopy height inside solar site |
| H_forest_threshold | 15–20 m | Defines tall forest zone |
| Safe distance (solar) | ≈85 m | For 15 m trees at 10° obstruction |
| Safe distance (wind) | ≈200 m | For 20 m trees using 10H rule |

These parameters can be regionally calibrated based on solar elevation angles,
typical vegetation structure, and technology-specific design constraints
(e.g., tracker tilt or turbine hub height).
"""
