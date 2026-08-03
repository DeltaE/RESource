# Development pipeline

This roadmap describes active development directions. It is not a promise that the
features are available in the current release. Experimental outputs must be labeled
and kept separate from validated baseline results.

## 1. Tenure lands to MW workflow — in progress

**Objective:** connect land-tenure information to technically feasible capacity so
that suitable land can be summarized in megawatts while retaining tenure context.

Planned pipeline:

1. Ingest and validate tenure polygons, identifiers, rights categories, dates,
   source license, and jurisdiction-specific definitions.
2. Harmonize CRS and geometry, resolve overlaps explicitly, and intersect tenure
   with otherwise feasible wind and solar land.
3. Calculate eligible area by technology and apply documented capacity-density and
   setback assumptions to estimate MW.
4. Preserve both gross and net area, excluded area by reason, conversion assumptions,
   and uncertainty in machine-readable outputs.
5. Validate aggregate areas and sampled parcels against authoritative records.

The MW estimate is a screening result, not proof of land access, development rights,
interconnection, consent, or project feasibility.

## 2. ERA6 preparation for higher-resolution cutout cells — planned research

**Objective:** prepare the climate-data pipeline for a future ERA6 product and
higher spatial resolution than current ERA5-based cutouts.

Because an operational product contract is not yet encoded in RESource, development
must isolate the dataset adapter from the assessment logic. Work includes defining a
versioned provider interface; preserving raw metadata; validating coordinates,
calendars, units, accumulation periods, and timezone; making cutout resolution an
explicit configuration value; and benchmarking storage, memory, download volume,
resampling, and capacity-factor effects against ERA5.

Higher nominal resolution must not be described as higher accuracy until comparisons
against observations and sensitivity tests support that conclusion.

## 3. Consultation flags in result metrics — planned

**Objective:** add informational flags that help users recognize where consultation
or further due diligence may be required before implementation decisions.

The result schema should use transparent source-specific fields rather than a single
opaque score. Candidate fields include flag type, source layer and version, feature
identifier, intersection area or distance, threshold, jurisdiction, confidence,
retrieval date, and a human-readable explanation. Multiple flags must be retained;
absence of a flag must not be interpreted as consultation clearance.

Flags are decision-support information only. They must not determine legal duties,
replace engagement with Indigenous peoples, rights holders, communities, regulators,
or landholders, or assert consent. Terminology and thresholds require jurisdictional
and affected-party review before release.

## Cross-cutting acceptance criteria

Each pipeline addition requires documented provenance and licensing, configuration
schema changes, deterministic unit tests, a small regional fixture, uncertainty and
limitations, backward-compatible result migration where practical, and comparison
against the current baseline before becoming a stable public feature.

