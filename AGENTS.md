# RESource agent agreement

This file is the operating agreement for AI coding agents working in this
repository. Read `README.md`, `CONTRIBUTING.md`, and the relevant configuration
before changing code. Treat scientific assumptions and regional data provenance as
part of the public API.

## Product identity and commands

- Distribution: `deltae-resource`
- Import package: `RESource`
- CLI: `resource`
- Layout: `src/RESource/`
- Environment and task runner: `uv`
- Canonical development setup: `uv sync --locked --all-extras`
- Canonical validation: `uv run pre-commit run --all-files` and `uv run pytest`

Do not introduce Makefile instructions, a second environment manager, or new
imports from the deprecated `RES` namespace.

## Safe working agreement

1. Inspect the working tree and preserve unrelated user changes.
2. Never delete source data, results, configurations, or notebooks merely because
   they appear unused. Establish provenance and obtain explicit authorization.
3. Do not execute a full regional workflow without confirming expected downloads,
   storage, credentials, runtime, and output location.
4. Never invent a data source, CRS, unit, administrative name, exclusion rule,
   capacity density, or policy threshold. Record uncertainty and request evidence.
5. Keep external credentials out of source, notebooks, logs, and configuration.
6. Put reusable behavior in `src/RESource`, tests in `tests`, notebooks only in
   `notebooks`, and generated outputs in ignored data/result locations.
7. Preserve third-party notices. Do not change the repository license without an
   explicit maintainer decision and completed provenance review.

## Using RESource for another region

Treat regional adaptation as a staged, auditable workflow:

1. **Define scope.** Record country and region identifiers, administrative level,
   wind/solar technologies, weather year, scenarios, required outputs, and decision
   question.
2. **Audit inputs.** For every dataset record provider, URL/version, retrieval date,
   license, CRS, units, resolution, coverage, and preprocessing. Verify the exact
   administrative names against the selected boundary dataset.
3. **Choose spatial conventions.** Select and justify a projected analysis CRS;
   validate bounds, geometry, area calculations, raster alignment, nodata handling,
   and timezone.
4. **Create configuration.** Copy the closest current file in `config/`, give it a
   descriptive regional/scenario name, and change only understood keys. Do not use
   an archived configuration as an unquestioned template.
5. **Run a smoke region.** Start with one small administrative unit and one resource
   type. Use `uv run resource CONFIG --year YEAR -r REGION`; inspect downloads,
   intermediate layers, logs, and output schema before scaling.
6. **Validate layers.** Map boundaries, exclusions, land availability, weather,
   grid proximity, capacity factors, capacities, scores, and clusters. Check units,
   plausible ranges, missingness, and edge effects against an independent source.
7. **Scale deliberately.** Only after the smoke run passes, expand to remaining
   regions/resources. Keep output directories scenario-specific and never overwrite
   comparison evidence silently.
8. **Document and test.** Add deterministic tests for new parsing or transformations,
   a regional workflow notebook only when it adds narrative value, and documentation
   for limitations and non-transferable assumptions.

## Definition of done for a regional skill

A regional adaptation is not complete until configuration validation passes, one
end-to-end smoke case succeeds, data provenance and licenses are documented, maps
and numerical ranges are reviewed, results can be recreated from a clean locked
checkout, and the limitations are explicit. Passing code tests alone is insufficient.

When asked to expose this workflow as an external agent skill, use this file as the
domain contract: collect the scope and data manifest first, produce a configuration
and validation report, run only authorized stages, and return paths to evidence.

