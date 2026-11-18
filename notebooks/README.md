# RESource notebooks

All repository notebooks live under this directory. Start Jupyter from the
repository root so configuration and data paths resolve consistently:

```bash
uv sync --locked --extra notebooks
uv run jupyter lab notebooks/
```

New notebooks must import the installed package as `RESource`. Do not modify
`sys.path`, change the process working directory, or install packages from a
notebook cell.

## Recommended starting points

The maintained end-to-end regional workflows are:

- `workflows/resources_playground_BC.ipynb`
- `workflows/resources_playground_CAN.ipynb`
- `workflows/resources_playground_WB6.ipynb`
- `workflows/resources_playground_BGD.ipynb`

These notebooks demonstrate orchestration and analysis. Reusable implementation
belongs in `src/RESource`, not in notebook cells.

## Organization

- `workflows/`: primary end-to-end examples.
- `case_studies/`: case-specific analyses and external-result validation.
- `validation/`: data and output validation notebooks.
- `case_playground/` and `WB6/`: exploratory regional analysis retained for
  research traceability.
- `Publication_resources_EGY360/`: notebooks supporting publication figures and
  sensitivity analysis.
- `NREL_ATB_notebooks/`: technology-cost data preparation.
- `enhancements/`: experimental features not part of the stable workflow.
- notebooks directly in this directory: exploratory visualization and data
  inspection retained because they are not exact duplicates of the workflows.

## Maintenance policy

A notebook may be removed when it is empty, an exact or source-equivalent copy,
explicitly marked old, replaced by package code, or dependent on APIs that no
longer exist. Historical work that remains scientifically relevant should move
into a clearly named case-study or publication folder instead of being copied
into documentation directories.

Automated tests verify that notebooks are valid JSON and that none are stored
outside this directory. Full notebook execution is intentionally separate because
many analyses require large external datasets and service credentials.

## WB6 combined analytics

In the `studies_WB6` notebooks, select the same scenario, weather year and
datestamp for all regional stores. The cell and land notebooks read regional
stores and do not create a combined store. The cell notebook uses stores under
`results/<country>/<region>/<run_tag>/`; the land notebook also checks the
legacy `data/store/` location if a results store is missing. The population notebook produces its
map without a combined store. They write each run to
`results/Studies/results_Publication/<cells|lands|population>/<timestamp>/`.
The run's `figures/` folder holds images and `images.md` describes them.

In the cell notebook, capacity haircuts are read from each regional store's
metadata. For stores without that metadata, it uses the previous WB6 plotting
values (solar 0.8, wind 0.6). It records the factors, source, and explanatory
notes in those same regional stores. It also writes separate adjusted cell and
cluster capacities under `capacity_haircuts/`, preserving the original tables.
Regional factors must agree for a combined plot. Solar 0.8 retains 80% of
capacity (a 20% reduction); wind 0.6 retains 60% (a 40% reduction). LCOE and
capacity-factor profiles are not recalculated.

The dashboard uses adjusted capacities when these tables validate against the
originals and reports the reductions. Its Data tab previews and exports original
tables as stored; the adjusted tables can be selected separately. If source
tables later change, the dashboard warns and shows original capacities instead.
Rebuild the dashboard after code changes. Existing stores are updated when the
cell notebook's store-loading cell runs.

## Publishing notebook studies

The dashboard's **Studies** tab displays PNG, JPEG, GIF and WebP images anywhere
under `Studies`. From the Stores tab, open `results/`, `Studies/`, or another
parent folder; browsers require this selection for access. Studies need no
store or Pyodide. Images without a description are still shown.

For another notebook, prepare an `images.md` template with an optional introductory
paragraph and one `## filename.png` heading per image, followed by its description as plain text. Headings must match
the image filenames exactly. Keep scientific assumptions, units and provenance
accurate; mark unknowns explicitly. Markdown is displayed as plain text.
Use `create_study_output` to create a fresh, non-overwriting run:

```python
from RESource.study import create_study_output

run = create_study_output(ROOT, "my_study", "my_notebook", ROOT / "notebooks/my_images.md")
figure_dir = run / "figures"
# Save figures into figure_dir.
```

The dashboard associates each image with the closest ancestor `images.md`.
Templates may use Python `string.Template` placeholders, supplied through the
optional `variables` argument. Each run gets its own copy, which may be edited
to record run-specific details. Previously generated runs are not moved or
overwritten.
