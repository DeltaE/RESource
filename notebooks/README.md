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
