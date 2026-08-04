# Reporting

Builds a self-contained HTML report for a country: resolved input config per
scenario (with provenance), a scenario-vs-scenario contrast, and solar plots.

The report is assembled purely from data already written to disk by the
`resource` pipeline (`results/<Country>/<Region>/<RUN_ID>/`) — it never
re-runs the assessment. For each requested scenario/region it picks the most
recent matching `RUN_ID` directory on disk.

```python
from RESource.reporting import build_report

build_report("CAN", regions=["BC"])
```

Or via the CLI (`reporting` extra required for `jinja2`):

```
resource-report CAN --regions BC
```

Images are embedded as base64 data URIs, so the output `.html` file is
portable and reproducible without any accompanying assets.
