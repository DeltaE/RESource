# Command-line usage

The installed command is:

```text
resource CONFIG [--year YYYY] [--regions CODE ...]
```

In a development checkout, prefix the command with `uv run`:

```bash
uv run resource config/config_BC_baseline.yaml --year 2024 -r BC
```

## Arguments

`CONFIG`
: Path to a YAML configuration file containing `region_mapping`.

`--year`, `-y`
: Weather year. The current CLI default is 2024 and overrides the corresponding
  configuration value.

`--regions`, `-r`
: Zero or more region codes. If omitted, every region declared by the
  configuration is processed.

## Examples

```bash
# All configured regions
uv run resource config/CAN_baseline.yaml --year 2024

# Selected regions
uv run resource config/config_WB6_2023.yaml --year 2023 -r AL BA

# Command reference
uv run resource --help
```

The pipeline processes wind and solar for each selected region. A failed
region/resource pair is recorded in `results/logs/runtime_log.txt`; remaining
pairs continue to run. A final status of `PARTIAL` means at least one pair failed.

## Multi-year runs

The packaged multi-year launcher runs each weather year as an independent
subprocess, continues after a failed year, and returns a failure status if any year
fails:

```bash
uv run resource-multiyear config/CAN_baseline.yaml \
  --start 2014 --end 2024 --regions BC
```

Use `resource-multiyear --help` for the complete command reference. Outputs retain
the normal RESource directory structure; ensure the selected configuration keeps
different weather years distinguishable before starting a large run.

### ERA5 cutouts only

For a fixed weather climatology, use the dedicated downloader. It does not run land,
capacity, scoring or generation stages:

```bash
uv run resource-cutout-multiyear config/CAN_baseline.yaml \
  --start 2016 --end 2025 --region BC
```

The command validates and reuses complete annual cutouts, quarantines invalid files,
continues after individual failures, and writes a JSON manifest under
`results/manifests/`. Before each annual CDS job it releases unreachable Python
objects from the previous iteration and records the memory-cleanup report. This
does not delete UV, CDS, temporary-file or downloaded-data caches.

The standard single-year `resource CONFIG --year YEAR` workflow uses the same
memory-release task immediately before ERA5/CDS preparation.

Both commands route CDS and Python temporary files to the repository filesystem at
`data/tmp/resource-cds/`. This avoids small system `/tmp` mounts. The scratch path is
ignored by Git and is not a durable output; validated NetCDF cutouts remain under
the configured cutout root.

## Compatibility launcher

`python run.py ...` remains available in a repository checkout, but it is only a
thin compatibility wrapper. Documentation and automation should use the installed
`resource` command.
