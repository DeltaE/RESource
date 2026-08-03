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

## Compatibility launcher

`python run.py ...` remains available in a repository checkout, but it is only a
thin compatibility wrapper. Documentation and automation should use the installed
`resource` command.
