# Modeling assumptions

This page records assumptions that materially affect resource eligibility,
connection distance, or interpretation of results. They are part of the scenario
definition and should be reviewed before transferring RESource to another region.

## Canadian grid-connection candidates

### Evidence and provenance

The Canadian substation screen is based on the
[CODERS `transmission_lines` naming convention and data-gap report](https://github.com/DeltaE/PyPSA_BC/blob/dev/docs/QA/CODERS_transmission_lines_dataQA_2026-07-23.md).
That QA was checked on 23 July 2026 against the CODERS 2025OCT28 British
Columbia pull. It audited 1,239 transmission-line segments and 2,478 endpoint
references.

The QA establishes a three-token endpoint code of the form
`BC_<location>_<facility type>` and reports the following endpoint suffixes:

| Suffix | QA count | Meaning established by the QA | RESource treatment |
| --- | ---: | --- | --- |
| `GSS` | 213 | Generating-station switchyard | Candidate |
| `TSS` | 175 | Major transmission terminal substation | Candidate |
| `DSS` | 987 | Distribution step-down/load substation | Excluded |
| `ISS` | 154 | Industrial substation/interconnection | Excluded by node type |
| `SWS` | 19 | Switching station | Excluded |
| `JCT` | 922 | Junction or line tap without a station | Excluded |
| `INT` | 4 | International intertie | Excluded |
| `IPT` | 4 | Interprovincial tie | Excluded |

Counts in this table are transmission-line endpoint references, not unique
substations. A circuit may contain multiple segments and a node may occur more
than once.

### Implemented screening rule

For Canadian scenarios, a CODERS substation is a candidate connection point only
when all of the following are true:

1. `node_type` is `Generation` or `Terminal`.
2. `node_code` appears in either `network_node_code_starting` or
   `network_node_code_ending` in the provincial CODERS `transmission_lines` table.
3. The final `node_code` token is not `INT`, `IPT`, `SWS`, or `JCT`.

The endpoint-membership check enforces referential closure with the transmission
network. This follows the QA finding that CODERS transmission lines have no
geometry of their own and must be resolved through a separate nodes/substations
table.

The active Canadian scenarios expose the rule explicitly:

```yaml
CODERS:
  connection_filter:
    enabled: true
    eligible_node_types: [Generation, Terminal]
    excluded_node_suffixes: [INT, IPT, SWS, JCT]
    require_transmission_endpoint: true
```

Setting `enabled: false` restores the unfiltered provincial CODERS substation
table. Changing eligible types or excluded suffixes creates a different grid-access
scenario and should be reported with the results.

### Interpretation and limitations

This screen is a planning proxy for plausible transmission-level connection
locations. Retention does **not** demonstrate:

- spare interconnection or transformer capacity;
- compatible voltage or protection equipment;
- an approved interconnection request;
- ownership access, available land, or constructability;
- a feasible route from a candidate resource site; or
- that CODERS transfer capability is a substation or line thermal rating.

The source QA notes that CODERS reports transfer capability rather than per-line
MVA thermal ratings and does not provide line status or commissioning/retirement
years. Project selection therefore still requires utility data, a system-impact
study, and project-level engineering review.

Against the CODERS BC data checked on 3 August 2026, the default screen retained
144 of 644 provincial substations: 112 `Generation` and 32 `Terminal` nodes. This
count is diagnostic, not a fixed expected result; it can change with CODERS data
updates.

### Fallback behavior

If CODERS credentials, retrieval, schema validation, or filtering are unavailable,
the Canadian workflow falls back to configured OpenStreetMap transmission lines.
For non-Canadian workflows, a configured uploaded substation CSV is used first and
OSM is the fallback. OSM and CODERS are different evidence sources and results
should record which source was actually used.
