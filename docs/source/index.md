# RESource

<img src="_static/Issue_msg_box.png" alt="Issue" width="600"/>


__One of the many solutions ?__

<!-- <img src="_static/RESource_logo_2025.07.jpg" alt="RESource logo" width="200"/> -->

<!-- # RESource  -->
<img src="_static/graphic_RES_banner_common.jpg" alt="assessment_steps" width="600"/>

__A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.__

```{note}
RESource is described and applied in the peer-reviewed publication
[Mapping feasible renewable transition space: Land-use, conservation, and grid-access constraints on wind and solar in British Columbia](https://doi.org/10.1016/j.energ.2026.100077).
```

RESource is developed to enable reproducible, adaptable assessments of VRE potential that are sensitive to local constraints and planning priorities. We developed a structured, modular workflow that integrates geospatial, temporal, economic, and regulatory data to evaluate site suitability for solar and wind energy development. This structured methodology ensures transparency and transferability, allowing RESource to be adapted for different regions and scaled for long-term strategic energy planning.

<img src="_static/Assessment_steps_highLevel.jpg" alt="assessment_steps" width="500"/>

## Workflow overview
<img src="_static/workflow.jpg" alt="high_level_workflow" width="1000"/>

---

## Quick start

```bash
python -m pip install deltae-resource
resource --help
```

See the [quick-start guide](notes/quickstart.md) for installation, notebook, and
API examples. Contributors and source-checkout users use `uv` exclusively.

```{tip}
**Ready to dive deeper?** Read the [installation guide](notes/setup_guide.md),
[command-line guide](notes/run_usage.md), or [BC case study](notes/case_BC.md).
```

---

```{toctree}
:caption: 'Contents:'
:maxdepth: 1
:titlesonly:

notes/resource_builder
notes/quickstart
notes/setup_guide
notes/run_usage
notes/citation
notes/case_BC
notes/config
notes/learning
notes/data
notes/developers
notes/development_pipeline
notes/step_cache
```

```{toctree}
:caption: 'for Developers:'
:maxdepth: 1
:titlesonly:

notes/api
notes/documentation_guide
notes/deployment
notes/publishing
notes/licensing
