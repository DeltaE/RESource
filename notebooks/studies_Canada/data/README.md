# Resource plot data

`electricity_generation_bc_scenarios.csv` was extracted from the inline table
in `notebooks/resource_plots.ipynb`. The original table labels the dataset
`2023`, but does not identify its provider, source URL, retrieval date, or
license. It also ends with a literal `...`, indicating omitted records. Treat
the values as an incomplete, unverified extract until the complete source and
its provenance are confirmed. The ellipsis is retained as a placeholder row
marked `source_row_placeholder=true`. The one source row that was commented
out in the notebook is retained and marked `source_row_commented=true`; the
plot excludes that row, as the original comment indicated.

`resource_plot_capacity_bins.csv` externalizes the manual capacity classes
previously embedded in the notebook. Capacity bounds and assignment behavior
are unchanged (`pandas.cut` right-closed intervals, including the lower edge
of the first interval). Labels describe those intervals in MW. These are
visualization classes, not policy thresholds.

`resource_plot_targets.csv` retains the 10 GW solar and 30 GW wind selection
budgets previously passed by the notebook. They are legacy visualization
settings, not policy recommendations.
