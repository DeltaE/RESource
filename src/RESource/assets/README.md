# Packaged assets

These small, versioned lookup tables are part of RESource's reproducible workflow
and are included in the `deltae-resource` wheel.

- `legends/` contains category labels and colours used by maps and plotting code.
- `mappings/` contains stable regional lookup tables.

They are source assets, not downloaded input data or generated results, so they
remain tracked in Git. Python code can locate them without depending on the current
working directory:

```python
import pandas as pd

from RESource.assets import legend_file

land_cover_legend = pd.read_csv(legend_file("LandCover_CANgov_2020_legend.csv"))
```

Add or change an asset only with its provenance documented and with tests updated.
