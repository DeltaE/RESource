# License transition assessment

## Current conclusion

RESource remains licensed under MIT. The maintainer may choose Apache License 2.0
for code they own, but the repository should not claim that license for the complete
existing work until contributor and third-party provenance is resolved. This is a
repository audit, not legal advice.

The Apache Software Foundation confirms that Apache-2.0 is reusable by non-ASF
projects and recommends including the full `LICENSE`, considering a `NOTICE`, and
adding source-file notices. See the [ASF licensing FAQ](https://www.apache.org/foundation/license-faq.html)
and [official license text](https://www.apache.org/licenses/LICENSE-2.0).

## Findings

1. Git history contains a substantial contribution by Emir Fejzic, including
   `eu_dem_pipeline.py` and related workflow/configuration changes. Obtain written
   confirmation from the relevant copyright owner(s) before relicensing those
   contributions under Apache-2.0, or retain/replace separable code under its
   existing terms following legal review.
2. Files under `src/RESource/lcoe_calculator/` identify code from ATB-calc. The
   upstream project uses the BSD 3-Clause License and requires preservation of its
   copyright, conditions, and disclaimer. A root-level third-party notice/license
   inventory is needed in source and binary distributions.
3. Regional data and documentation identify additional data licenses and attribution
   requirements. Code licensing does not override those dataset terms. Generated or
   bundled data must be reviewed separately.
4. Dependency licenses normally remain the dependencies' own licenses; they should
   be inventoried for release compliance, especially if future distributions bundle
   them.

## Clearance checklist

- [ ] Confirm the legal copyright owner represented by `Copyright (c) 2023 ΔE+`.
- [ ] Obtain and archive consent for non-maintainer contributions, including employer
      rights where applicable.
- [ ] Compare copied/adapted files with upstream sources and record their exact
      versions and modifications.
- [ ] Add complete BSD-3-Clause text and attribution for ATB-calc-derived files to a
      third-party notices mechanism included in wheels and source distributions.
- [ ] Audit committed images, notebooks, configuration data, and sample datasets for
      redistribution rights.
- [ ] Decide whether historical releases stay MIT and document the effective version
      and commit of any license change.
- [ ] Replace `LICENSE` with the unmodified Apache-2.0 text, update `pyproject.toml`
      and classifiers, add an accurate `NOTICE` if required, and add SPDX headers only
      after the preceding rights review.
- [ ] Build both artifacts and verify that all required license and notice files are
      included.
- [ ] Have counsel or an appropriately authorized institutional representative review
      the final provenance record when ownership is uncertain.

