# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD

### Added

* Finding connected components and doing Lieden based community detection using.
  networkx/graspologic (experimental feature).
* Experimental 3D heatmap plotting feature.
* Optional caching of layouts to speed up computations in some scenarios.
* `experimental` mark that can be added to functions that are not yet production ready.
* Graph layout computations using networkx as the graph backend (experimental feature).
* The underlying graph instance e.g. a igraph or networkx instance is exposed as a property called `raw` from the `Graph` class.
* Monte Carlo permutation support for calculated Moran's I (`morans_z_sim`) in `polarization_scores`.

### Changed

* `mean_reads` and `median_reads` in adata.obs to `mean_reads_per_molecule` and `median_reads_per_molecule` respectively.
* Drop support for python 3.8 and 3.9.
* Change output format of `collapse` from csv to parquet.
* Change input and output format of `graph` from csv to parquet.
* Change input format of `annotate` from csv to parquet.
* Rename the report to "qc report"
* Add a Reads per Molecule frequency figure to the sequencing section of the qc report.
* Remove placeholder warning of missing data for not yet implemented features.
* Change "Median antibody molecules per cell" to "Average antibody molecules per cell" in the qc report.
* Refactoring of the graph backend implementations module.
* Activating networkx as the backend is now done by setting `PIXELATOR_GRAPH_BACKEND="NetworkXGraphBackend"`
  (previously `PIXELATOR_GRAPH_BACKEND=True` was used).
* Speeding up `amplicon` step by roughly 3x

### Fixed

* Nicer error messages when there are no components valid for computing colocalization.
* Cleaned out remaining igraph remnants from `Graph` class
* A bunch of warnings

### Removed

* `graph` no longer outputs the raw edge list

## [0.15.2] - 2023-10-23

### Fixed

* Fixed broken pixeldataset aggregation for more than two samples.
* Fixed a bug in graph generation caused by accidentally writing the index to the parquet file.
  For backwards compatibility, if there is a column named `index` in the edgelist, this
  will be removed and the user will get a warning indicating that this has happened.


## [0.15.1] - 2023-10-18

### Fixed

* Fixed a bug in filtering pixeldataset causing it to return the wrong types.
* Fixed a bug in graph layout generation due to incorrect data frame concatenation.


## [0.15.0] - 2023-10-16

### Added

* Add support for Python 3.11.
* Add early enablement work for a networkx backend for the graph stage.

### Fixed

* Fix report color axis in report figures not updating when selecting markers or cell types.
* Remove placeholder links in report tooltips.
* Fix a bug where aggregating data did not add the correct sample, and unique component columns.


## [0.14.0] - 2023-10-05

### Added

* Lazy option for edge list loading (`pixeldataset.edgelist_lazy`), which returns a
  `polars` `LazyFrame` that can be used to operate on the edge list without reading
  all of it into memory.
* Option (`ignore_edgelists`) to skip the edge lists when aggregating files. This defaults
  to `False`.

### Changed

* Types on the edge list in memory will utilize the `pandas` `category` type for string, and
  `uint16` for numeric values to lower the memory consumption when working with the
  edge list
* Remove `--pbs1` and `--pbs2` commandline arguments to `pixelator single-cell adapterqc`.
* Restructure report figures.
* Improve metric names and tooltips in the report.
* Synchronize zoom level between the scatter plots in cell annotations section of the report.
* Add report placeholder for missing cell annotation data
* Add `Fraction of discarded UMIs` and `Avg. Reads per Molecule` metrics to the report.

### Fixed

* Fix an issue where pixelator --version would return 0.0.0 when installing in editable mode.


## [0.13.1] - 2023-09-15

### Added

### Changed

* Unpin igraph dependency to allow for newer versions of igraph to be used.
* Cleanup README and point to the external documentation site.
* Change PyPi package name to pixelgen-pixelator.

### Fixed

* Fix an issue where `--keep-workdirs` option for pytest was not available when running pytest without
  restricting the testdir to `tests/integration`.
* Fix an issue where pixelator --version would return 0.0.0.

### Removed

* `clr` and `relative` transformation options for the colocalization computations in `analysis`


## [0.13.0] - 2023-09-13

* First public release of pixelator.
