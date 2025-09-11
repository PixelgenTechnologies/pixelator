# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [x.x.x] - 2025-xx-xx


### Fixed

-   `pixelator single-cell demux` respects the `--cores` option.

### Added

-   Validation on antibody panel sequence lengths and characters.
-   If panel contains uniprot_id column, it will be added to adata.var.

## [0.21.4] - 2025-08-27

### Added

-   Support limited concurrent access to pixelfiles for selected read-only operations

### Fixed

-   Reduce memory usage during layout computation.
-   Fix support for single-end reads.

## [0.21.3] - 2025-07-04

### Fixed

-   Fix support for long over-lapping reads in paired-end amplicon.

## [0.21.2] - 2025-06-26

### Added

-   We now distribute container images on [quay.io](https://quay.io/repository/pixelgen-technologies/pixelator).

### Fixed

-   Performance regression in `graph` causing memory usage to grow by a factor of two.

## [0.21.1] - 2025-06-24

### Fixed

-   Added missing `gzip` dependency in docker container.

## [0.21.0] - 2025-06-24

### Added

-   Added component denoising functionality to be invoked as `pixelator single-cell-pna denoise`.
    Denoising removes nodes from markers that are overrepresented in the 1-core layer
    of a cell graph which suggests that they are caused by bleed-over from other cells.
-   Added the proxiome-immuno-156-FMC63 panel, which includes all the markers from the
    proxiome-immuno-155 panel, but with the addition of the FMC63 marker.
-   Support for python 3.12. The container builds for pixelator will use python 3.12 from
    this point.
-   Added molecules_post_umi_collision_removal and reads_post_umi_collision_removal
    to the graph report file.

### Changed

-   The proximity data will no longer be reported for markers with fewer than 10
    molecules in the cell.
-   Two columns, `marker_1_freq` and `marker_2_freq`, have been added to the proximity
    data to indicate the respective marker frequencies.
-   Panel files no longer allow `_` as a character in the marker ids, since this causes
    problems with Seurat in R in downstream analysis.

### Fixed

-   Pixelator can now handle sample names starting with a number for PNA data.
-   Work around a bug in pyarrow that could cause the `pixelator single-cell-pna demux` to generate invalid
    molecule embeddings for very large input files.
-   Update tau metric information after denoising components.
-   Fix the number of input reads and molecules in the graph report to correspond to the raw counts and add
    the number of reads and molecules after UMI collision removal as separate entries in the report.

## [0.20.1] - 2025-04-24

### Changed

-   The source dist no longer includes the `tests` directory since this is to big
    to upload to pypi when including the necessary test data.

## [0.20.0] - 2025-04-24

This is a large release that reorganizes Pixelator in a major way. We introduce the new module `pixelator.pna`,
which is used to process and analyze data from the Proximity Network Assay (PNA). As a consequence of this, we
have moved what used to be Pixelator into two modules: `pixelator.mpx` for processing and analyzing Molecular
Pixelation (MPX) data, and `pixelator.common`, which contains functions and classes used by both methods.

As a user, the most noticeable difference is that if you want to read pixel files to work with MPX data, you will now need to use:

**Molecular Pixelation data**

```
from pixelator import read_mpx
from pixelator import simple_aggregate_mpx
```

**Proximity Network Assay data**

```
from pixelator import read_pna
```

With the introduction of the Proximity Network Assay, we have revamped the entire data processing
pipeline to handle the new data type. We have also introduced a new pixel file format
that utilizes [DuckDB files](https://duckdb.org/) to store the data. This allows us to be much
more flexible and faster in loading and processing the data compared to the zip-based format we use
for MPX data.

Another major workflow change in how Pixelator is used to work with Proximity Network Assay data
is that there is no need to explicitly aggregate data. Instead, the `read_pna` function
accepts a list of files and automatically aggregates the data for you,
without the need to create a new aggregated PXL file.

For an updated view on how to work with Pixelator, please check out the [documentation](https://software.pixelgen.com/).
There, you will also find updates on the new algorithms used for the Proximity Network Assay, etc.

### Added

-   All functionality for analyzing PNA data.

### Changed

-   The command-line entry point for MPX data has changed from `pixelator single-cell` to `pixelator single-cell-mpx`.

For analysis of MPX data, you should now read MPX data like this:

```
from pixelator import read_mpx

pxl_dataset = read_mpx("path/to/pxl")
```

-   The function `dsb_normalize` is now a common method for both MPX and PNA data:

```
from pixelator.common.statistics import dsb_normalize
```

### Fixed

-   In some cases, `local_g` would return scrambled results due to inconsistent node ordering. This
    has now been fixed.

## [0.19.0] - 2024-12-10

### Changed

-   Name components using UPI hashes.
-   Run multiple iterations of multiplet recovery during the `graph` stage specified using `--max-refinement-recursion-depth`.
-   Specify maximum number of edges that can be removed between two sub-components during multiplet recovery using `--max-edges-to-split`.
-   Support for MultiGraphs in `pmds_layout`
-   Support multiple targets in `plot_colocalization_diff_volcano` and `plot_colocalization_diff_heatmap`.
-   If demultiplexing has a success rate lower than 50% the command will exit with a status of 1. This prevents further pipeline stages to be run on
    what is probably bad data.
-   Clarify that `--min-size` and `--max-size` in the `annotate` stage should not be used at the same time as `--dynamic-filter`.
-   Setting a lower threshold of 300 edges when `--dynamic-filter` is used in the `annotate` stage, components smaller than that will always
    be filtered. Note that this can still be overridden by setting `--min-size` explicitly.
-   Clarify error message when all components are filtered out in the `annotate` stage.

### Added

-   Add `depth` column to `discarded_edgelist.parquet` output of the `graph` stage that indicates at which refinement iteration the edge is removed.
-   Add `edges_removed_in_multiplet_recovery_first_iteration`, `edges_removed_in_multiplet_recovery_refinement` and `fraction_edges_removed_in_refinement` to graph report.json.
-   Add `is_potential_doublet` and `n_edges_to_split_doublet` columns to adata.obs.
-   Add `fraction_potential_doublets` and `n_edges_to_split_potential_doublets` to annotate report.json.
-   Add `--max-edges-to-split` option to `graph` to specify the maximum number of edges that can be removed between two sub-components during multiplet recovery.
-   Add `abundance_colocalization_plot` function to make scatter plots of selected marker-pairs' abundance.
-   Add `plot_polarity_diff_volcano` to make statistical comparison plots of selected component groups.
-   Add `get_differential_polarity` to statistically compare polarity scores of selected component groups.

### Removed

-   Remove the `components_recovered.csv` output from the `graph` stage.

### Fixed

-   Improved error message when the number of nodes is lower than the number of requested dimensions in `pmds_layout`.

## [0.18.3] - 2024-09-26

### Fixed

-   Improved memory usage when aggregating PXL files with precomputed layouts.

## [0.18.2] - 2024-07-16

### Changed

-   Bump polars to stable 1.x series

### Fixed

-   Fix a qc report crash issue when the layout stage is run in a pipeline due to an unsupported parameter type.

## [0.18.1] - 2024-07-12

### Changed

-   Bump `umi_tools` version requirements

## [0.18.0] - 2024-07-11

### Added

-   Add minimum marker count `colocalization_min_marker_count` parameter to calculate colocalization score.
-   Add `density_scatter_plot` function to make two-marker abundance scatter plots with pseudo-density coloring.
-   Add `wpmds` option in `pmds_layout` to compute edge weighted layouts. This is now set as the default layout algorithm.
-   Add `dsb_normalization` function for normalization of marker abundance.
-   Add a `Fraction of Outlier Cells` metric to the QC report.
-   Add a `Panel Version` metadata field to the QC report.
-   Add support for datasets generated using the `human-sc-immunology-spatial-proteomics-2` panel.

### Changed

-   The default value for `normalize_counts` in `local_g` is now `False` instead of `True`.
-   The default transformation for the calculation of the colocalization score is now `rate-diff` instead of `log1p`.
-   Rename `edge_rank_plot` function to `molecule_rank_plot`.

### Fixed

-   Fix a bug in `compute_transition_probabilities` when `k>1` where the stochastic matrix was not correctly row-normalized.
-   Fix a bug in `local_g` when `use_weights=False` where the adjacency matrix was not correctly expended if `k>1`.
-   Fix a bug where `a_pixels_per_b_pixel` summary statistics where equal to the `b_pixels_per_a_pixel` statistics.
-   `collapse` will return exit code 137 when one of the child processes is killed by the system (e.g. because it is
    to much memory). This allows e.g. Nextflow to retry the process with more memory automatically.
-   Hide the `Sample Description` metadata field in the QC report when no value is available.
-   Fix an issue where boolean parameters were formatted as integers in the Parameters section of the QC report.
-   Fix a bug in aggregating files with precomputed layouts, where the lazy-loading of the layouts was not working correctly.

### Removed

-   Remove the `Pixel Version` metadata field from the QC report.

## [0.17.1] - 2024-05-27

### Fixed

-   Poor performance when writing many small layouts to pxl file (~45x speed-up). This should almost only
    impact test scenarios, since most real components should be large enough for this not to be an issue.

## [0.17.0] - 2024-05-23

### Added

-   Add `rate_diff_transformation` function with `rate-diff` alias as an alternative option for transforming marker counts before colocalization calculation.
-   Add `local_g` function to compute spatial autocorrelation of marker counts per node.
-   Add `compute_transition_probabilities` function to compute transition probabilities for k-step random walks for node pairs in a graph.
-   Add QC plot showing UMIs per UPIA vs Tau.
-   Add plot functions showing edge rank and cell counts.
-   Add 2D and 3D graph plot functions.
-   Add heatmap plot functions showing colocalization and differential colocalization.
-   Add volcano plot (value difference vs log p-value) function for differential colocalization.
-   Add a function to calculate the differential colocalization between two conditions.
-   Performance improvements and reduced bundle size in QC report.
-   Improved console output in verbose mode.
-   Improved logging from multiprocessing jobs.
-   Improved runtime for graph creation.
-   Added PMDS layout algorithm.
-   Add `--sample_name` option to `single-cell amplicon` to overwrite the name derived from the input filename.
-   Add `--skip-input-checks` option to `single-cell amplicon` to make input filename checks warnings instead of errors.
-   `PixelDataset` instances are now written to disk without creating intermediate files on-disk.
-   A nice string representation for the `Graph` class, to let you know how many nodes and edges there are in the current graph object instance.
-   Metric to collect molecules (edges) in cells with outlier distributions of antibodies (aggregates).
-   Provide typed interfaces for all per-stage report files using pydantic.
-   Centralize pixelator intermediate file lookup and access.
-   Add a `precomputed_layouts` property to `PixelDataset` to allow for loading precomputed layouts.
-   Add `pixelator single-cell layout` stage to pixelator, which allows users to compute layouts for a PXL file that can then be used to visualize the graph in 2D or 3D downstream.
-   Add minimum marker count `polarization_min_marker_count` parameter to calculate Polarity Score.
-   Add "log1p" as an alternative for `PolarizationNormalizationTypes`.
-   Add `convert_indices_to_integers` option when creating graphs.
-   Add a feature flag module to aid in the development of new features.

### Changed

-   Change name and description of `Avg. Reads per Cell` and `Avg. Reads Usable per Cell` in QC report.
-   The output name of the `.pxl` file from the `annotate` step is now `*.annotated.dataset.pxl`.
-   The output name of the `.pxl` file from the `analysis` step is now `*.analysis.dataset.pxl`.
-   The term `edges` in `metrics` and `adata` is now replaced with `molecules`.
-   Renaming of variables in per-stage JSON reports.
-   Changed name of TCRb to TCRVb5 antibody in human-immunology-panel file and bumped to version 0.5.0.
-   Renaming of component metrics in adata.
-   Use MPX graph compatible permutation strategy when calculating Moran's I related statistics.
-   Marker filtering is now done after count transformation in polarization score calculation.
-   Use the input read count at the annotate stage for the `fraction_antibody_reads_in_outliers` metric denominator instead of the total raw input reads.
-   Use common analysis engine to orchestrate running different "per component" analyses, like polarization and colocalization analysis (yielding a roughly 3x speed-up over the previous approach).
-   The default transformation for the calculation of the polarity score is now `log1p` instead of `clr`.

### Fixed

-   Fix a bug in how discarded UMIs are calculated and reported.
-   Fix deflated counts in the edgelist after collapse.
-   Fix a bug where an `r1` or `r2` in the directory part of a read file would break file name sanity checks.
-   Fix a bug where the wrong `r1` or `r2` in the filename would be removed when multiple matches are present.
-   Logging would cause deadlocks in multiprocessing scenarios, this has been resolved by switching to a server/client-based logging system.
-   Fix a bug in the amplicon stage where read suffixes were not correctly recognized.
-   Ensure deterministic results from `pmds_layout` (given a set seed).
-   Fix an issue with the `fraction_antibody_reads_usable_per_cell` metric where the denominator read count was not correctly averaged with the cell count.

### Removed

-   Remove multi-sample processing from all `single-cell` subcommands.
-   Remove `--input1_pattern` and `--input2_pattern` from `single-cell amplicon` command.
-   Self-correlations, e.g. CD8 vs CD8 are no longer part of the colocalization results, as these values will always be undefined.
-   Remove `umi_unique_count` and `upi_unique_count` from `edgelist`.
-   Remove `umi` and `median_umi_degree` from `component` metrics.
-   Remove `normalized_rel` and `denoised` from `obsm` in `anndata`.
-   Remove the `denoise` function.
-   Remove cell type selector in QC report for UMAP colored by molecule count plots.
-   Remove `clr` as a transformation option in `pixelator analysis`.

## [0.16.2] - 2024-03-19

### Fixed

-   Uninitialized value for `--polarization-n-permutations`

## [0.16.1] - 2024-01-12

### Fixed

-   Bug in README shield formatting

## [0.16.0] - 2024-01-12

This release introduces two major change in pixelator:

1.  the Graph backend has been switched from using igraph to using networkx
2.  the license has been changed from GLP2.0 to MIT

### Added

-   Experimental 3D heatmap plotting feature.
-   Optional caching of layouts to speed up computations in some scenarios.
-   `experimental` mark that can be added to functions that are not yet production ready.
-   The underlying graph instance e.g. a networkx `Graph` instance is exposed as a property called `raw` from the pixelator `Graph` class.
-   Monte Carlo permutations supported when calculating Moran's I (`morans_z_sim`) in `polarization_scores`.

### Changed

-   The default (and only) graph backend in pixelator is now based on networkx.
-   `mean_reads` and `median_reads` in adata.obs to `mean_reads_per_molecule` and `median_reads_per_molecule` respectively.
-   Drop support for python 3.8 and 3.9.
-   Change output format of `collapse` from csv to parquet.
-   Change input and output format of `graph` from csv to parquet.
-   Change input format of `annotate` from csv to parquet.
-   Rename the report to "qc report"
-   Add a Reads per Molecule frequency figure to the sequencing section of the qc report.
-   Remove placeholder warning of missing data for not yet implemented features.
-   Change "Median antibody molecules per cell" to "Average antibody molecules per cell" in the qc report.
-   Refactoring of the graph backend implementations module.
-   Speeding up the `amplicon` step by roughly 3x.

### Fixed

-   Nicer error messages when there are no components valid for computing colocalization.
-   A bunch of warnings.

### Removed

-   `graph` no longer outputs the raw edge list.
-   igraph has been dropped as a graph backend for pixelator.

## [0.15.2] - 2023-10-23

### Fixed

-   Fixed broken pixeldataset aggregation for more than two samples.
-   Fixed a bug in graph generation caused by accidentally writing the index to the parquet file.
    For backwards compatibility, if there is a column named `index` in the edgelist, this
    will be removed and the user will get a warning indicating that this has happened.

## [0.15.1] - 2023-10-18

### Fixed

-   Fixed a bug in filtering pixeldataset causing it to return the wrong types.
-   Fixed a bug in graph layout generation due to incorrect data frame concatenation.

## [0.15.0] - 2023-10-16

### Added

-   Add support for Python 3.11.
-   Add early enablement work for a networkx backend for the graph stage.

### Fixed

-   Fix report color axis in report figures not updating when selecting markers or cell types.
-   Remove placeholder links in report tooltips.
-   Fix a bug where aggregating data did not add the correct sample, and unique component columns.

## [0.14.0] - 2023-10-05

### Added

-   Lazy option for edge list loading (`pixeldataset.edgelist_lazy`), which returns a
    `polars` `LazyFrame` that can be used to operate on the edge list without reading
    all of it into memory.
-   Option (`ignore_edgelists`) to skip the edge lists when aggregating files. This defaults
    to `False`.

### Changed

-   Types on the edge list in memory will utilize the `pandas` `category` type for string, and
    `uint16` for numeric values to lower the memory consumption when working with the
    edge list
-   Remove `--pbs1` and `--pbs2` commandline arguments to `pixelator single-cell adapterqc`.
-   Restructure report figures.
-   Improve metric names and tooltips in the report.
-   Synchronize zoom level between the scatter plots in cell annotations section of the report.
-   Add report placeholder for missing cell annotation data
-   Add `Fraction of discarded UMIs` and `Avg. Reads per Molecule` metrics to the report.

### Fixed

-   Fix an issue where pixelator --version would return 0.0.0 when installing in editable mode.

## [0.13.1] - 2023-09-15

### Added

### Changed

-   Unpin igraph dependency to allow for newer versions of igraph to be used.
-   Cleanup README and point to the external documentation site.
-   Change PyPi package name to pixelgen-pixelator.

### Fixed

-   Fix an issue where `--keep-workdirs` option for pytest was not available when running pytest without
    restricting the testdir to `tests/integration`.
-   Fix an issue where pixelator --version would return 0.0.0.

### Removed

-   `clr` and `relative` transformation options for the colocalization computations in `analysis`

## [0.13.0] - 2023-09-13

-   First public release of pixelator.
