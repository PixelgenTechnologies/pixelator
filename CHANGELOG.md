# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED] - 2023-*-*

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

### Fixed

* Fix an issue where `--keep-workdirs` option for pytest was not available when running pytest without
  restricting the testdir to `tests/integration`.
* Fix an issue where pixelator --version would return 0.0.0.

### Removed


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
