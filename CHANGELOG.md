# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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


## [0.13.0] - 2023-09-13

* First public release of pixelator.
