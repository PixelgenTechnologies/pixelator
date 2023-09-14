# Pixelator


![python-version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

[//]: # (TODO: Enable this once available)
[//]: # (![conda]&#40;https://anaconda.org/bioconda/pixelator/badges/version.svg&#41;)
[//]: # (![pypi]&#40;https://img.shields.io/pypi/v/pixelgen-pixelator&#41;)


[**Documentation**](#Documentation) |
[**Installation**](#installation) |
[**License**](#License)


Pixelator is a software package to process sequencing FASTQ from Molecular Pixelation (MPX) assays
and analyze PXL data.

It provides the `pixelator` commandline tool to process FASTQ files and generate PXL files and reports
and can be used as a python library for further downstream processing.

© 2023 - Pixelgen Technologies AB

<p align="center">
    <img src="https://www.pixelgen.com/wp-content/uploads/2022/12/share-image-pixelgen.png" height=200
     alt="Pixelgen Technologies" />
</p>

## Documentation

More information about pixelator is available on the [Pixelgen Techologies documentation site](https://software.pixelgen.com/).

## Installation

### Installation with pip

pixelator is available on pypi as `pixelgen-pixelator` and can be installed with pip.
It is recommended to install pixelator in a separate virtual environment.

```shell
pip install pixelgen-pixelator
```

### Installation with conda / mamba

A conda package is available on the bioconda channel and can be installed with conda or mamba.

```shell
conda install -c bioconda pixelator
```
or
```shell
mamba install -c bioconda pixelator
```

### Installation from source

Pixelator can be installed from source by cloning the repository.

```shell
git clone https://github.com/pixelgentechnologies/pixelator.git
cd pixelator
pip install .
```

### Using docker

The pixelator commandline tool can be run with docker images available on
the [GitHub container registry](https://github.com/PixelgenTechnologies/pixelator/pkgs/container/pixelator).

```shell
docker pull ghcr.io/pixelgentechnologies/pixelator:latest
docker run ghcr.io/pixelgentechnologies/pixelator:latest pixelator --help
```

## License

Pixelator is licensed under the [GPL-2.0](./LICENSE) license.

## How to contribute

Contribution are welcome!
Please check out the [contributing guidelines](./CONTRIBUTING.md) for more information.

## Contact

For feature requests or bug reports, please use the GitHub issues.
For questions, comments, or suggestions you can use the GitHub discussions.

You can also email the development team at [developers@pixelgen.com](mailto:developers@pixelgen.com).

# Credits

Pixelator is developed and maintained by the [developers](https://github.com/PixelgenTechnologies) at [Pixelgen Technologies](https://pixelgen.com).

Main development happened thanks to:

- Jose Fernandez Navarro ([@jfnavarro](https://github.com/jfnavarro))
- Alvaro Martinez Barrio ([@ambarrio](https://github.com/ambarrio))
- Johan Dahlberg ([@johandahlberg](https://github.com/johandahlberg))
- Florian De Temmerman ([@fbdtemme](https://github.com/fbdtemme))

A huge thank you to all code contributors! See the [Contributors Graph](https://github.com/PixelgenTechnologies/pixelator/graphs/contributors) for details.

A non-exhaustive list of contributors follows:

- Filip Karlsson ([@fika-pixelgen](https://github.com/fika-pixelgen))
- Max Karlsson ([@maxkarlsson](https://github.com/maxkarlsson))
