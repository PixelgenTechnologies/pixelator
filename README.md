# Pixelator

![python-version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1038/S41592--024--02268--9-B31B1B.svg)](https://doi.org/10.1038/s41592-024-02268-9)
[![conda](https://anaconda.org/bioconda/pixelator/badges/version.svg)](https://bioconda.github.io/recipes/pixelator/README.html#package-pixelator)
[![pypi](https://img.shields.io/pypi/v/pixelgen-pixelator)](https://pypi.org/project/pixelgen-pixelator/)
[![Docker Repository on Quay](https://quay.io/repository/pixelgen-technologies/pixelator/status "Docker Repository on Quay")](https://quay.io/repository/pixelgen-technologies/pixelator)
[![Tests](https://github.com/PixelgenTechnologies/pixelator/actions/workflows/tests.yml/badge.svg)](https://github.com/PixelgenTechnologies/pixelator/actions/workflows/tests.yml)

[**Documentation**](#documentation) |
[**Installation**](#installation) |
[**Usage**](#usage) |
[**Contributing**](#contributing) |
[**Contact**](#contact) |
[**License**](#license) |
[**Credits**](#credits)

Pixelator is a software package to process sequencing FASTQ from Molecular Pixelation (MPX) and
Proximity Network (PNA) assays and analyze PXL data.

It provides the `pixelator` commandline tool to process FASTQ files and generate PXL files and reports
and can be used as a python library for further downstream processing.

<p align="center">
    <img src="https://www.pixelgen.com/wp-content/uploads/2022/12/share-image-pixelgen.png" height=200
     alt="Pixelgen Technologies" />
</p>
<div align="center">© 2023 - Pixelgen Technologies AB</div>

## Documentation

More information about pixelator is available on the [Pixelgen Technologies Software documentation site](https://software.pixelgen.com/).

## Installation

Pixelgen Technologies has developed and tested pixelator extensively in Ubuntu 20.04.6 LTS. However, pixelator should run on computers installed with any recent version of the major Linux distributions, even if installed in Windows WSL.

It should only take a few minutes to install pixelator on any modern computer using any of the following methods.

### Installation with pip

Our software pixelator is available on PyPi as `pixelgen-pixelator` and can be installed with pip.
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

You can also install pixelator from source by cloning the repository.

```shell
git clone https://github.com/pixelgentechnologies/pixelator.git
cd pixelator
pip install .
```

### Using docker

The `pixelator` command-line tool can be run with docker images available on
the [GitHub container registry](https://github.com/PixelgenTechnologies/pixelator/pkgs/container/pixelator).

```shell
docker pull ghcr.io/pixelgentechnologies/pixelator:latest
docker run ghcr.io/pixelgentechnologies/pixelator:latest pixelator --help
```

You can also use the containers provided by the biocontainers project on [quay.io](https://quay.io/repository/biocontainers/pixelator).

## Usage

Our recommendation is to use pixelator via the specific Nextflow pipeline, [nf-core/pixelator](https://github.com/nf-core/pixelator).

It should take only a few seconds to download the pipeline and approx. 20 min to run the default test dataset in a normal commodity computer.

However, with MPX data, we recommend running pixelator in specialized hardware with at least 32GB RAM.

## Contributing

Contribution are welcome!
Please check out the [contributing guidelines](./CONTRIBUTING.md) for more information.

## Contact

For feature requests or bug reports, please use the GitHub [issues](https://github.com/PixelgenTechnologies/pixelator/issues).
For questions, comments, or suggestions you can use the GitHub [discussions](https://github.com/PixelgenTechnologies/pixelator/discussions).

You can also email the development team at [developers@pixelgen.com](mailto:developers@pixelgen.com).

## License

Pixelator is licensed under the [MIT](./LICENSE) license.

## Credits

Pixelator is developed and maintained by the [developers](https://github.com/PixelgenTechnologies) at [Pixelgen Technologies](https://pixelgen.com).

When using pixelator in your research, please cite the following publication:

> Karlsson, Filip, Tomasz Kallas, Divya Thiagarajan, Max Karlsson, Maud Schweitzer, Jose Fernandez Navarro, Louise Leijonancker, _et al._ "Molecular pixelation: spatial proteomics of single cells by sequencing." Nature Methods, May 8, 2024. https://doi.org/10.1038/s41592-024-02268-9.

Main development happened thanks to:

-   Jose Fernandez Navarro ([@jfnavarro](https://github.com/jfnavarro))
-   Alvaro Martinez Barrio ([@ambarrio](https://github.com/ambarrio))
-   Johan Dahlberg ([@johandahlberg](https://github.com/johandahlberg))
-   Florian De Temmerman ([@fbdtemme](https://github.com/fbdtemme))

A huge thank you to all [code contributors](https://github.com/PixelgenTechnologies/pixelator/graphs/contributors)!
