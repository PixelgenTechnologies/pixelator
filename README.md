# Pixelator

Pixelator is a software package to process sequencing FASTQ from Molecular Pixelation (MPX) assays and analyze PXL data (maintained by Pixelgen Technologies AB).

Pixelator is compatible with Python 3.8, 3.9 and 3.10 and has been tested on Linux and macOS x64 machines.

## Installation

### Create a virtual environment using conda/mamba

You will need to [install conda](https://docs.anaconda.com/free/anaconda/install/) to create a dedicated environment.
We also recommend to install mamba in your base conda environment and use that command instead of conda.
This will make the installation of the dependencies much faster.

```shell
conda activate base
conda install mamba -c conda-forge
mamba create -n pixelator python==3.10
mamba activate pixelator
```

### Using pip

```shell
pip install pixelgen-pixelator
```

### Using conda

```shell
conda install -c bioconda pixelgen-pixelator
```

### Using mamba

```shell
mamba install -c bioconda pixelgen-pixelator
```

### Using docker

You can also [use Pixelator from our distributed docker images](./USAGE.md#pixelator-docker-images).

## How to use Pixelator

See [USAGE.md](./USAGE.md)

## How to develop

See [DEVELOPERS.md](./DEVELOPERS.md)

## How to contribute

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Contact and credits

See [AUTHORS.md](./AUTHORS.md)
