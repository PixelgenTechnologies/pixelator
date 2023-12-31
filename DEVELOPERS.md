# Developer documentation

## Installation

### Using a dedicated conda/mamba environment (recommended)

You will need to [install conda](https://docs.anaconda.com/free/anaconda/install/) to use this method.
We also recommend to install mamba in your base conda environment and use that command instead of conda.
This will make the installation of the dependencies much faster.

```shell
conda activate base
conda install mamba -c conda-forge
```

All base dependencies are included in the `environment.yml` file provided in the repo root.
You can now run the following commands to create your pixelator environment:

> __NOTE__ You can replace all occurrences of `conda` in the following section with `mamba`.

- clone the repository, change directory and checkout the `dev` branch
    ```shell
    git clone git@github.com:PixelgenTechnologies/pixelator.git
    cd pixelator
    git checkout dev
    ```

- create the pixelator conda environment
    ```shell
    conda env create -f environment.yml
    ```

- activate the conda environment.
    ```shell
    conda activate pixelator
    ```

### With conda, using pip

- use editable mode (preferred)
    ```shell
    pip install -e .
    ```

- without editable mode
    ```shell
    pip install .
    ```

### With conda, using poetry

- install pixelator:
    ```shell
    poetry install
    ```

Note that poetry will not create a new virtual environment when using with a conda environment.
It will detect the activated conda environment and install pixelator dependencies directly in it.

### Without conda, using poetry

You can install the project in a separate environment managed by poetry.
If you do not have installed poetry yet, it is recommended to
[install it with pipx](https://python-poetry.org/docs/#installing-with-pipx).

Additionally, it requires the following poetry plugins to be installed:

 - poetry-plugin-export
 - poetry-dynamic-versioning

These can be installed using following commands:

```shell
poetry self add poetry-plugin-export
poetry self add poetry-dynamic-versioning[plugin]
```

You also need to install the [fastp](https://github.com/OpenGene/fastp) program.

- Clone the repository, navigate to the project root and checkout `dev`.

    ```shell
    git@github.com:PixelgenTechnologies/pixelator.git
    cd pixelator
    git checkout dev
    ```

- We recommend that you use a poetry virtual environment in the root directory of the pixelator repository.
You can do this be running the following command in the repository root:

    ```shell
    poetry config --local virtualenvs.in-project true
    ```

- Now install the project from the repository root using.

    ```shell
    poetry install
    ```

- If you want to activate a shell within your local env you can run:
    ```shell
    source .venv/bin/activate
    ```

## Update poetry.lock

Currently, the `poetry.lock` file is in the source code to ensure the consistence
with package versions. If you need to add or update a package in `pyproject.toml`
you must then update and commit the changes in the `poetry.lock` file to the repository.

The recommended way to add dependencies in poetry is by using the following command:

```shell
poetry add <package>
```

That way the `poetry.lock`` file will automatically be updated to match it and you can
commit those changes.

## How to develop using VSCode containers

If you want to use a VSCode container to develop pixelator. When you open VSCode accept the offer
to reopen the folder in a container. This will launch a docker based development environment that
should have everything you need to get going pre-installed.

A few things will work a little differently:

 - To be able to push to the repo using ssh-keys you need to run:
    ```shell
    ssh-add $HOME/.ssh/github_rsa # Replace with the name of your key
    ```
to ensure you have an ssh-key added to your ssh-agent. This will now be available inside the
container as well.

 - When you start the container VSCode might complain that e.g. `pylint` is not installed. Just wait
   until the installation process has finished - then it should have been installed.

 - An issue that you might run into is that the `pytest` test discovery does not work. A work-around
   for that is to install the pre-release version of the `ms-python.python` vscode extension, and then
   running the `Python: Clear Cache and Reload Window` action in vscode.

## Plugins system

### CLI plugins

Pixelator has a simple plugin system that allows you as a developer to add groups
of commands to the pixelator command-line interface.

This uses the ability of the python package `Click https://click.palletsprojects.com/en/8.1.x/`_
to add groups of commands to the command line interface. To write a plugin you need to define a
group, below is an example of a new group `my_plugin` with a single command `my_command`:

```python
import sys

import click


@click.group()
def my_cli_plugin():
    """My custom plugin"""
    pass


@click.command()
def my_command():
    """My custom command"""
    click.echo("Hello!")


my_cli_plugin.add_command(my_command)


if __name__ == "__main__":
    sys.exit(my_cli_plugin())
```

For pixelator to find this you need to define an `entrypoint https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins`_.
Exactly how to do this depends on which packaging method you use to build your python package. In the link above you can find
instructions for some common scenarios. The name you need use for your entry point is: `pixelator.cli_plugin`, so if you are using,
`pyproject.toml` to configure your project it could look something like this:

```{code-block} toml
:caption: pyproject.toml

[project.entry-points."pixelator.cli_plugin"]
my_plugin = "my_plugin:my_cli_plugin"
```

### Pixelator config plugins

Pixelator has a config object that contains information about assays loaded from pixelator.assays.
Additional assays can be loaded by creating a plugin that defines the `pixelator.config_plugin` entrypoint.
This entrypoint accepts a function with the current config as a parameter:

```{code-block} python
:caption: myplugin.py

def extend_config(config: "Config"):
    """
    Plugin entrypoint for extending config

    A config object will be passed in from pixelator
    at import time and can be extended here.

    :param config: The config object to extend
    """
    assay_basedir = Path(__file__).parent / "assays"
    config.load_assays(assay_basedir)
```

```{code-block} toml
:caption: pyproject.toml

[project.entry-points."pixelator.config_plugin"]
myplugin = "myplugin:extend_config"
```

## Pixelator integration tests

Some infrastructure is defined in `src/pixelator/test_utils` to run a pipeline of all commands using pytest.

Tests can be easily defined in YAML files.
See `tests/integration/test_small_D21.yaml` for an example.

To pass the panel you can use `panel` or `panel_file`.
Use panel with the key of a build in panel or panel_file with the path to a custom panel.
Only one of the two should be set and the other left empty or set to null.

e.g.:

```{yaml}
panel: "human-sc-immunology-spatial-proteomics"
panel_file: null
```

## Pixelator benchmark tests

Pixelator uses `pytest-benchmark` to enable running micro-benchmarks. Normally when running the tests these are disabled.
You can enabled them by running `pytest --benchmark-enable tests/`.
