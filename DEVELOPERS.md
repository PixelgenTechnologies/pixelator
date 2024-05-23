# Developer documentation

This document covers how to get up and running with developing pixelator.

## Table of contents

- [Setup the developer environment](#setup-the-developer-environment)
- [Pixelator developer tools](#pixelator-developer-tools)
- [Adding new dependencies](#adding-new-dependencies)
- [Testing pixelator](#testing-pixelator)
  - [Pixelator unit tests](#pixelator-unit-tests)
  - [Pixelator integration tests](#pixelator-integration-tests)
- [Understanding the pixelator plugin system](#understanding-the-pixelator-plugin-system)


## Setup the developer environment

We recommend that you develop pixelator in a separate python virtual environment. There are
many ways to setup a virtual environment, we recommend using mamba. If you have a different
way that you prefer, you can use that as well.

Below we outline the steps you need to take to setup your development environment using mamba.

### 1. Prepare and checkout the pixelator repository for Github

Pixelator uses `git-lfs` to manage large files. If you do not have `git-lfs` installed, you can install it using the following commands:

```shell
git lfs install
```

You can then clone the pixelator repo (this will checkout the default `dev` branch for you):

```shell
git clone git@github.com:PixelgenTechnologies/pixelator.git
cd pixelator
```

You are now ready to setup your development environment.

### 2. Setup a virtual environment for pixelator

You will need to [install mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to use this method.

If you are on a Mac or Linux you can use the following commands to install mamba:

```shell
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

If you are one Windows refer to these instructions: https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows

For more information on how to install mamba see the [mamba documentation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).


All base dependencies are included in the `environment.yml` file provided in the repo root.
You can now run the following commands to create your pixelator environment:

```shell
mamba env create -f environment.yml
```

Activate the conda environment:
```shell
mamba activate pixelator
```

### 3. Install pixelator

To install pixelator you can now run:

```shell
task install
```

This will install pixelator with all it's dependencies in your virtual environment, and
setup git hooks to run pre-commit checks before you commit your changes.

You are now ready to start developing pixelator. Congrats!

## Pixelator developer tools

Pixelator provides a number of utility scripts (you have already used one above) to help with development and testing using [Task](https://taskfile.dev/).
We are using `tasks` to make it easier to run common development tasks such as running tests, and formatting code.
The `tasks` scripts allow us to self-document these commands and make them easy to discover.

> [!NOTE]
> If you have followed the installation instructions above you should have `task` installed
> in your virtual environment.
> Otherwise you can follow the installation instructions here [here](https://taskfile.dev/#/installation).

> [!TIP]
> Depending on the installation method the task executable might be named `task` or `go-task`.

### Viewing the taskfile

Run the following command to view the available tasks:

```shell
task --list
```
It will look something like this:

```console
* coverage:                            Run tests using pytest and generate a coverage report.
* format:                              Format code using black.
* format-check:                        Check code formatting using ruff.
* install:                             Install the project using poetry (including dependencies and git hooks).
* lint:                                Run linting using ruff.
* test:                                Run tests using pytest with the default flags defined in pyproject.toml.
* test-all:                            Run all tests using pytest.
* test-nf-core-pixelator:              Run the default nf-core/pixelator test profile with this version of pixelator.
* test-web:                            Run web tests using pytest.
* test-workflow:                       Run workflow tests using pytest.
* test-workflow-external:              Run external workflow tests using pytest.
* typecheck:                           Run type checking using mypy.
* tests:update-report-test-data:       Update the report test data using the nf-core/pixelator test profile.
* tests:update-web-test-data:          Create web test data.
```

View more detailed documentation for a specific task, for example:

```shell
task test-nf-core-pixelator --summary
```
```console
task: test-nf-core-pixelator

Run the default nf-core/pixelator test profile with this version of pixelator.

If the pipeline is not found in the directory `PIPELINE_SOURCE_DIR`, it will be cloned
from the PixelgenTechnologies/nf-core-pixelator repository using the `PIPELINE_BRANCH` branch.
By default the "pixelator-next" branch of nf-core-pixelator will be used.

If the `PIPELINE_SOURCE_DIR` exists this task will assume that the pipeline is already present and checked out
in the right branch.

Note that the current dev version of pixelator must be installed and available in the PATH.

If the `RESUME` environment variable is set to "true", the pipeline will be resumed if it was previously run.

commands:
 - Task: tests:pull-nf-core-pixelator
 - Task: tests:run-nf-core-pixelator-test-profile
```

Some commands have confirmation prompts for potentially dangerous actions such as deleting files.
These prompts can be skipped by adding `--yes` to the command.

Some commands have variables that can be set to change the behavior of the command.
These variables are passed as environment variables to the command.

For example:

```shell
RESUME=true task test-nf-core-pixelator
```
## Adding new dependencies

If you need to add a new dependency to pixelator this should be managed through `poetry`.
You do this by running:

```shell
poetry add <package>
```

This will update `pyproject.toml` and `poetry.lock` with the new dependency. Both
these files should be committed to the repository to record the new dependency.

## Testing pixelator

Pixelator uses `pytest` to run tests. We have a number of different types of tests that can be run.
The section below outline how to run these tests, and in what situations they are useful.


### Pixelator unit tests

The pixelator unit tests suite is focused on testing the individual classes, functions, etc.
Unit tests are run in pixelator in isolation. They are intended to be fast to run and provide quick
feedback during development.

You can run the pixelator unit tests by using:

```shell
task test
```

This skips some particularly slow tests and is useful for quick feedback during development.
If you want to run all tests you can do so by using:

```shell
task test-all
```

This can be useful to run before pushing changes to the repository to make sure that
everything is working as expected.

You can also initiate a test loop that runs every time you save your files by:

```shell
task test-watch
```

This can be useful when you are working on the code and want to get immediate feedback.

Since it can be a little bit slow to run all the tests during development you might want
to start a test loop that only runs the tests that are affected by the files you are working on
by using something like below:

```shell
TEST_PATH="tests/pixeldataset" task test-watch
```


### Pixelator integration tests

#### Pixelator internal integration tests
Pixelator has a basic set of integration tests that that will run all pixelator commands and
check that they run without errors. Note that these do not verify the outputs of each command in
any comprehensive way, but are useful for catching basic errors.

These tests can be run by using:

```shell
task test-workflow
```

The code necessary to run these tests are defined in `src/pixelator/test_utils`, and it allows you
to run a pipeline of all commands using `pytest`.

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


#### Pixelator nf-core/pixelator integration tests

Pixelator is built to be orchestrated by the nextflow pipeline [nf-core/pixelator](https://github.com/nf-core/pixelator).
This means that is is useful to have a simple way to test the integration between the two.

You can do this using tasks:

```shell
task test-nf-core-pixelator
```

It can also be triggered on Github Actions:

```
gh workflow run --ref <your-branch-name> nf-core-pixelator-tests.yml
```


### Pixelator benchmark tests

Sometimes it is useful to be able to run micro-benchmarks to see how changes affect performance.
Pixelator uses `pytest-benchmark` to enable running micro-benchmarks. Normally, when running the tests these are disabled.

To run these tests use:

```shell
task test-benchmark
```

### Pytest markers

Pixelator defines several additional markers for tests.
These are usually disabled by default and can be enabled by running `pytest -m <marker> tests/`

- integration_test: Marks a test as an integration test, which is often slow
- workflow_test: Marks a test as a complete pixelator workflow, which is extremely slow
- external_workflow_test: Marks a test as a complete pixelator workflow that requires external data, which is extremely slow and requires additional setup before running
- web_test: Marks a test as a browser integration test, which requires a playwright browser to be installed.
            Additionally, the full pipeline is run on the micro testdata to generate reports.


All `external_workflow_test` tests are also `workflow_tests`.

The default configuration that is applied when just running `pytest`,
is to run all unmarked tests and `integration_tests`.

You can use the pytest `-m` flag to select tests based on these markers.
e.g.

```shell
# Only internal workflow_tests
pytest -m "workflow_test and not external_workflow_test"

# All test except external workflow tests
pytest -m "not external_workflow_test"
```

You can list all available markers using:

```shell
pytest --markers
```

## Understanding the pixelator plugin system
Pixelator has a plugin system that allows you to extend the functionality of pixelator.
This section outlines how to create new plugins and make sure these are picked-up
by pixelator at runtime.

### CLI plugins

One part of the pixelator plugin system allows you to add groups of commands to the
pixelator command-line interface.

This uses the ability of the python package [Click](https://click.palletsprojects.com/en/8.1.x/)
to add groups of commands to the command line interface. To write a plugin you need to define a
group, below is an example of a new group `my_cli_plugin` with a single command `my_command`:

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

For pixelator to find this you need to define an [entrypoint](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins).

Exactly how to do this depends on which packaging method you use to build your python package. In this [link](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins) you can find
instructions for some common scenarios. The name you need to use for your entry point is: `pixelator.cli_plugin`, so if you are using `pyproject.toml` to configure your project, it could look something like this:

```toml
[project.entry-points."pixelator.cli_plugin"]
my_plugin = "my_plugin:my_cli_plugin"
```

### Pixelator config plugins

Pixelator has a config object that contains information about assays loaded from `pixelator.resources.assays`.
Additional assays can be loaded by creating a plugin that defines the `pixelator.config_plugin` entrypoint.
This entrypoint accepts a function with the current config as a parameter:

```python
def extend_config(config: "Config"):
    """
    Plugin entrypoint for extending config

    A config object will be passed in from pixelator
    at import time and can be extended here.

    :param config: The config object to extend
    """
    assay_basedir = Path(__file__).parent / "resources" / "assays"
    config.load_assays(assay_basedir)
```

```{code-block} toml
:caption: pyproject.toml

[project.entry-points."pixelator.config_plugin"]
myplugin = "myplugin:extend_config"
```
