# Versioning Pixelator

## Releasing

A reminder for the maintainers on how to release/deploy.

- [ ]  Make sure all your changes are committed in the `dev` branch
- [ ]  Check out a new branch `release-<version>` from the `dev`-branch
    - [ ]  Write properly written release notes
(including an entry in [CHANGELOG.md](./CHANGELOG.md)) and commit them
    - [ ]  In one commit, make sure that the entry `version` in `pyproject.toml` is updated with the correct version and
that `tool.poetry-dynamic-versioning` is disabled.
        - [ ]  Set the `version` in `pyproject.toml` to the version you want to release
        - [ ]  Set `tool.poetry-dynamic-versioning` to `false`
    - [ ]  Create a build by running `poetry build`
        - [ ]  Try installing this the new version by running `pip install dist/<your build>`
        - [ ]  Check that the version installed is correct by running `pixelator --version`
    - [ ]  Create the git tag locally either unsigned:
        - [ ]  `git tag -a "v<version e.g. 0.11.0>" -m "Release <your number e.g. 0.11.0>`

        Or signed (requires that you have proper keys setup):

        - [ ]  `git tag -s -a "v<version e.g. 0.11.0>" -m "Release <your number e.g. 0.11.0>`
    - [ ]  Run `git tag -l`  to check that your new tag is here
    - [ ]  Push your branch and tags to github `git push --set-upstream origin --tags release-<version>`
- [ ]  Make a PR from `release-<version>` to `main` with all the changes.
- [ ]  Get someone to approve it for you
- [ ]  Add a release in the Github web interface to the tag you just created. The name should be `Pixelator v<your version>`. Paste the change long information as the notes. Set it as the latest release (unless it’s a pre-release).
- [ ]  Create a new branch from `main` to `dev` to sync them
    - [ ]  Get someone to approve it
    - [ ]  Merge the PR

The CI/CD will then perform the following if tests pass:

- Create and release a Docker image

## Building the PyPI package

You can create a distribution package with the following command:

    ```shell
    python -m build
    ```

## Publishing on PyPI

Building and publishing Python packages to PyPI is [extremely simple](https://python-poetry.org/docs/libraries/#publishing-to-pypi) with poetry.

You will upload the package to the PyPI production server by entering Pixelgen Technologies credentials for one of
the maintainer accounts registered. Only maintainers should be able to do this.

    ```shell
    poetry publish
    ```
Check that you can install pixelator from the real PyPI:

    ```shell
    python -m pip install pixelator.
    ```

## Building the conda package

Update the version number to the pixelator [bioconda recipe](./conda-recipe/pixelator/meta.yaml).

First create a source package with `poetry build`.

Then run:

    ```shell
    conda build -c conda-forge -c bioconda ./conda-recipe/pixelator
    ```

You can install the local package with conda/mamba using:

    ```shell
    conda install -c conda-forge -c bioconda --use-local pixelator
    ```

You may have to update the meta.yaml file to match the current release version.

## Publishing the conda package

TBD

## Publishing container images (CI/CD)

You can build and push a production image tagged with the version specified by  the `git tag`
command.

The GitHub pipeline (.github/workflows/build.yml) is configured to build and publish
these images automatically when a new git tag is added.

Each PR will also build its own container with a label that matches the branch name.

For every commit to dev a production container tagged with `dev-$(git rev-parse --short HEAD)` will be created as well.
