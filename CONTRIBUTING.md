# Contributing

There are many types of contributions you could do. Contributions are welcomed,
and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/PixelgenTechnologies/pixelator/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the Github issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the Github issues for features. Anything tagged with "feature", "improvement",
"enhancement" and "help wanted" is open to whoever wants to implement it.

Any new feature is tagged with "feature" and "improvement" if it is already implemented and
looking for improvements, but if the feature doesn't exist, we will tag it with "feature" and
"enhancement".

### Write Documentation

Pixelator could always use more documentation, whether as part of the
official Pixelator docs, in docstrings, or even on the web in blog posts,
articles, and such.

Docstrings are automatically checked in your code commits if you install our pre-commit hooks.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/PixelgenTechnologies/pixelator/issues

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Get Started!

Ready to contribute? Here's how to set up ``pixelator`` for local development.

1. Fork the `pixelator` repo on Github.

2. Clone your fork locally.

    ```shell
    git clone git@github.com:yourgithubuser/pixelator.git
    ```

    where your `yourgithubuser` is the user/org where you have cloned the repo.

3. Configure your environment and install pixelator.

   We recommend that you follow the instructions in [DEVELOPERS.md](DEVELOPERS.md).

4. Create a branch for local development.

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```shell
    tox
    ```

6. Commit your changes and push your branch to GitHub:

    ```shell
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Testing

Make sure **all** your new tests and existing tests pass before you create a PR.

```shell
pytest .
```

To run a subset of tests (i.e. on a figured `tests.test_pixelator` module):

```shell
pytest tests.test_pixelator
```

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds some command-line functionality, the docs should
   be updated. Put your new functionality into a function, and add the
   feature to the list in [USAGE.md](USAGE.md). If your functionality are
   new methods or algorithms, please, include detailed docstrings.
3. The pull request should work for Python 3.8, 3.9 and 3.10.
   Make sure that the tests pass for all supported Python versions with `tox`.

For a good standardization with commit messages in the project, we try to use semantic commits (e.g.
[Conventional Commits](https://www.conventionalcommits.org/en/)) but this is not
enforced.

You can find a full list of items we regularly ask for in PRs in our [PULL_REQUEST_TEMPLATE.md](./.github/PULL_REQUEST_TEMPLATE.md).

## Install pre-commit hooks

We strongly recommend installing the pre-commit hooks when contributing. Run `pre-commit install` to set up the git hook scripts so pre-commit will run automatically on each `git commit`.

```shell
pre-commit install
#pre-commit installed at .git/hooks/pre-commit
```
