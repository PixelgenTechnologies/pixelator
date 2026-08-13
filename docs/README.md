# Docs handbook

## Overview

This document describes how the Pixelator documentation is generated, what is
included, how it is maintained, and how it is built and deployed.

Key files:

- `docs/conf.py` — Sphinx configuration (extensions, AutoAPI, theme, cross-referencing, warning handling).
- `docs/index.rst` — landing page and the root table of contents.
- `docs/api/index.rst` — API reference table of contents (AutoAPI-generated tree).
- `docs/api/overview.rst` — curated list of primary API entry points.
- `docs/cli/index.rst` — CLI reference.
- `docs/_static/` — custom CSS and logo.
- `docs/_templates/autoapi/` — AutoAPI template overrides (`autoapi_template_dir`); `python/class.rst` sets `py:currentmodule` and uses `qual_name` on own-page class docs so shortened same-module `Bases:` links resolve (imported-member pages also re-qualify local bases with the defining module).
- `docs/_scripts/update_switcher.py` — regenerates the version switcher during deployment.
- `.github/workflows/deploy-docs.yml` — the workflow that builds and deploys the docs.

## Documentation generator

- The generation of the docs is implemented using [Sphinx](https://www.sphinx-doc.org/en/master/) with extensions listed in `extensions` in `docs/conf.py`.
- Documentation is automatically generated from docstrings and help-text (CLI) in the source code of `pixelator`.
- Docstrings are formatted according to [Google conventions for docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Where possible, internal names have cross-referencing pointing to a page in the docs, and external names map to the public external documentation site. Links to stable external documentation are kept in `intersphinx_mapping` in `docs/conf.py`.
- The `__all__` lists in the package `__init__.py` files determine which name-links are shown on the parent package page in the API reference.
- The CLI docs use [sphinx-click](https://sphinx-click.readthedocs.io/en/latest/). Only the `single-cell-pna` command tree is rendered, selected via `:commands: single-cell-pna` in `docs/cli/index.rst`.
- The site uses the [PyData Sphinx theme](https://pydata-sphinx-theme.readthedocs.io/) with a version switcher, configured in `html_theme_options` in `docs/conf.py` (the switcher reads `switcher.json`; see "Deployment").



## Inclusion

`autoapi` inclusion is mainly determined by `autoapi_options` and `autoapi_ignore` in `docs/conf.py`. Some options worth mentioning include:

- Imported members (`imported-members`) are intentionally included in the docs generation via `autoapi_options`.
- Private members (`private-members`) are not included via `autoapi_options`.
- Parts of the codebase, e.g. names related to MPX, are ignored via `autoapi_ignore`.



## Warnings and strict builds

- Builds run in nitpicky mode (`nitpicky = True` in `docs/conf.py`, and the `-n` flag), so unresolved cross-references are reported as warnings. CI builds additionally pass `--fail-on-warning`, which means any warning fails the build.
- Known-acceptable warnings are suppressed:
  - `nitpick_ignore_regex` silences specific cross-reference targets that cannot be resolved (e.g. MPX names).
  - `suppress_warnings` silences whole Sphinx warning categories (`ref.python`, `autoapi.python_import_resolution`).
  - A logging filter (`_keep_warning`) drops a small set of warnings that are emitted without a Sphinx type/subtype and so cannot be matched by `suppress_warnings`.
- When new code introduces a cross-reference warning, prefer fixing the reference.



## Maintenance

- Update or add docstrings and CLI help-text in the source code as the code changes.
- The overview page (`overview.rst`) lists some primary entry points of the API. Additions to this page are made manually.
- `intersphinx_mapping` in `docs/conf.py` lists the URLs for documentation of external names used for cross-referencing. This list is maintained manually.
- Keep this handbook (`docs/README.md`) up to date when the docs setup changes.



## Deployment

- Building and deploying the docs is handled by the "Docs" workflow (`.github/workflows/deploy-docs.yml`).
- The workflow runs on three triggers: published releases, manual dispatch (`workflow_dispatch`, which requires a `tag` input), and pull requests.
- The build job runs for all three triggers and builds with `sphinx-build ... --fail-on-warning -n`, so warnings fail the build (see "Warnings and strict builds").
- The deploy job only runs for manual dispatch and stable releases. It does not run on pull requests nor prereleases.
- Deployment publishes the built HTML to the `gh-pages` branch under `docs/<version>/` via [GitHub Pages](https://docs.github.com/en/pages) (using `peaceiris/actions-gh-pages`). The version is the release tag without the leading `v` (or the dispatch `tag`), defaulting to `latest` when unset.
- After publishing, `docs/_scripts/update_switcher.py` regenerates `switcher.json` (the version list used by the theme's version switcher) and a redirect at `docs/index.html` that points to the newest version.
- GitHub Pages must be configured to serve from the `gh-pages` branch with the `/docs` folder as the source (repo Settings → Pages).



## Development

1. Install the docs dependency group: `uv sync --group docs`.
2. Build the HTML from a clean state (logging warnings to a file):

```
rm -rf docs/_build docs/api/generated && uv run sphinx-build -b html docs docs/_build/html --keep-going -n -w docs/_build/sphinx-warnings.log
```

3. Open `docs/_build/html/index.html`.

Notes:

- The command above uses `--keep-going` and logs warnings to `docs/_build/sphinx-warnings.log` so you can review them. CI instead uses `--fail-on-warning`.
- For a live-reloading preview while editing, `sphinx-autobuild` is included in the docs dependency group: `uv run sphinx-autobuild docs docs/_build/html`.



## References

- [Sphinx documentation](https://www.sphinx-doc.org/en/master/)
- [AutoAPI documentation](https://sphinx-autoapi.readthedocs.io/en/latest/index.html)
- [sphinx-click](https://sphinx-click.readthedocs.io/en/latest/)
- [PyData Sphinx theme](https://pydata-sphinx-theme.readthedocs.io/)
- [Google docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
