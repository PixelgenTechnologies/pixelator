"""Miscellaneous CLI functions for the Pixelator PNA package.

Copyright © 2024 Pixelgen Technologies AB
"""

from typing import Any

import click

from pixelator.common.utils import click_echo


def list_single_cell_pna_designs(ctx: click.Context, param: Any, value: Any) -> None:
    """Return a list of single cell designs supported by the config.

    Args:
    ctx: The click context
    param: The click parameter
    value: The click value
    """
    from pixelator.pna.config import pna_config

    if not value or ctx.resilient_parsing:
        return

    options = list(pna_config.assays.keys())
    for option in options:
        click.wrap_text(option)
        click_echo(option)

    ctx.exit()


def list_single_cell_pna_panels(
    ctx: click.Context, param: Any, value: Any, include_archived: bool = False
) -> None:
    """Return a list of single cell panels supported by the config.

    Args:
    ctx: The click context
    param: The click parameter
    value: The click value
    include_archived: Include archived.
    """
    from pixelator.pna.config import pna_config

    if not value or ctx.resilient_parsing:
        return

    options = pna_config.list_panel_names(
        include_aliases=True, include_archived=include_archived
    )
    for option in options:
        click_echo(option)

    ctx.exit()


def list_single_cell_pna_panels_including_archived(
    ctx: click.Context, param: Any, value: Any
) -> None:
    """Return a list of single cell panels supported by the config, including archived panels.

    Args:
    ctx: The click context
    param: The click parameter
    value: The click value
    """
    list_single_cell_pna_panels(ctx, param, value, include_archived=True)
