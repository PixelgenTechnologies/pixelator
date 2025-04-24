"""Helper commands for pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

from typing import Any

import click

from pixelator.common.utils import click_echo


def list_single_cell_designs(ctx: click.Context, param: Any, value: Any) -> None:
    """Return a list of single cell designs supported by the config.

    :param ctx: The click context
    :param param: The click parameter
    :param value: The click value
    """
    from pixelator.mpx.config import config

    if not value or ctx.resilient_parsing:
        return

    options = list(config.assays.keys())
    for option in options:
        click_echo(option)

    ctx.exit()


def list_single_cell_panels(ctx: click.Context, param: Any, value: Any) -> None:
    """Return a list of single cell panels supported by the config.

    :param ctx: The click context
    :param param: The click parameter
    :param value: The click value
    """
    from pixelator.mpx.config import config

    if not value or ctx.resilient_parsing:
        return

    options = config.list_panel_names(include_aliases=True)
    for option in options:
        click_echo(option)

    ctx.exit()
