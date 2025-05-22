"""Common click CLI helpers for the Pixelator PNA CLI.

Copyright © 2024 Pixelgen Technologies AB
"""

import functools
import logging
from pathlib import Path

import click

from pixelator.common.utils.units import parse_size


def output_option(func):
    """Wrap a Click entrypoint to add the --output option."""

    @click.option(
        "--output",
        required=True,
        type=click.Path(exists=False),
        help=(
            "The path where the results will be placed (it is created if it does not"
            " exist)"
        ),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def threads_option(func):
    """Decorate a click command and add the --design option."""

    @click.option(
        "--threads",
        default=-1,
        required=False,
        show_default=True,
        help="The number of total worker threads available for parallel processing",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _memory_validator(ctx, param, value):
    if value is None:
        return None
    try:
        return int(parse_size(value))
    except ValueError as exc:
        raise click.BadParameter(
            "--memory option must be a positive integer, optionally with a unit suffix [K, M, G]"
        )


def memory_option(func):
    """Decorate a click command and add the --memory option."""

    @click.option(
        "--memory",
        default=None,
        required=False,
        callback=_memory_validator,
        show_default=True,
        help="The maximum amount of memory available for processing",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def design_option(func):
    """Decorate a click command and add the --design option."""
    from pixelator.pna.config import pna_config

    # TODO: Support assay aliases here as well
    #   implement pna_config.list_assay_names()
    assay_options = list(pna_config.assays.keys())

    @click.option(
        "--design",
        required=True,
        default=None,
        type=click.Choice(assay_options),
        help="The design to load from the configuration file",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def validate_panel(ctx, param, value):
    """Validate the panel commandline option.

    :param ctx: The click context
    :param param: The click parameter
    :param value: The click value
    :returns: The validated value
    """
    try:
        if Path(value).exists():
            return value
    except Exception:
        pass
    else:
        from pixelator.pna.config import pna_config

        panel_options = list(pna_config.list_panel_names())

        if value not in panel_options:
            raise click.UsageError(
                f"Panel {value} is not a file and is not an id of a supported panel."
            )

    return value


def panel_option(func):
    """Decorate a click command and add the --design option."""
    from pixelator.pna.config import pna_config

    # TODO: Support assay aliases here as well
    #   implement pna_config.list_assay_names()
    panel_options = list(pna_config.list_panel_names())

    @click.option(
        "--panel",
        required=True,
        default=None,
        type=click.UNPROCESSED,
        callback=validate_panel,
        help="The name of a  panel to load from the supported panels. Optionally, provide a path to a custom panel file.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


logger = logging.getLogger("pixelator.pna.cli")
