"""
Console script for pixelator (common functions)

Copyright © 2022 Pixelgen Technologies AB.
"""

import collections
import functools
import logging.handlers
from pathlib import Path
from typing import Dict, Mapping, Optional

import click

BASE_DIR = str(Path(__file__).parent)
logger = logging.getLogger("pixelator.cli")


# code snippet obtained from
# https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
# the purpose is to order subcommands in order of addition
class OrderedGroup(click.Group):
    """Custom click.Group that keeps insertion order for subcommands."""

    def __init__(  # noqa: D107
        self,
        name: Optional[str] = None,
        commands: Optional[Dict[str, click.Command]] = None,
        **kwargs,
    ):
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(  # type: ignore
        self, ctx: click.Context
    ) -> Mapping[str, click.Command]:
        """Return a list of subcommands."""
        return self.commands


# snippet adapted from
# https://stackoverflow.com/questions/46765803/python-click-multiple-group-names
class AliasedOrderedGroup(OrderedGroup):
    """Custom click.Group that supports aliases.

    Currently only supports aliases for subgroups.
    """

    def group(self, *args, **kwargs):
        """Attach a click group that supports aliases."""

        def decorator(f):
            aliased_group = []
            aliases = kwargs.pop("aliases", [])
            main_group = super(AliasedOrderedGroup, self).group(*args, **kwargs)(f)

            for alias in aliases:
                grp_kwargs = kwargs.copy()
                del grp_kwargs["name"]
                grp = super(AliasedOrderedGroup, self).group(
                    alias, *args[1:], **grp_kwargs
                )(f)
                grp.short_help = "Alias for '{}'".format(main_group.name)
                aliased_group.append(grp)

            # for all the aliased groups, link to all attributes from the main group
            for aliased in aliased_group:
                aliased.commands = main_group.commands
                aliased.params = main_group.params
                aliased.callback = main_group.callback
                aliased.epilog = main_group.epilog
                aliased.options_metavar = main_group.options_metavar
                aliased.add_help_option = main_group.add_help_option
                aliased.no_args_is_help = main_group.no_args_is_help
                aliased.hidden = main_group.hidden
                aliased.deprecated = main_group.deprecated

            return main_group

        return decorator


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


def design_option(func):
    """Decorate a click command and add the --design option."""
    from pixelator.mpx.config import config

    assay_options = list(config.assays.keys())

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
