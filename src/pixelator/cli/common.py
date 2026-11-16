"""Console script for pixelator (common functions).

Copyright © 2022 Pixelgen Technologies AB.
"""

import collections
import functools
import logging.handlers
from pathlib import Path
from typing import Dict, Mapping, Optional

import click
import duckdb

BASE_DIR = str(Path(__file__).parent)
logger = logging.getLogger("pixelator.cli")

# Exit code for failures caused by insufficient resources (memory or DuckDB
# spill space). Same code as a kernel OOM kill, so workflow engines that retry
# on it (e.g. the nf-core default) will re-run the command with more resources.
# All other failures, including input errors, keep exit code 1.
EXIT_CODE_RESOURCE_EXHAUSTED = 137


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
        """Initialize the ordered Click command group."""
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(  # type: ignore
        self, ctx: click.Context
    ) -> Mapping[str, click.Command]:
        """Return a list of subcommands.

        Args:
            ctx: Click context from the command decorator.
        """
        return self.commands


# snippet adapted from
# https://stackoverflow.com/questions/46765803/python-click-multiple-group-names
class AliasedOrderedGroup(OrderedGroup):
    """Custom click.Group that supports aliases.

    Currently only supports aliases for subgroups.

    Out-of-memory failures in any subcommand exit with
    ``EXIT_CODE_RESOURCE_EXHAUSTED`` instead of 1.
    """

    def invoke(self, ctx: click.Context):
        """Invoke the command, mapping out-of-memory errors to a dedicated exit code."""
        try:
            return super().invoke(ctx)
        except (duckdb.OutOfMemoryException, MemoryError) as error:
            logger.debug("Out of memory error", exc_info=True)
            logger.error(
                "Out of memory: %s\nRe-run with more memory (--memory), or set "
                "PIXELATOR_DUCKDB_TEMP_DIR or PIXELATOR_DUCKDB_MAX_TEMP_DIR_SIZE "
                "to give DuckDB more spill space.",
                error,
            )
        # Exit outside the except block, so that the original exception is no
        # longer active and is not logged again as an unhandled exception.
        ctx.exit(EXIT_CODE_RESOURCE_EXHAUSTED)

    def group(self, *args, **kwargs):
        """Attach a click group that supports aliases.

        Args:
            args: Positional arguments passed to ``click.group``.
            kwargs: Keyword arguments passed to ``click.group``.
        """

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
