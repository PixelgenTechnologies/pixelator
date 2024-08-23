"""Main console script for pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

import atexit
import multiprocessing
import sys

import click
import yappi

from pixelator import __version__
from pixelator.cli.adapterqc import adapterqc
from pixelator.cli.amplicon import amplicon
from pixelator.cli.analysis import analysis
from pixelator.cli.annotate import annotate
from pixelator.cli.collapse import collapse
from pixelator.cli.common import AliasedOrderedGroup, logger
from pixelator.cli.demux import demux
from pixelator.cli.graph import graph
from pixelator.cli.layout import layout
from pixelator.cli.misc import list_single_cell_designs, list_single_cell_panels
from pixelator.cli.plugin import add_cli_plugins
from pixelator.cli.preqc import preqc
from pixelator.cli.report import report
from pixelator.logging import LoggingSetup
from pixelator.utils import click_echo


@click.group(cls=AliasedOrderedGroup, name="pixelator")
@click.version_option(__version__)
@click.option(
    "--verbose",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="Show extended messages during execution",
)
@click.option(
    "--profile",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="Activate profiling mode",
)
@click.option(
    "--log-file",
    required=False,
    default=None,
    type=click.Path(exists=False),
    help="The path to the log file (it is created if it does not exist)",
)
@click.option(
    "--cores",
    default=max(1, multiprocessing.cpu_count() - 1),
    required=False,
    type=click.INT,
    show_default=True,
    help="The number of cpu cores to use for parallel processing",
)
@click.pass_context
def main_cli(ctx, verbose: bool, profile: bool, log_file: str, cores: int):
    """Run the main CLI entrypoint for pixelator."""
    # early out if run in help mode
    if any(x in sys.argv for x in ["--help", "--version"]):
        return 0

    if verbose:
        logger.info("Running in VERBOSE mode")

    # activate profiling mode
    if profile:
        logger.info("Running in profiling mode")
        yappi.start()

        def exit():
            yappi.stop()
            logger.info("Profiling completed")
            processes = yappi.get_thread_stats()
            # Make sure to get profile metrics for each thread
            for p in processes:
                click_echo(f"Function stats for {p.name} {p.id}")
                yappi.get_func_stats(ctx_id=p.id).print_all()

        atexit.register(exit)

    # Pass arguments to other commands
    ctx.ensure_object(dict)

    # This registers the logger with it's context manager,
    # so that it is clean-up properly when the command is done.
    ctx.obj["LOGGER"] = ctx.with_resource(LoggingSetup(log_file, verbose=verbose))
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["CORES"] = max(1, cores)
    return 0


@main_cli.group(name="single-cell")
@click.option(
    "--list-designs",
    is_flag=True,
    metavar="",
    is_eager=True,
    expose_value=False,
    required=False,
    callback=list_single_cell_designs,
    help="List available designs and exit.",
)
@click.option(
    "--list-panels",
    is_flag=True,
    metavar="",
    is_eager=True,
    expose_value=False,
    required=False,
    callback=list_single_cell_panels,
    help="List available panels and exit.",
)
def single_cell_mpx():
    """Build the click group for single-cell commands."""


# Add single-cell top level command to cli
main_cli.add_command(single_cell_mpx)

# Add single-cell commands
single_cell_mpx.add_command(amplicon)
single_cell_mpx.add_command(preqc)
single_cell_mpx.add_command(adapterqc)
single_cell_mpx.add_command(demux)
single_cell_mpx.add_command(collapse)
single_cell_mpx.add_command(graph)
single_cell_mpx.add_command(annotate)
single_cell_mpx.add_command(layout)
single_cell_mpx.add_command(analysis)
single_cell_mpx.add_command(report)

# Add cli plugins as commands on top level
add_cli_plugins(main_cli)

if __name__ == "__main__":
    sys.exit(main_cli())  # pragma: no cover
