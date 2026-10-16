"""Utility functions supporting the pixelator command line interface.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from functools import wraps
from pathlib import Path
from typing import List, Optional

import click

logger = logging.getLogger(__name__)


def click_echo(msg: str, multiline: bool = False):
    """Print a line to the console with optional long-line wrapping.

    Args:
        msg: the message to print
        multiline: True to use text wrapping or False otherwise (default)
    """
    if multiline:
        click.echo(textwrap.fill(textwrap.dedent(msg), width=100))
    else:
        click.echo(msg)


def log_step_start(
    step_name: str,
    input_files: Optional[List[str] | str] = None,
    output: Optional[str] = None,
    **kwargs,
) -> None:
    """Add information about the start of a pixelator step to the logs.

    Args:
        step_name: name of the step that is starting
        input_files: collection of input file paths
        output: optional path to output
        **kwargs: any additional parameters that you wish to log

    Returns:
        None
    """
    from pixelator import __version__

    logger.info("Start pixelator %s %s", step_name, __version__)

    if isinstance(input_files, list):
        logger.info("Input file(s) %s", ",".join(input_files))

    if isinstance(input_files, str):
        logger.info("Input file %s", input_files)

    if output is not None:
        logger.info("Output %s", output)

    if kwargs is not None:
        params = [f"{key.replace('_', '-')}={value}" for key, value in kwargs.items()]
        logger.info("Parameters:%s", ",".join(params))


def timer(func):
    """Time the different steps of a function."""

    @wraps(func)
    def wrapper(*args, **kwds):
        """Run the wrapped function and log its execution time."""
        start_time = time.perf_counter()
        res = func(*args, **kwds)
        run_time = time.perf_counter() - start_time
        logger.info("Finished pixelator %s in %.2fs", func.__name__, run_time)
        return res

    return wrapper


def _serialize_parameter_value(param: click.Parameter, value):
    """Serialize a Click parameter value for the parameters JSON file."""
    if value is None:
        return None

    # NB: that this checks the type of the parameter, not the value
    is_path = isinstance(param.type, click.Path)
    if isinstance(value, (list, tuple)):
        if is_path:
            return [str(Path(v).resolve()) for v in value]
        return list(value)

    if is_path:
        return str(Path(value).resolve())

    return value


def write_parameters_file(
    click_context: click.Context, output_file: Path, command_path: Optional[str] = None
) -> None:
    """Write the parameters used in for a command to a JSON file.

    Args:
        click_context: the click context object
        output_file: the output file
        command_path: the command to use as command name
    """
    command_path_fixed = command_path or click_context.command_path
    parameters = click_context.command.params
    parameter_values = click_context.params

    options_data = {}
    arguments_data = {}

    for param in parameters:
        value = _serialize_parameter_value(param, parameter_values.get(str(param.name)))

        if isinstance(param, click.core.Option):
            options_data[param.opts[0]] = value
        elif isinstance(param, click.core.Argument):
            arguments_data[str(param.name)] = value

    data = {
        "cli": {
            "command": command_path_fixed,
            "options": options_data,
            "arguments": arguments_data,
        }
    }

    logger.debug("Writing parameters file to %s", str(output_file))

    with open(output_file, "w") as fh:
        json.dump(data, fh, indent=4)
