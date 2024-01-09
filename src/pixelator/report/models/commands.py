from __future__ import annotations

import json
import logging
import typing
from typing import TypeVar, Mapping, Any, Self

from pathlib import Path
import click
import pydantic

logger = logging.getLogger(__name__)
CommandOptionValue = TypeVar("CommandOptionValue", None, str, int, float, bool)


# -----------------------------------------------------------------------
# Some helper functions to deal with commandline metadata
# -----------------------------------------------------------------------


def _clean_commmand_path(click_context: click.Group, data: Mapping[str, Any]) -> str:
    """Clean up the command path from a parsed meta.json file.

    This is a workaround for weird behaviour when using ctx.invoke in the old pipeline command.
    It can be phased out but is still useful to load old reports.
    """
    command_path = data["cli"]["command"].split(" ")

    # This is a hack to rewrite the weird command_path when using ctx.invoke
    # with the pipeline command
    rnd_pipeline_path = ("rnd", "pipeline", "CSV")
    pipeline_path = ("pipeline", "CSV")

    runs_from_pipeline_ctx = pipeline_path in zip(command_path, command_path[1:])
    runs_from_rnd_pipeline_ctx = rnd_pipeline_path in zip(
        command_path, command_path[1:], command_path[2:]
    )

    if runs_from_pipeline_ctx or runs_from_rnd_pipeline_ctx:
        path_to_remove = (
            rnd_pipeline_path if runs_from_rnd_pipeline_ctx else pipeline_path
        )
        command_path = [part for part in command_path if part not in path_to_remove]
        if runs_from_rnd_pipeline_ctx:
            command_path.insert(1, "single-cell")

    return " ".join(command_path)


def _find_click_command(
    click_context: click.Group, data: Mapping[str, Any]
) -> click.Command:
    """Find the click command for a given parsed meta.json file."""
    command_path = data["cli"]["command"].split(" ")
    command_group: click.Group = click_context

    clean_command_path = _clean_commmand_path(click_context, data).split(" ")

    if len(command_path) <= 1:
        raise ValueError("Expected at least one subcommand")

    for subcommand in clean_command_path[1:-1]:
        if subcommand in command_group.commands:
            command_group = typing.cast(click.Group, command_group.commands[subcommand])
        else:
            raise ValueError(f"Unknown command {subcommand}")

    leaf_command = command_group.commands[clean_command_path[-1]]
    return leaf_command


def _process_meta_json_data(data: Mapping[str, Any]) -> CommandInfo:
    """Process a single metadata file and generate the parameter info object.

    :param click_context: The click context of the main pixelator command
    :param data: The parsed meta.json file
    :returns: A CommandInfo object
    :rtype: CommandInfo
    """
    from pixelator.cli import main_cli

    leaf_command = _find_click_command(main_cli, data)
    param_data: list[CommandOption] = []
    opt_lookup = {p.opts[0]: p for p in leaf_command.params}
    clean_command_name = _clean_commmand_path(main_cli, data)

    for param_name, param_value in data["cli"]["options"].items():
        option_info = opt_lookup.get(param_name)

        if option_info is None:
            logger.warning(
                f'Unknown parameter "{param_name}" for command: "{clean_command_name}"'
            )
            param_data.append(
                CommandOption(
                    name=param_name,
                    value=param_value,
                    default_value=None,
                    description=None,
                )
            )
            continue

        help_text = None
        if isinstance(option_info, click.Option):
            help_text = option_info.help

        param_data.append(
            CommandOption(
                name=param_name,
                value=param_value,
                default_value=option_info.default,
                description=help_text,
            )
        )

    command = CommandInfo(
        command=clean_command_name,
        options=param_data,
    )

    return command


class CommandOption(pydantic.BaseModel, typing.Generic[CommandOptionValue]):
    """Dataclass for passing command options/flags to qc report."""

    #: The name of the option (e.g. `--min-reads`)
    name: str
    #: The value of the option.
    value: CommandOptionValue
    #: The default value for this option.
    default_value: CommandOptionValue | None
    #: The help text for this option.
    description: str | None


class CommandInfo(pydantic.BaseModel):
    """Dataclass for passing all options of a command to qc report."""

    #: The name of the parameter group.
    command: str

    #: A list of options for the parameter group.
    options: list[CommandOption]

    @classmethod
    def from_json(cls, p: Path) -> Self:
        with open(p, "r") as fh:
            file_data = json.load(fh)

        command_info = _process_meta_json_data(file_data)
        return command_info
