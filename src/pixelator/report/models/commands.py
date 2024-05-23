"""Module for handling commandline metadata.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import logging
import typing
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar

import click
import pydantic

logger = logging.getLogger(__name__)
CommandOptionValue = TypeVar(
    "CommandOptionValue", None, str, int, float, bool, list[str]
)


# -----------------------------------------------------------------------
# Some helper functions to deal with commandline metadata
# -----------------------------------------------------------------------


def _clean_commmand_path(data: Mapping[str, Any]) -> str:
    """Clean up the command path from a parsed meta.json file.

    :param data: The parsed meta.json file
    :returns str: The cleaned command.
    """
    command_path = data["cli"]["command"].split(" ")
    return " ".join(command_path)


def _find_click_command(
    click_context: click.Group, data: Mapping[str, Any]
) -> click.Command:
    """Find the click command for a given parsed meta.json file."""
    command_path = data["cli"]["command"].split(" ")
    command_group: click.Group = click_context

    clean_command_path = _clean_commmand_path(data).split(" ")

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

    :param data: The parsed meta.json file
    :return: A :class:`CommandInfo` object.
    """
    from pixelator.cli import main_cli

    leaf_command = _find_click_command(main_cli, data)
    param_data: list[CommandOption] = []
    opt_lookup = {p.opts[0]: p for p in leaf_command.params}
    clean_command_name = _clean_commmand_path(data)

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

    name: str = pydantic.Field(
        description="The name of the option. (e.g. `--min-reads`)"
    )

    value: CommandOptionValue = pydantic.Field(description="The value of the option.")
    default_value: Optional[CommandOptionValue] = pydantic.Field(
        description="The default value for this option."
    )
    description: str | None = pydantic.Field(
        description="The description of the option."
    )


class CommandInfo(pydantic.BaseModel):
    """Dataclass for passing all options of a command to qc report."""

    command: str = pydantic.Field(description="The name of the command.")

    options: list[CommandOption] = pydantic.Field(
        description="The list of options for the command."
    )

    @classmethod
    def from_json(cls, p: Path) -> CommandInfo:
        """Initialize an :class:`CommandInfo` from a json file.

        :param p: The path to the report file.
        :return: A :class:`CommandInfo` object.
        """
        with open(p, "r") as fh:
            file_data = json.load(fh)

        command_info = _process_meta_json_data(file_data)
        return command_info
