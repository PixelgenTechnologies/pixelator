from __future__ import annotations

import enum
import typing

from typing import Optional, Iterable
from collections import defaultdict

from pixelator.report.models import CommandInfo, CommandOption
from pixelator.report.workdir import SingleCellStage, SingleCellStageLiteral

CommandInfoDict: typing.TypeAlias = dict[str, CommandInfo]
CommandOptionDict: typing.TypeAlias = dict[str, CommandOption]
CommandIndexTuple: typing.TypeAlias = tuple[
    dict[str, CommandInfo], dict[str, CommandOptionDict]
]


class CLIInvocationInfo(Iterable[CommandInfo]):
    """list of commandline invocations from a pixelator working directory.

    :param data: A list of CommandInfo objects
    :param sample_id: The sample_id on which these command where run
    """

    def __init__(self, data: list[CommandInfo], sample_id: Optional[str] = None):
        self.sample_id = sample_id
        self._data = data
        self._commands_index, self._options_index = self._index_parameter_info(data)

    def __iter__(self) -> Iterable[CommandInfo]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> CommandInfo:
        return self._data[index]

    @staticmethod
    def _index_parameter_info(data: list[CommandInfo]) -> CommandIndexTuple:
        """Create two lookup tables for querying parameter info.

        :param data: the result from `generate_parameter_info`
        :returns: A tuple with two lookup tables
        :rtype: Tuple[dict[str, CommandInfo], dict[str, dict[str, CommandOption]]]
        """
        command_index = {}
        command_option_index: dict[str, dict[str, CommandOption]] = defaultdict(dict)

        for command_info in data:
            command_index[command_info.command] = command_info
            for option in command_info.options:
                command_option_index[command_info.command][option.name] = option

        return command_index, command_option_index

    def get_option(
        self, stage: SingleCellStage | SingleCellStageLiteral, option: str
    ) -> CommandOption:
        """Return the commandline options used to invoke a pixelator command.

        :param sample: The sample to return the commandline for
        :raises KeyError:
            If no commandline metadata is found for the stage,
            or the option does not exist for that stage.
        :returns CommandOption: The commandline option for the stage.
        """
        stage_key = stage.value if isinstance(stage, enum.Enum) else stage
        stage_dict = self._options_index.get(stage_key)
        if stage_dict is None:
            raise KeyError(f"No commandline metadata found for stage: {stage_key}")

        return stage_dict[option]
