"""Copyright © 2023 Pixelgen Technologies AB."""

from __future__ import annotations

import enum
import json
import typing
from collections import defaultdict
from typing import Iterable, Optional

from pixelator.mpx.report.common.workdir import SingleCellStageLiteral
from pixelator.mpx.report.models import CommandInfo, CommandOption
from pixelator.pna.report.common.workdir import (
    SingleCellPNAStage,
    SingleCellPNAStageLiteral,
)

CommandInfoDict: typing.TypeAlias = dict[str, CommandInfo]
CommandOptionDict: typing.TypeAlias = dict[str, CommandOption]
CommandIndexTuple: typing.TypeAlias = tuple[
    dict[str, CommandInfo], dict[str, CommandOptionDict]
]

_SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING: dict[
    SingleCellStageLiteral | SingleCellPNAStageLiteral, str | list[str]
] = {
    "amplicon": [
        "pixelator single-cell-pna amplicon",
    ],
    "demux": [
        "pixelator single-cell-pna demux",
    ],
    "collapse": [
        "pixelator single-cell-pna collapse",
    ],
    "graph": ["pixelator single-cell-pna graph"],
    "layout": [
        "pixelator single-cell-pna layout",
    ],
    "analysis": [
        "pixelator single-cell-pna analysis",
    ],
    "report": [
        "pixelator single-cell-pna report",
    ],
}


class CLIInvocationInfo:
    """List of commandline invocations from a pixelator working directory.

    :ivar sample_id: The sample_id on which these commands where run
    """

    def __init__(self, data: list[CommandInfo], sample_id: Optional[str]):
        """Initialize the CLIInvocationInfo object.

        :param data: The commandline invocations
        :param sample_id: The sample_id on which these commands where run
        """
        self.sample_id = sample_id
        self._data = data
        self._commands_index, self._options_index = self._index_parameter_info(data)

    def __iter__(self) -> typing.Iterator[CommandInfo]:
        """Iterate over the commandline invocations."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of commandline invocations."""
        return len(self._data)

    def __getitem__(self, index) -> CommandInfo:
        """Return the commandline invocation at the given index."""
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

    def get_stage(
        self, stage: SingleCellPNAStage | SingleCellPNAStageLiteral
    ) -> CommandInfo | None:
        """Return the CommandInfo object for a given stage.

        If no commandline metadata is found for the stage, None is returned.

        :param stage: The stage to return the commandline for
        :returns: The commandline info for given stage
        """
        stage = stage.value if isinstance(stage, enum.Enum) else stage
        if stage in _SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING:
            stage_key = _SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING[stage]
        else:
            stage_key = stage

        res = None

        if isinstance(stage_key, list):
            for key in stage_key:
                res = self._commands_index.get(key)
                if res:
                    break
        else:
            res = self._commands_index.get(stage_key)

        return res

    def get_option(
        self, stage: SingleCellPNAStage | SingleCellPNAStageLiteral, option: str
    ) -> CommandOption:
        """Return the commandline options used to invoke a pixelator command.

        :param stage: The stage to return the commandline for
        :param option: The option to return. eg. "--design"
        :raises KeyError:
            If no commandline metadata is found for the stage,
            or the option does not exist for that stage.
        :returns CommandOption: The commandline option for the stage.
        """
        stage = stage.value if isinstance(stage, enum.Enum) else stage

        if stage in _SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING:
            stage_key = _SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING[stage]
        else:
            stage_key = stage

        stage_dict = None

        if isinstance(stage_key, list):
            for key in stage_key:
                stage_dict = self._options_index.get(key)
                if stage_dict:
                    break

        else:
            stage_dict = self._options_index.get(stage_key)

        if stage_dict is None:
            stage_key = stage_key[0] if isinstance(stage_key, list) else stage_key
            raise KeyError(f"No commandline metadata found for stage: {stage_key}")

        return stage_dict[option]

    def to_json(self, **json_options) -> str:
        """Return a JSON formatted string of the commandline invocations.

        :param json_options: Additional options to pass to json.dumps
        """
        return json.dumps(
            [v.model_dump(mode="json") for v in self._data], **json_options
        )
