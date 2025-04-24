"""Base classes for pixelator models.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pydantic

try:
    from typing import Any, Self
except ImportError:
    from typing_extensions import Self


class SampleReport(pydantic.BaseModel):
    """Base class for all pixelator reports of `single-cell` subcommands.

    :ivar sample_id: The sample id for which the report is generated.
    """

    sample_id: str

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`SampleReport` from a report file.

        :param p: The path to the report file.
        :return: A :class:`SampleReport` object.
        """
        with open(p) as fp:
            json_data = json.load(fp)

        return cls(**json_data)

    def to_json(self, **kwargs: Any) -> str:  # noqa: DOC103
        """Dump the report to a json string.

        :param kwargs: Additional arguments to pass to `json.dumps`.
        :return: The report serialized to JSON as a string.
        """
        return json.dumps(self.model_dump(mode="json"), **kwargs)

    def write_json_file(self, p: str | os.PathLike, **kwargs: Any) -> None:
        """Write a JSON serialized SampleReport to a file.

        Non-existing intermediate directories in the path will be created.

        :param p: The path to the file to write.
        :param kwargs: Additional arguments to pass to pydantics `model_dump_json`.
        """
        Path(p).resolve().parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w") as fp:
            r = self.model_dump_json(**kwargs)
            fp.write(r)
