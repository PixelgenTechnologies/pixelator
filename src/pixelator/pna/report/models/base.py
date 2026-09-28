"""Base classes for pixelator models.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pydantic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class SampleReport(pydantic.BaseModel):
    """Base class for all pixelator reports of `single-cell` subcommands.

    Attributes:
        sample_id: The sample id for which the report is generated.
        product_id: The product id for which the report is generated.
        report_type: The command for which the report is generated.
        status: ``passed`` or ``failed``. JSON omits this field when the step passed.
        null_reason: Why a failed step produced a null file. JSON includes this
            string only when ``status`` is ``failed``.
    """

    sample_id: str
    product_id: Literal["single-cell-pna"]
    report_type: str
    status: Literal["passed", "failed"] = "passed"
    null_reason: str | None = None

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize a SampleReport from a report file.

        Args:
            p: The path to the report file.

        Returns:
            A :class:`~pixelator.pna.report.models.base.SampleReport` object.
        """
        with open(p) as fp:
            json_data = json.load(fp)

        return cls(**json_data)

    def _json_payload(self) -> dict[str, Any]:
        """Serialize the report, omitting a passed status and an empty reason.

        Successful reports stay compatible with existing JSON. A failed step
        keeps ``status`` and ``null_reason``.
        """
        data = self.model_dump(mode="json")
        if data.get("status") == "passed":
            data.pop("status", None)
        if data.get("null_reason") is None:
            data.pop("null_reason", None)
        return data

    def to_json(self, **kwargs: Any) -> str:  # noqa: DOC103
        """Dump the report to a json string.

        Args:
            kwargs: Additional arguments to pass to `json.dumps`.

        Returns:
            The report serialized to JSON as a string.
        """
        return json.dumps(self._json_payload(), **kwargs)

    def write_json_file(self, p: str | os.PathLike, **kwargs: Any) -> None:
        """Write a JSON serialized SampleReport to a file.

        Non-existing intermediate directories in the path will be created.

        Args:
            p: The path to the file to write.
            kwargs: Additional arguments to pass to `json.dumps`.
        """
        Path(p).resolve().parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w") as fp:
            fp.write(self.to_json(**kwargs))
