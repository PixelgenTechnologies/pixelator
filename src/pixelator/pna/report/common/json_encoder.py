"""Custom JSON encoder for use in Pixelator.

Copyright © 2023 Pixelgen Technologies AB.
"""

import dataclasses
import json
from typing import Any

from pydantic import BaseModel


class PixelatorJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for use in Pixelator.

    Support pydantic models and dataclasses serialization.
    """

    def default(self, obj: Any) -> Any:
        """Return a serializable object for ``obj``.

        Dump pydantic models to json or calls the base class implementation
        for other types.

        :param obj: object to serialize
        :returns: a serializable object
        """
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)  # type: ignore
        return super().default(obj)
