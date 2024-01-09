from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Type, TypeVar

import pydantic

from pixelator.utils import get_sample_name


try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class StageReport(pydantic.BaseModel):
    sample_id: str

    @classmethod
    @abstractmethod
    def from_json(cls, p: Path) -> Self:
        ...


class BaseSampleDataModel(pydantic.BaseModel):
    sample_id: str

    @classmethod
    @abstractmethod
    def from_json(cls, p: Path) -> Self:
        ...
