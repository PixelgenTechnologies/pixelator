"""Model for sample metadata.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import pydantic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class SampleMetadataRecord(pydantic.BaseModel):
    """Model for sample metadata.

    :ivar sample_id: The sample id
    :ivar description: A description of the sample
    :ivar panel_version: Semantic version string of a panel file
    :ivar panel_name: The name of the panel
    """

    sample_id: str = pydantic.Field(description="The sample id")

    description: str = pydantic.Field(description="A description of the sample")

    panel_version: str = pydantic.Field(
        description="Semantic version string of a panel file"
    )

    panel_name: str = pydantic.Field(description="The name of the panel")


class SampleMetadata:
    """Metadata for a collection of samples."""

    def __init__(self, metadata: Iterable[SampleMetadataRecord] = ()):
        """Create a SampleMetadata object.

        :param metadata: An iterable of :class:`SampleMetadataRecord` objects.
        """
        self._data = list(metadata)
        self._lookup = {v.sample_id: v for v in self._data}
        self._validate_data()

    def get_by_id(self, sample_id: str) -> SampleMetadataRecord | None:
        """Retrieve a metadata record from a sample id.

        :param sample_id: The sample id to retrieve metadata for
        :returns: The metadata record for the sample id or None if not found
        """
        return self._lookup.get(sample_id)

    def _validate_data(self):
        """Check for duplicate sample ids."""
        if len(self._data) != len(self._lookup.keys()):
            id_counts = Counter((v.sample_id for v in self._data))
            for sample_id, count in id_counts.items():
                if count > 1:
                    raise ValueError(
                        f"Every sample must have a unique id: duplicate "
                        f"sample id: {sample_id}"
                    )

    @classmethod
    def from_csv(cls, p: Path | str) -> Self:
        """Create a SampleMetadata object from a csv file.

        :arg p: The path to the csv file.
        :returns: A SampleMetadata object.
        """
        logger.debug("Parsing metadata from %s", str(p))
        data = pd.read_csv(
            p,
            sep=",",
            index_col=0,
            dtype={
                "sample_id": str,
                "description": str,
                "panel_version": str,
                "panel_name": str,
            },
        )
        records = []

        for record_name, record in data.iterrows():
            records.append(SampleMetadataRecord(**record.to_dict()))

        return cls(records)
