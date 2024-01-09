from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Self

import pandas as pd

from pixelator.report.models.base import BaseSampleDataModel
from collections import Counter


logger = logging.getLogger(__name__)


class SampleMetadataRecord(BaseSampleDataModel):
    #: A description of the sample
    description: str

    #: Semantic version string of a panel file
    panel_version: str

    #: The name of the panel
    panel_name: str

    @classmethod
    def from_csv(cls, p: Path) -> Self:
        data = pd.read_csv(p, sep=",", index_col=0)


###


class SampleMetadata:
    """A collection of per sample metadata
    :param metadata: An iterable of :class:`SampleMetadata` objects
    """

    def __init__(self, metadata: Iterable[SampleMetadataRecord] = ()):
        self._data = list(metadata)
        self._lookup = {v.sample_id: v for v in self._data}
        self._validate_data()

    def get_by_id(self, sample_id: str) -> SampleMetadataRecord | None:
        """Retrieve a metadata record from a sample id

        :param sample_id: The sample id to retrieve metadata for
        """
        return self._data.get(sample_id)

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
        logger.debug("Parsing metadata from %s", str(p))
        data = pd.read_csv(p, sep=",", index_col=0)
        records = []

        for record_name, record in data.iterrows():
            records.append(SampleMetadataRecord(**record.to_dict()))

        return cls(records)
