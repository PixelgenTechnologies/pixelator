"""Implements a CombiningModifier interface for a Pipeline.

A modifier must be callable and typically implemented as a class with a
__call__ method.

Copyright © 2024 Pixelgen Technologies AB.
"""

from abc import ABC, abstractmethod

from cutadapt.info import ModificationInfo
from dnaio import SequenceRecord


class CombiningModifier(ABC):
    """Interface for a modifier that combines two reads into a single read."""

    @abstractmethod
    def __call__(
        self,
        read1: SequenceRecord,
        read2: SequenceRecord,
        info1: ModificationInfo,
        info2: ModificationInfo,
    ) -> SequenceRecord | None:
        """Combine two reads into a single read."""
        pass
