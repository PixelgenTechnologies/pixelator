"""Tests for the AmpliconPipeline single-end pre-modifiers.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pytest
from cutadapt.info import ModificationInfo
from cutadapt.modifiers import SingleEndModifier
from dnaio import SequenceRecord

from pixelator.pna.read_processing.modifiers import CombiningModifier
from pixelator.pna.read_processing.pipeline import AmpliconPipeline


class _MockCombiner(CombiningModifier):
    """A no-op combiner used for unit testing."""

    def __call__(self, read1, read2, info1, info2):
        return read1 if read1 is not None else read2


class _TrimFrontModifier(SingleEndModifier):
    """A modifier that trims a fixed number of bases from the front of a read."""

    def __init__(self, n: int):
        self._n = n

    def __call__(self, read, info):
        return read[self._n :]


class _ReturnNoneModifier(SingleEndModifier):
    """A modifier that always returns None (simulates read filtering)."""

    def __call__(self, read, info):
        return None


def _make_read(sequence: str) -> SequenceRecord:
    return SequenceRecord(
        name="test_read",
        sequence=sequence,
        qualities="C" * len(sequence),
    )


class TestPreProcessSingle:
    """Unit tests for AmpliconPipeline._pre_process_single."""

    def _make_pipeline(self, *modifiers):
        return AmpliconPipeline(
            combiner=_MockCombiner(),
            pre_modifiers=list(modifiers),
        )

    def test_single_modifier_is_applied(self):
        """A single pre-modifier should be applied to the read."""
        pipeline = self._make_pipeline(_TrimFrontModifier(3))
        read = _make_read("ACGTACGT")

        result_read, info, n_bp = pipeline._pre_process_single(read)

        assert result_read.sequence == "TACGT"
        assert n_bp == 8

    def test_multiple_modifiers_applied_sequentially(self):
        """Multiple pre-modifiers must be applied in order, each receiving the
        output of the previous modifier.

        Regression test: before the fix, only the last modifier was applied
        (each restarted from the original read).
        """
        pipeline = self._make_pipeline(
            _TrimFrontModifier(2),
            _TrimFrontModifier(3),
        )
        read = _make_read("ACGTACGT")  # 8 bases

        result_read, info, n_bp = pipeline._pre_process_single(read)

        # With sequential application: trim 2 then trim 3 → 3 bases remain.
        # Without the fix (last modifier wins): trim 3 from original → 5 bases.
        assert result_read.sequence == "CGT"
        assert n_bp == 8

    def test_n_bp_reflects_original_length_not_trimmed(self):
        pipeline = self._make_pipeline(_TrimFrontModifier(5))
        read = _make_read("ACGTACGT")  # 8 bases

        _result_read, _info, n_bp = pipeline._pre_process_single(read)
        assert len(_result_read.sequence) == 3  # 8 - 5 = 3 bases remain
        assert n_bp == 8

    def test_modifier_returning_none_stops_chain_and_returns_original_n_bp(self):
        pipeline = self._make_pipeline(
            _TrimFrontModifier(1),
            _ReturnNoneModifier(),
            _TrimFrontModifier(2),  # must NOT be reached
        )
        read = _make_read("ACGTACGT")

        result_read, info, n_bp = pipeline._pre_process_single(read)

        assert result_read is None
        assert n_bp == 8  # must not raise TypeError
