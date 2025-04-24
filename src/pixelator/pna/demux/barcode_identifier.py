"""Pipeline step to identify the antibody for a given barcode.

Copyright © 2024 Pixelgen Technologies AB.
"""

from collections import Counter
from typing import Any, cast

from cutadapt.steps import HasFilterStatistics, SingleEndStep
from dnaio import SequenceRecord

from pixelator.pna.config import PNAAntibodyPanel, PNAAssay, get_position_in_parent
from pixelator.pna.demux.correction import BKTree, build_bktree, build_exact_dict_lookup
from pixelator.pna.read_processing.statistics import HasCustomStatistics


class BarcodeIdentifierStatistics:
    """Helper class to collect statistics for the barcode identifier step.

    Each individual worker in a PipelineExecutor will have its own
    independent instance of this class. At the end of the execution,
    the statistics will be merged into a single instance and passed
    to the final statistics object.

    :ivar passed: the number of reads that passed the barcode identifier
    :ivar failed: the number of reads that failed the barcode identifier
    :ivar missing_pid1: the number of reads with a missing PID1
    :ivar missing_pid2: the number of reads with a missing PID2
    :ivar pid1_matches_distance_distribution:
        A counter object capturing the distribution of distances for PID1 matches.
    :ivar pid2_matches_distance_distribution:
        A counter object capturing the distribution of distances for PID2 matches.

    :ivar pid_pair_counter: A counter object capturing the number of reads per PID pair.
    """

    def __init__(self):
        """Initialize the BarcodeIdentifierStatistics object."""
        # Passsed counters
        self.exact = 0
        self.corrected = 0

        # Failed counters
        self.missing_pid1 = 0
        self.missing_pid2 = 0
        self.missing_pid1_pid2 = 0

        self.pid1_matches_distance_distribution = Counter({})
        self.pid2_matches_distance_distribution = Counter({})

        self.pid_pair_counter = Counter({})

    @property
    def passed(self) -> int:
        """Return the number of reads that passed the barcode identification."""
        return self.exact + self.corrected

    @property
    def failed(self) -> int:
        """Return the total number of reads that failed the barcode identification step."""
        return self.missing_pid1_pid2 + self.missing_pid1 + self.missing_pid2

    def __iadd__(self, other):
        """Merge statistics from another object into this one."""
        if isinstance(other, BarcodeIdentifierStatistics):
            self.exact += other.exact
            self.corrected += other.corrected
            self.missing_pid1 += other.missing_pid1
            self.missing_pid2 += other.missing_pid2
            self.pid1_matches_distance_distribution += (
                other.pid1_matches_distance_distribution
            )
            self.pid2_matches_distance_distribution += (
                other.pid2_matches_distance_distribution
            )
            self.pid_pair_counter += other.pid_pair_counter
            return self

        return NotImplemented

    def get_pid_group_counter(self) -> dict[tuple[str, str], int]:
        """Return the number of reads per PID pair."""
        return dict(self.pid_pair_counter)

    @property
    def input(self) -> int:
        """Return the total number of reads processed."""
        return self.passed + self.failed

    def collect(self) -> dict[str, Any]:
        """Return a dictionary with statistics."""
        pid_pairs = list(
            (marker1, marker2, count)
            for (marker1, marker2), count in self.pid_pair_counter.items()
        )

        return {
            "input_reads": self.input,
            "output_reads": self.passed,
            "failed_reads": self.failed,
            "output_corrected_reads": self.corrected,
            "output_exact_reads": self.exact,
            "invalid_pid1_reads": self.missing_pid1,
            "invalid_pid2_reads": self.missing_pid2,
            "invalid_pid1_pid2_reads": self.missing_pid1_pid2,
            "pid1_matches_distance_distribution": dict(
                self.pid1_matches_distance_distribution
            ),
            "pid2_matches_distance_distribution": dict(
                self.pid2_matches_distance_distribution
            ),
            "pid_group_sizes": pid_pairs,
        }

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"<BarcodeIdentifierStatistics [input={self.input} passed={self.passed} failed={self.failed}]>"


class BarcodeIdentifier(SingleEndStep, HasFilterStatistics, HasCustomStatistics):
    """A pipeline filter that finds the nearest antibody for a given barcode."""

    def __init__(
        self, assay: PNAAssay, panel: PNAAntibodyPanel, mismatches: int = 1, writer=None
    ):
        """Initialize the BarcodeIdentifier object.

        :param assay: the assay design
        :param panel: the antibody panel
        :param mismatches: the maximum number of mismatches allowed when aligning the LBS sequences
        :param writer: a writer to save failed reads to
        """
        self.assay = assay
        self.panel = panel
        self.mismatches = mismatches
        self._writer = writer

        self._tree_seq_1: BKTree = build_bktree(panel, sequence_key="sequence_1")
        self._exact_lookup_seq_1 = build_exact_dict_lookup(
            panel, sequence_key="sequence_1"
        )

        self._tree_seq_2: BKTree = build_bktree(panel, sequence_key="sequence_2")
        self._exact_lookup_seq_2 = build_exact_dict_lookup(
            panel, sequence_key="sequence_2"
        )

        self._pid1_slice = slice(*get_position_in_parent(assay, "pid-1"))
        self._pid2_slice = slice(*get_position_in_parent(assay, "pid-2"))

        self._stats = BarcodeIdentifierStatistics()

    def filtered(self):
        """Return the number of reads that passed the barcode identifier."""
        return self._stats.passed

    def descriptive_identifier(self):
        """Return a short name describing the step for the "filtered" section of the Statistics."""
        return "demux"

    def __call__(self, read: SequenceRecord, info=None) -> SequenceRecord | None:
        """Find the nearest antibody for a given barcode.

        :param read: the read to process
        :return: the read with the antibody information added
        """
        # Extract the barcode sequence
        _stats = self._stats

        pid1 = read.sequence[self._pid1_slice].encode("ascii")
        pid2 = read.sequence[self._pid2_slice].encode("ascii")

        # Check if we have an exact match first
        id1 = self._exact_lookup_seq_1.get(pid1)
        id2 = self._exact_lookup_seq_2.get(pid2)

        # Fast path - two exact matches
        if id1 and id2:
            _stats.pid1_matches_distance_distribution[0] += 1
            _stats.pid2_matches_distance_distribution[0] += 1
            _stats.exact += 1
            t = (id1, id2)
            _stats.pid_pair_counter[t] += 1
            read.name += f" {id1}:{id2}"
            return read

        # Error correcting path
        missing_pid1 = False
        missing_pid2 = False
        failed = False
        if not id1:
            nearest_matches1 = self._tree_seq_1.find(pid1, self.mismatches)
            if len(nearest_matches1) == 0:
                missing_pid1 = True
                failed = True
            else:
                pid1_score, pid1_node = nearest_matches1[0]
                _stats.pid1_matches_distance_distribution[pid1_score] += 1
                id1 = pid1_node.id

        if not id2:
            nearest_matches2 = self._tree_seq_2.find(pid2, self.mismatches)
            if len(nearest_matches2) == 0:
                missing_pid2 = True
                failed = True
            else:
                pid2_score, pid2_node = nearest_matches2[0]
                _stats.pid1_matches_distance_distribution[pid2_score] += 1
                id2 = pid2_node.id

        if failed:
            if missing_pid1 and missing_pid2:
                _stats.missing_pid1_pid2 += 1
            elif missing_pid1:
                _stats.missing_pid1 += 1
            elif missing_pid2:
                _stats.missing_pid2 += 1

            read.name += f" {id1}:{id2}"
            if self._writer:
                self._writer.write(read)
            return None

        # Mypy cannot infer that id1 and id2 are not None here
        t = cast(tuple[str, str], (id1, id2))
        _stats.pid_pair_counter[t] += 1
        _stats.corrected += 1

        read.name += f" {id1}:{id2}"
        return read

    def get_statistics_name(self):
        """Return a short name describing the statistics object."""
        return "demux"

    def get_statistics(self) -> BarcodeIdentifierStatistics:
        """Return the statistics object."""
        return self._stats
