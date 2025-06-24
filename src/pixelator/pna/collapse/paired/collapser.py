"""Collapse molecules from a demultiplexed PNA dataset.

Copyright © 2024 Pixelgen Technologies AB.
"""

import dataclasses
import logging
import math
import multiprocessing
import time
import typing
from contextlib import contextmanager
from functools import cached_property
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Generator

import joblib
import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet

from pixelator.common.report.models import SummaryStatistics
from pixelator.pna.collapse.adjacency import (
    build_network_cluster,
    build_network_directional,
)
from pixelator.pna.collapse.paired.statistics import (
    CollapseStatistics,
    MarkerLinkGroupStats,
)
from pixelator.pna.collapse.utilities import (
    _find_connected_components_cluster,
    _find_connected_components_directional,
    build_binary_index,
)
from pixelator.pna.config import PNAAntibodyPanel, PNAAssay
from pixelator.pna.demux.barcode_demuxer import PNAEmbedding
from pixelator.pna.utils import pack_2bits
from pixelator.pna.utils.two_bit_encoding import pack_4bits

logger = logging.getLogger("collapse")


def _collect_label_array_indices(
    labels: npt.NDArray[np.int32], n_components: int
) -> npt.NDArray[np.object_]:
    """Collect an array with a label for each item into an array of indices for each label.

    Args:
        labels: The labels for each item.
        n_components: Number of labels in the label array.

    Returns:
        An array of indices for each label.

    """
    # Single pass over the labels to collect the indices of each connected component
    groups = np.ndarray(shape=(n_components,), dtype=object)  # type: ignore

    for idx, value in enumerate(labels):
        g = groups[value]
        if g is None:
            groups[value] = [idx]
        else:
            g.append(idx)

    return groups


def _split_chunks(
    n_components: int, chunk_size: int
) -> Generator[tuple[int, int], None, None]:
    """Split a range [0, n_components) into chunks of size chunk_size.

    This is a generator function that yields the start and stop indices of each chunk.
    The last chunk may be smaller than chunk_size.

    Args:
        n_components: The size of the input range.
        chunk_size: The size of each chunk.

    Yields:
        A tuple of (start, stop) indices for each chunk.

    """
    pos = 0
    complete_chunks = n_components // chunk_size

    for i in range(complete_chunks):
        yield (pos, pos + chunk_size)
        pos += chunk_size

    yield (pos, n_components)


@dataclasses.dataclass(slots=True, frozen=True)
class _DistributedResults:
    start: int
    stop: int

    umi1: npt.ArrayLike
    umi2: npt.ArrayLike
    uei: npt.ArrayLike

    read_count: npt.ArrayLike
    corrected_reads: int


MoleculeCollapserAlgorithm = typing.Literal["directional", "cluster"]


class MoleculeCollapser:
    """Collapse molecules from a demultiplexed PNA dataset.

    This class is responsible for collapsing molecules from a demultiplexed PNA dataset.
    Collapsing is parallelized for each marker pair so parallelizing collapsing over multiple
    independent groups is not needed.
    """

    _collapsed_schema = pa.schema(
        [
            pa.field("umi1", pa.uint64()),
            pa.field("umi2", pa.uint64()),
            pa.field("uei", pa.uint64()),
            pa.field("read_count", pa.int64()),
        ]
    )

    _output_schema = pa.schema(
        [
            pa.field("marker_1", pa.string()),
            pa.field("marker_2", pa.string()),
            pa.field("umi1", pa.uint64()),
            pa.field("umi2", pa.uint64()),
            pa.field("read_count", pa.int64()),
            pa.field("uei_count", pa.int64()),
        ]
    )

    def __init__(
        self,
        assay: PNAAssay,
        panel: PNAAntibodyPanel,
        output: Path,
        max_mismatches: int | float = 0.1,
        algorithm: MoleculeCollapserAlgorithm = "directional",
        threads: int = -1,
        logger: logging.Logger | None = None,
        min_parallel_chunk_size: int = 2_000,
    ):
        """Initialize a MoleculeCollapser.

        Args:
            assay: The assay configuration.
            panel: The antibody panel configuration.
            output: The output path for the collapsed molecules.
            max_mismatches: The maximum number of mismatches allowed when collapsing molecules.
                Either an integer >= 1 or a float in the range [0, 1).
            algorithm: The algorithm to use for collapsing molecules. Either "directional" or "cluster".
            threads: The number of threads to use for parallel processing.
            logger: The logger to use for output. The default is a logger named "collapse".
            min_parallel_chunk_size: The minimum number of connected components to process in parallel.
                Components below this size will be processed serially.

        """
        self.assay = assay
        self.panel = panel
        self.algorithm = algorithm

        if max_mismatches >= 1:
            if isinstance(max_mismatches, float):
                raise ValueError(
                    "max_mismatches must be either an integer value > 1"
                    " or a float value between 0 and 1."
                )
            self.max_mismatches = max_mismatches
        else:
            molecule_len = (
                self.assay.get_region_by_id("umi-1").max_len
                + self.assay.get_region_by_id("umi-2").max_len
                + self.assay.get_region_by_id("uei").max_len
            )
            self.max_mismatches = math.ceil(molecule_len * max_mismatches)

        self.threads = threads
        self._memory_manager = SharedMemoryManager()
        self._parallel_worker = joblib.Parallel(n_jobs=threads, return_as="list")
        self._logger = logger or logging.getLogger("collapse")

        self._writer = pyarrow.parquet.ParquetWriter(
            output, schema=self._output_schema, compression="zstd", compression_level=6
        )
        self._marker_group_state = {
            "db_shm": None,
            "read_counts_shm": None,
        }

        self._bitvector = PNAEmbedding(self.assay)
        self._stats = CollapseStatistics()

        self.min_parallel_chunk_size = min_parallel_chunk_size

    def __enter__(self):
        """Enter the context manager for the MoleculeCollapser.

        This wraps the context managers for the shared memory manager, parallel worker pool,
        and the parquet writer.
        """
        self._memory_manager = self._memory_manager.__enter__()
        self._parallel_worker = self._parallel_worker.__enter__()
        self._writer = self._writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager for the MoleculeCollapser.

        This wraps the context managers for the shared memory manager, parallel worker pool,
        and the parquet writer.
        """
        self._memory_manager.__exit__(exc_type, exc_val, exc_tb)
        self._parallel_worker.__exit__(exc_type, exc_val, exc_tb)
        self._writer.__exit__(exc_type, exc_val, exc_tb)

    @cached_property
    def _effective_threads(self):
        if self.threads <= -1:
            return multiprocessing.cpu_count() + 1 - abs(self.threads)

        return self.threads

    @property
    def _current_db_memory(self) -> SharedMemory | None:
        return self._marker_group_state["db_shm"]

    @_current_db_memory.setter
    def _current_db_memory(self, buffer):
        self._marker_group_state["db_shm"] = buffer

    @property
    def _current_read_counts_memory(self) -> SharedMemory | None:
        return self._marker_group_state["read_counts_shm"]

    @_current_read_counts_memory.setter
    def _current_read_counts_memory(self, buffer):
        self._marker_group_state["read_counts_shm"] = buffer

    @contextmanager
    def _init_binary_vectors(self, data):
        """Return a contextmanager for the shared memory containing embeddings.

        Args:
            data: The data to initialize the binary vectors with.

        Yields:
            The binary vectors.

        """
        vector_length = 32
        shm_buffer = self._memory_manager.SharedMemory(size=(len(data) * vector_length))
        self._current_db_memory = shm_buffer

        db = np.ndarray(
            (len(data), vector_length), dtype=np.uint8, buffer=shm_buffer.buf
        )
        db[:, :] = 0

        for idx, vector in enumerate(data["molecule"]):
            db[idx, :] = np.frombuffer(vector, dtype=np.uint8, count=vector_length)

        yield db

        self._current_db_memory.unlink()
        self._current_db_memory = None

    @contextmanager
    def _init_read_counts(self, data):
        """Return a contextmanager for the shared memory containing read counts.

        Args:
            data: The data to initialize the read counts with.

        Yields:
            The read counts.

        """
        n_molecules = len(data)
        shm_buffer = self._memory_manager.SharedMemory(
            size=(n_molecules * np.dtype(np.uint64).itemsize)
        )
        read_counts = np.ndarray(n_molecules, dtype=np.uint64, buffer=shm_buffer.buf)
        read_counts[:] = data["read_count"].to_numpy()
        self._current_read_counts_memory = shm_buffer

        yield read_counts

        self._current_read_counts_memory.unlink()
        self._current_read_counts_memory = None

    @cached_property
    def max_hamming_mismatches(self) -> int:
        """Return the maximum hamming distance for the number of allowed mismatches."""
        return self.max_mismatches * 2

    @staticmethod
    def _get_db_from_shared_memory(db_shm: SharedMemory, count: int):
        """Return a numpy array backed by a shared memory buffer for the embeddings.

        Args:
            db_shm: the shared memory buffer.
            count: the number of vectors in the buffer

        Returns:
            A numpy array containing the embeddings.

        """
        vector_len = 32
        db = np.frombuffer(db_shm.buf, dtype=np.uint8, count=count * vector_len)
        db.shape = (count, vector_len)
        return db

    @staticmethod
    def _get_counts_from_shared_memory(counts_shm: SharedMemory, count: int):
        """Return a numpy array backed by a shared memory buffer for the read counts.

        Args:
            counts_shm: The shared memory buffer.
            count: The number of read counts in the buffer.

        Returns:
            The numpy array.

        """
        counts = np.frombuffer(counts_shm.buf, dtype=np.int64, count=count)
        return counts

    def _record_group_serial_fn(
        self,
        output_idx,
        component_indices,
        db_shm,
        read_counts_shm,
        db_size,
        outputs,
        n_components,
    ):
        """Record a group of connected components serially.

        Args:
            output_idx: The output index.
            component_indices: The indices of the components.
            db_shm: The shared memory buffer for the database.
            read_counts_shm: The shared memory buffer for the read counts.
            db_size: The size of the database.
            outputs: The outputs.
            n_components: The number of components.

        Returns:
            The number of corrected reads.

        """
        db = self._get_db_from_shared_memory(db_shm, db_size)
        read_count = self._get_counts_from_shared_memory(read_counts_shm, db_size)

        # Aggregate the read_count for this connected component
        comp_read_counts = read_count[component_indices]
        total_read_count = np.sum(comp_read_counts)

        # Select the molecule with the most support as "representative" of the component
        representative_idx = np.argmax(comp_read_counts)
        comp_idx = component_indices[representative_idx]
        umi1, umi2, uei = self._bitvector.decode(db[comp_idx], skip_uei=False)

        umi1 = pack_2bits(umi1)
        umi2 = pack_2bits(umi2)
        uei = pack_4bits(uei)
        read_count = total_read_count
        corrected_reads = int(total_read_count - comp_read_counts[representative_idx])

        outputs[0][output_idx] = umi1
        outputs[1][output_idx] = umi2
        outputs[2][output_idx] = uei
        outputs[3][output_idx] = read_count

        return int(corrected_reads)

    @staticmethod
    def _record_group_worker_fn(
        subrange: tuple[int, int],
        component_indices,
        db_shm,
        read_counts_shm,
        db_size,
        embedding: PNAEmbedding,
    ):
        """Process a batch of connected components.

        The database and read counts are loaded from shared memory to reduce
        python multiprocessing IPC overhead.

        Args:
            subrange: The range of connected components to process.
                A tuple with the start and stop indices.
            component_indices: A list of lists containing the indices
                in the database and read counts vector for each connected component.
            db_shm: The shared memory buffer containing the binary vectors.
            read_counts_shm: The shared memory buffer containing the read counts.
            db_size: The size of the binary vectors memory buffer in bytes.
            embedding: The PNAEmbedding instance for encoding/decoding vectors.

        Returns:
            The distributed results.

        """
        cls = MoleculeCollapser

        db = cls._get_db_from_shared_memory(db_shm, db_size)
        read_count = cls._get_counts_from_shared_memory(read_counts_shm, db_size)
        corrected_reads = 0

        subrange_len = subrange[1] - subrange[0]
        umi1_out = np.zeros((subrange_len,), dtype=np.uint64)
        umi2_out = np.zeros((subrange_len,), dtype=np.uint64)
        uei_out = np.zeros((subrange_len,), dtype=np.uint64)
        reads_out = np.zeros((subrange_len,), dtype=np.int64)

        for idx in range(subrange_len):
            this_cmp_indices = component_indices[idx]

            # Aggregate the read_count for this connected component
            molecule_read_counts = read_count[this_cmp_indices]
            component_reads = np.sum(molecule_read_counts)

            # Select the molecule with the most support as "representative" of the component
            # TODO: Voting on a representative molecule per nucleotide?
            representative_idx = np.argmax(molecule_read_counts)
            comp_idx = this_cmp_indices[representative_idx]
            umi1_bytes, umi2_bytes, uei_bytes = embedding.decode(
                db[comp_idx], skip_uei=False
            )

            umi1 = pack_2bits(umi1_bytes)
            umi2 = pack_2bits(umi2_bytes)
            uei = pack_4bits(uei_bytes)
            corrected_reads += (
                component_reads - molecule_read_counts[representative_idx]
            )

            umi1_out[idx] = umi1
            umi2_out[idx] = umi2
            uei_out[idx] = uei
            reads_out[idx] = component_reads

        res = _DistributedResults(
            start=subrange[0],
            stop=subrange[1],
            umi1=umi1_out,
            umi2=umi2_out,
            uei=uei_out,
            read_count=reads_out,
            corrected_reads=int(corrected_reads),
        )
        return res

    def _process_molecule_graph(
        self, csgraph, local_stats
    ) -> tuple[pa.Table, MarkerLinkGroupStats]:
        """Determine connected components and collapse the molecules in each component.

        Args:
            csgraph: The sparse adjacency matrix of the connected components.
            local_stats: The statistics object for this marker pair.

        Returns:
            A tuple with a pyarrow Table with the collapsed molecules
            and the updated statistics object.

        """
        _logger = self._logger

        n_components: int
        labels: npt.NDArray[np.int32]

        if self.algorithm == "cluster":
            n_components, labels = _find_connected_components_cluster(csgraph)
        elif self.algorithm == "directional":
            n_components, labels = _find_connected_components_directional(csgraph)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        n_molecules = csgraph.shape[0]

        # Single pass over the labels to collect the indices of each connected component
        groups = _collect_label_array_indices(labels, n_components)
        local_stats.cluster_size_distribution = np.bincount(
            np.vectorize(len)(groups)
        ).tolist()

        db_mem = self._current_db_memory
        rc_mem = self._current_read_counts_memory
        bitvector = self._bitvector

        def work_fn_parallel(subrange):
            start, stop = subrange
            res = MoleculeCollapser._record_group_worker_fn(
                subrange, groups[start:stop], db_mem, rc_mem, n_molecules, bitvector
            )
            return res

        # Allocate output vectors in shared memory or as normal numpy arrays
        if (
            self._effective_threads > 1
            and n_components >= 2 * self.min_parallel_chunk_size
        ):
            chunk_size = max(
                self.min_parallel_chunk_size, n_components // self._effective_threads
            )
            _logger.info(
                "Detecting unique molecules from %s connected components (parallel: [chunk_size=%s])",
                n_components,
                chunk_size,
            )
            subranges = list(_split_chunks(n_components, chunk_size=chunk_size))

            job_results: list[_DistributedResults] = self._parallel_worker(
                joblib.delayed(work_fn_parallel)(r) for r in subranges
            )
        else:
            _logger.info(
                "Determining collapsed read for %s connected components (serial)",
                n_components,
            )
            job_results = [work_fn_parallel((0, n_components))]

        umi1_arrays = []
        umi2_arrays = []
        uei_arrays = []
        read_count_arrays = []

        for r in job_results:
            umi1_arrays.append(r.umi1)
            umi2_arrays.append(r.umi2)
            uei_arrays.append(r.uei)
            read_count_arrays.append(r.read_count)

            local_stats.corrected_reads_count += r.corrected_reads

        table = pyarrow.Table.from_arrays(
            arrays=[
                pa.chunked_array(umi1_arrays),
                pa.chunked_array(umi2_arrays),
                pa.chunked_array(uei_arrays),
                pa.chunked_array(read_count_arrays),
            ],
            schema=self._collapsed_schema,
        )

        local_stats.collapsed_molecules_count = len(table)
        read_count_collapsed = table.column("read_count")
        local_stats.read_count_per_collapsed_molecule_stats = (
            SummaryStatistics.from_series(read_count_collapsed)
        )

        grouped = table.group_by(["umi1", "umi2"]).aggregate(
            [("read_count", "sum"), ("uei", "count_distinct")]
        )

        grouped = grouped.rename_columns(["umi1", "umi2", "read_count", "uei_count"])

        local_stats.unique_marker_links_count = len(grouped)

        read_count_grouped = grouped.column("read_count")
        local_stats.read_count_per_unique_marker_link_stats = (
            SummaryStatistics.from_series(read_count_grouped)
        )
        local_stats.uei_count_per_unique_marker_link_stats = (
            SummaryStatistics.from_series(grouped["uei_count"])
        )
        return grouped, local_stats

    def _process_marker_group(self, idx, num_groups, markers, data):
        """Process a group of markers.

        Args:
            idx: The index of the group.
            num_groups: The total number of groups.
            markers: The markers in the group.
            data: The data for the group.

        """
        starttime = time.time()
        _logger = self._logger

        marker1_name = self.panel.markers[markers[0]]
        marker2_name = self.panel.markers[markers[1]]

        input_molecules_count = len(data)
        input_reads_count = data["read_count"].sum()

        local_stats = MarkerLinkGroupStats(
            marker_1=marker1_name,
            marker_2=marker2_name,
            input_molecules_count=input_molecules_count,
            input_reads_count=input_reads_count,
        )

        _logger.info(
            "Processing %s molecules from markers [%s, %s] (%s/%s)",
            input_molecules_count,
            marker1_name,
            marker2_name,
            idx + 1,
            num_groups,
        )

        # Allocate shared memory for the binary vectors that make up the search space
        # and the read counts for each molecule. These are loaded with a context manager
        # to ensure proper cleanup
        with (
            self._init_binary_vectors(data) as db,
            self._init_read_counts(data) as read_counts,
        ):
            _logger.info(
                "Building binary index for [%s, %s] from %s molecules",
                marker1_name,
                marker2_name,
                len(db),
            )
            index = build_binary_index(db)

            _logger.info(
                "Querying binary index for [%s, %s] for similar molecules",
                marker1_name,
                marker2_name,
            )
            distances, indices = index.search(db, 48)

            _logger.info(
                "Building sparse adjacency matrix for [%s, %s]",
                marker1_name,
                marker2_name,
            )

            if self.algorithm == "cluster":
                adjacency = build_network_cluster(
                    distances, indices, read_counts, self.max_hamming_mismatches
                )
            elif self.algorithm == "directional":
                adjacency = build_network_directional(
                    distances, indices, read_counts, self.max_hamming_mismatches
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            _logger.info(
                f"Collapsing clustered reads ({self.algorithm}) for [%s, %s]",
                marker1_name,
                marker2_name,
            )

            records, cluster_stats = self._process_molecule_graph(
                adjacency, local_stats
            )

            marker1_array = np.zeros(shape=(len(records)), dtype="<U16")
            marker2_array = np.zeros(shape=(len(records)), dtype="<U16")
            marker1_array[:] = marker1_name
            marker2_array[:] = marker2_name

            records = records.add_column(
                0, self._output_schema.field("marker_1"), pa.array(marker1_array)
            )
            records = records.add_column(
                1, self._output_schema.field("marker_2"), pa.array(marker2_array)
            )

            _logger.info(
                "Collapsed %s molecules into %s unique antibody links, correcting %s reads of %s total reads",
                input_molecules_count,
                len(records),
                cluster_stats.corrected_reads_count,
                cluster_stats.input_reads_count,
            )

            _logger.info("Streaming %s records to parquet", len(records))
            self._writer.write_table(records)

            _logger.info(
                "Completed processing %s molecules from markers [%s, %s] (%s/%s)",
                input_molecules_count,
                marker1_name,
                marker2_name,
                idx + 1,
                num_groups,
            )

        elapsed_time = time.time() - starttime
        self._stats.add_marker_stats(
            marker1_name,
            marker2_name,
            cluster_stats=cluster_stats,
            elapsed_time=elapsed_time,
        )

        return

    def process_file(self, path: Path) -> None:
        """Collapse molecules from a single parquet file containing molecular data.

        Args:
            path: The path to the parquet file.

        """
        _logger = self._logger

        _logger.info("Loading data from parquet: %s", str(path))
        df = pl.read_parquet(path)

        # Handle legacy data
        if "marker1" in df.columns or "marker2" in df.columns:
            df = df.rename({"marker1": "marker_1", "marker2": "marker_2"}, strict=False)

        self._stats.add_input_file(path, df.shape[0])

        _logger.info("Partitioning dataset by marker pairs")
        # Partition the data by the marker1 and marker2 columns and store each partition as a separate DataFrame
        partitions = df.partition_by(
            ["marker_1", "marker_2"], as_dict=True, include_key=False
        )

        num_partitions = len(partitions)
        _logger.info("Dataset partitioned into %s subsets", num_partitions)

        for idx, (markers, parts) in enumerate(partitions.items()):
            self._process_marker_group(idx, num_partitions, markers, parts)

    def statistics(self) -> CollapseStatistics:
        """Return the statistics collected during the collapse process.

        Returns:
            The collapse statistics.

        """
        return self._stats
