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
from pathlib import Path, PurePath
from typing import Literal

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pyarrow
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet
import pydantic

from pixelator.pna.collapse.adjacency import (
    build_network_cluster,
    build_network_directional,
)
from pixelator.pna.collapse.independent.shared_memory_registry import (
    ReadOnlySharedMemoryRegistry,
    SharedMemoryRegistry,
)
from pixelator.pna.collapse.utilities import (
    FAISSBackend,
    MoleculeCollapserAlgorithm,
    _collect_label_array_indices,
    _find_connected_components_cluster,
    _find_connected_components_directional,
    _split_chunks,
)
from pixelator.pna.config import PNAAntibodyPanel, PNAAssay
from pixelator.pna.demux.barcode_demuxer import PNAEmbedding
from pixelator.pna.report.models import SampleReport

logger = logging.getLogger("collapse")

CollapsibleRegion = Literal["umi-1", "umi-2"]


class MarkerCorrectionStats(pydantic.BaseModel):
    """Collect statistics of groups of molecules that will be collapsed.

    Attributes:
        marker: The name of the marker for which we are collapsing UMIs.
        region_id: The region id of the UMI to collapse. Either "umi-1" or "umi-2".
        input_reads: The total number of input reads to the collapse process.
        input_molecules: The total number of unique input reads to the collapse process.
        input_unique_umis: The total number of unique UMIs of `region_id` among the input reads.
        corrected_reads: The total number of reads that were modified by the error correction process.
        corrected_molecules: The total number of unique reads that were modified by the error correction process.
        corrected_unique_umis: The total number of unique UMIs of type `region_id` that were modified by the error correction process.
        output_unique_umis: The total number of unique UMIs of type `region_id` after correction.

    """

    marker: str
    region_id: CollapsibleRegion

    input_reads: int = 0
    input_molecules: int = 0
    input_unique_umis: int = 0

    corrected_reads: int = 0
    corrected_unique_umis: int = 0

    output_unique_umis: int = 0

    @pydantic.computed_field(return_type=int)  # type: ignore
    @property
    def output_reads(self) -> int:
        """The total number of reads after correction.

        This is always the same as `input_reads_count` but is included for consistency.
        """
        return self.input_reads

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def corrected_reads_fraction(self) -> float:
        """The fraction of input reads that were corrected."""
        return self.corrected_reads / self.input_reads if self.input_reads else 0.0

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def corrected_unique_umis_fraction(self) -> float:
        """The fraction of unique UMIs of `region_type` that were corrected."""
        return (
            self.corrected_unique_umis / self.input_unique_umis
            if self.input_unique_umis
            else 0.0
        )


@dataclasses.dataclass(slots=True)
class CollapseInputFile:
    """Keep track of the input file to collapse.

    :param path: Path to the input file.
    :param file_size: The total size of the input file.
    :param molecule_count: The number of rows in the input dataframe.
        i.e. The number of molecules (unique reads).
    """

    path: str
    file_size: int
    molecule_count: int


class SingleUMICollapseSampleReport(SampleReport):
    """Model for report data returned by the collapse stage."""

    report_type: str = "collapse-umi"

    region_id: str = pydantic.Field(
        ..., description="The UMI region that was collapsed."
    )

    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )

    input_molecules: int = pydantic.Field(
        ...,
        description="The number of molecules (unique reads) processed.",
    )

    input_unique_umis: int = pydantic.Field(
        ...,
        description="The number of unique input UMIs processed.",
    )

    output_unique_umis: int = pydantic.Field(
        ...,
        description="The number of unique UMIs after correction.",
    )

    corrected_unique_umis: int = pydantic.Field(
        ...,
        description="The number of UMIs with errors that were corrected.",
    )

    corrected_reads: int = pydantic.Field(
        ...,
        description="The sum of the reads over all UMIs that were corrected.",
    )

    processed_files: list[CollapseInputFile] = pydantic.Field(
        ..., description="The files processed during the collapse step."
    )

    markers: list[MarkerCorrectionStats] = pydantic.Field(
        ..., description="Correction statistics per UMI region for each marker group."
    )


T = typing.TypeVar("T")


class IndependentCollapseStatisticsCollector:
    """Collect statistics about the collapse process."""

    class _SummaryStatsDict(typing.TypedDict):
        input_reads: int
        input_molecules: int
        input_unique_umis: int
        corrected_reads: int
        corrected_unique_umis: int
        output_unique_umis: int

    def __init__(self, region_id: CollapsibleRegion) -> None:
        """Initialize the statistics collector."""
        self._processed_files: list[CollapseInputFile] = []
        self._marker_umi_data: dict[tuple[str, str], MarkerCorrectionStats] = {}
        self.region_id = region_id

    @property
    def markers(self) -> list[MarkerCorrectionStats]:
        """Return the per marker stats for the umi-1 region."""
        return [v for v in self._marker_umi_data.values()]

    @typing.overload
    def add_input_file(
        self, input_file: PurePath, molecule_count: int, file_size: int
    ) -> None: ...

    @typing.overload
    def add_input_file(
        self, input_file: Path, molecule_count: int, file_size: int | None = None
    ) -> None: ...

    def add_input_file(
        self,
        input_file: Path | PurePath,
        molecule_count: int,
        file_size: int | None = None,
    ) -> None:
        """Collect file statistics for an input file to the MoleculeCollapser.

        :param input_file: The input file to collapse.
        :param molecule_count: The number of molecules in the input file
        :raise TypeError: If file_size is not provided when input_file is a PurePath.
        """
        if file_size is None and isinstance(input_file, Path):
            file_size = input_file.stat(follow_symlinks=True).st_size
        elif file_size is None:
            raise TypeError("file_size must be provided if input_file is a PurePath")

        self._processed_files.append(
            CollapseInputFile(
                path=str(input_file),
                file_size=file_size,
                molecule_count=molecule_count,
            )
        )

    def add_marker_stats(
        self,
        stats: MarkerCorrectionStats,
    ) -> None:
        """Add statistics for a marker pair.

        Args:
            stats: The statistics for the correction of UMIs for reads with this marker.

        Raises:
            KeyError: If data for the marker pair already exists.

        """
        key = (stats.region_id, stats.marker)

        if stats.region_id != self.region_id:
            raise ValueError(
                f"Region id mismatch: {stats.region_id} != {self.region_id}"
            )

        if key in self._marker_umi_data:
            raise KeyError(
                f"Data for {stats.region_id} of {stats.marker} already exists"
            )

        self._marker_umi_data[key] = stats

    def get_combined_stats(self) -> _SummaryStatsDict:
        """Compute combined summary statistics from all the marker pairs."""
        df = pd.DataFrame([v.model_dump() for v in self._marker_umi_data.values()])

        aggregate_cols = [
            "input_reads",
            "input_molecules",
            "input_unique_umis",
            "corrected_reads",
            "corrected_unique_umis",
            "output_unique_umis",
        ]

        sums = df[aggregate_cols].sum()
        stats = sums.to_dict()
        return stats

    def to_sample_report(self, sample_id) -> SingleUMICollapseSampleReport:
        """Return the statistics as a SampleReport.

        Returns:
            A `SingleUMICollapseSampleReport` containing the statistics.

        """
        stats = self.get_combined_stats()
        return SingleUMICollapseSampleReport(
            sample_id=sample_id,
            product_id="single-cell-pna",
            region_id=self.region_id,
            **stats,
            processed_files=self._processed_files,
            markers=list(self._marker_umi_data.values()),
        )

    def to_dict(self) -> dict:
        """Return the statistics as a dictionary."""
        stats = self.get_combined_stats()
        return {
            "region_id": self.region_id,
            "processed_files": [dataclasses.asdict(f) for f in self._processed_files],
            "markers": [v.model_dump() for v in self._marker_umi_data.values()],
            **stats,
        }


@dataclasses.dataclass(slots=True, frozen=True)
class _DistributedResults:
    start: int
    stop: int

    #: A list of lists containing the molecule indices for each connected component
    unique_molecule_ids: list
    #: A list of the idx of the UMI that a cluster of UMIs need to be corrected to
    representative_indices: npt.ArrayLike
    corrected_reads_count: int

    collapsed_umi_count: int = 0
    output_umi_count: int = 0


class RegionCollapser:
    """Error correct UMI sequences based on similarity.

    Attributes:
        _umi1_data: A numpy array containing the unique UMI-1 sequences for the current processing batch.
            These are recoded to a 2-bit encoding and cast to a 64-bit integer.
        _umi2_data: A numpy array containing the unique UMI-2 sequences for the current processing batch.
            These are recoded to a 2-bit encoding and cast to a 64-bit integer.

        _db_to_molecule_idx: A numpy array containing for each of the input moleces the index of the unique UMI.
            This is used to link corrections of the unique umis back to all molecules that share the same UMI.
        _unique_umi_to_molecule_count: The number of molecules that map to each unique umi.
            This is used to map "unique umi" counts to the corresponding "molecule" counts.

    """

    _umi1_schema = pa.schema(
        fields=[
            pa.field("marker_1", pa.string()),
            pa.field("marker_2", pa.string()),
            pa.field("read_count", pa.uint16()),
            pa.field("original_umi1", pa.uint64()),
            pa.field("original_umi2", pa.uint64()),
            pa.field("uei", pa.uint64()),
            pa.field("corrected_umi1", pa.uint64()),
        ]
    )
    _umi2_schema = pa.schema(
        fields=[
            pa.field("marker_1", pa.string()),
            pa.field("marker_2", pa.string()),
            pa.field("read_count", pa.uint16()),
            pa.field("original_umi1", pa.uint64()),
            pa.field("original_umi2", pa.uint64()),
            pa.field("uei", pa.uint64()),
            pa.field("corrected_umi2", pa.uint64()),
        ]
    )

    def __init__(
        self,
        assay: PNAAssay,
        panel: PNAAntibodyPanel,
        region_id: CollapsibleRegion,
        max_mismatches: int | float = 0.1,
        algorithm: MoleculeCollapserAlgorithm = "directional",
        threads: int = -1,
        logger: logging.Logger | None = None,
        min_parallel_chunk_size: int = 500,
        similarity_backend: Literal["faiss"] = "faiss",
    ):
        """Initialize a RegionCollapser.

        Args:
            assay: The assay configuration.
            panel: The antibody panel configuration.
            region_id: The region id of the UMI to collapse. Either "umi-1" or "umi-2".
            max_mismatches: The maximum number of mismatches allowed when collapsing molecules.
                Either an integer >= 1 or a float in the range [0, 1).
            algorithm: The algorithm to use for collapsing molecules. Either "cluster" or "directional".
            threads: The number of threads to use for parallel processing.
            logger: The logger to use for output. The default is a logger named "collapse".
            min_parallel_chunk_size: The minimum number of com to process in parallel.
            similarity_backend: The backend to use for similarity search. Currently only "faiss".

        """
        self.assay = assay
        self.panel = panel
        self.algorithm = algorithm
        self.region_id = region_id

        if max_mismatches >= 1:
            if isinstance(max_mismatches, float):
                raise ValueError(
                    "max_mismatches must be either an integer value > 1"
                    " or a float value between 0 and 1."
                )
            self.max_mismatches = max_mismatches
        else:
            region_len = self.assay.get_region_by_id(self.region_id).max_len
            self.max_mismatches = math.ceil(region_len * max_mismatches)

        self.threads = threads
        self._memory = SharedMemoryRegistry()
        self._parallel_worker = joblib.Parallel(n_jobs=threads, return_as="list")
        self._logger = logger or logging.getLogger("collapse")
        self._simsearch = FAISSBackend(threads=threads)

        match self.region_id:
            case "umi-1":
                self._output_schema = self._umi1_schema
            case "umi-2":
                self._output_schema = self._umi2_schema
            case _:
                raise ValueError(f"Unknown region_id: {self.region_id}")

        self._embedding = PNAEmbedding(self.assay)
        self._stats = IndependentCollapseStatisticsCollector(region_id=self.region_id)
        self._marker_idx_to_name = np.vectorize(self.panel.markers.__getitem__)
        self.min_parallel_chunk_size = min_parallel_chunk_size

        # TODO vectorize these in the Embedding class itself

        self._umi1_region_extractor = np.vectorize(
            self._embedding.get_umi1_bytes, cache=True
        )
        self._umi2_region_extractor = np.vectorize(
            self._embedding.get_umi2_bytes, cache=True
        )
        self._uei_region_extractor = np.vectorize(
            self._embedding.get_uei_bytes, cache=True
        )
        self._compress_umi_embedding = np.vectorize(
            self._embedding.compress_umi_embedding, cache=True
        )
        self._compress_uei_embedding = np.vectorize(
            self._embedding.compress_uei_embedding, cache=True
        )

    def __enter__(self):
        """Enter the context manager for the RegionCollapser.

        This wraps the context managers for the shared memory manager, parallel worker pool,
        and the parquet writer.
        """
        self._memory = self._memory.__enter__()
        self._parallel_worker = self._parallel_worker.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager for the RegionCollapser.

        This wraps the context managers for the shared memory manager, parallel worker pool,
        and the parquet writer.
        """
        self._memory.__exit__(exc_type, exc_val, exc_tb)
        self._parallel_worker.__exit__(exc_type, exc_val, exc_tb)

    def statistics(self) -> IndependentCollapseStatisticsCollector:
        """Return the statistics collected during the collapse process."""
        return self._stats

    @cached_property
    def _effective_threads(self):
        if self.threads <= -1:
            return multiprocessing.cpu_count() + 1 - abs(self.threads)

        return self.threads

    @property
    def _umi_data(self) -> npt.NDArray:
        """Return either umi1 or umi2 data depending on the region_id."""
        return self._region_id_choice(self._umi1_data, self._umi2_data)

    @property
    def _db_to_molecule_idx(self) -> npt.NDArray:
        """Return the unique to molecule mapping for the current region_id."""
        return self._region_id_choice(
            self._db_to_molecule_idx_umi1, self._db_to_molecule_idx_umi2
        )

    def _extract_unique_umis(self, molecules: pl.Series):
        """Extract unique UMI sequences from a series of binary embeddings.

        Depending on the region_id either the UMI1 or UMI2 region is extracted.

        Args:
            molecules (pl.Series): A series containing the molecule data.

        Returns:
            A numpy array containing the unique UMI sequences.

        """
        umi1_data = self._umi1_region_extractor(molecules)
        unique_umi1s, inverse_umi1, counts_umi1 = np.unique(
            umi1_data, return_inverse=True, return_counts=True
        )
        del umi1_data

        umi2_data = self._umi2_region_extractor(molecules)
        unique_umi2s, inverse_umi2, counts_umi2 = np.unique(
            umi2_data, return_inverse=True, return_counts=True
        )
        del umi2_data

        # Re-encode the UMI sequences to a 32-bit integer.
        # These numeric embeddings will be writen to the output parquet files.
        # The original UMI embeddings are used to search for near-identical sequences since those are equidistant in the hamming space.
        self._umi1_data = self._compress_umi_embedding(unique_umi1s)
        self._umi2_data = self._compress_umi_embedding(unique_umi2s)

        # The indices in the unique umi array (eg: self._umi_data or the search db embedding)
        # that reconstruct the input molecule array.
        self._db_to_molecule_idx_umi1 = inverse_umi1
        self._db_to_molecule_idx_umi2 = inverse_umi2

        # The number of molecules that map to each unique umi.
        self._unique_umi_to_molecule_count_umi1 = counts_umi1

        # The number of molecules that map to each unique umi.
        self._unique_umi_to_molecule_count_umi2 = counts_umi2

        return (unique_umi1s, unique_umi2s)

    def _region_id_choice(self, umi1_res: T, umi2_res: T) -> T:
        """Return the first or second argument based on the current region id (umi1|umi2)."""
        # Avoid repeating this kind of logic everywhere
        if self.region_id == "umi-1":
            return umi1_res
        elif self.region_id == "umi-2":
            return umi2_res
        else:
            raise ValueError(f"Invalid region_id: {self.region_id}")

    @contextmanager
    def _init_shared_memory(self, data):
        """Return a contextmanager for the shared memory containing embeddings and read counts.

        This will allocate 2 shared memory buffers in the registry:

        db: A 2D array of uint8 with shape (num_unique_umis, 12) containing the unique umis
            Each row is a 12-byte binary vector representing a unique umi.

        read_counts: A 1D array of uint64 with shape (num_unique_umis) containing the read counts
        """
        # Inputs are 3 bit encoded UMI sequences
        # For the current 28 bases UMIs this results in 28/8 * 3 = 11 bytes per UMI
        # The UMI regions in the molecule embedding created by demux is zero-padded to 12 bytes.
        vector_length = 12

        unique_umi1s, unique_umi2s = self._extract_unique_umis(data["molecule"])
        unique_umis = self._region_id_choice(unique_umi1s, unique_umi2s)
        num_unique_umis = len(unique_umis)

        db = self._memory.allocate_array(
            "db", shape=(num_unique_umis, vector_length), dtype=np.uint8, zero_init=True
        )

        for idx, umi in enumerate(unique_umis):
            db[idx, 0 : len(umi)] = np.frombuffer(umi, dtype=np.uint8)

        read_counts = self._memory.allocate_array(
            "read_counts", shape=(num_unique_umis,), dtype=np.uint16, zero_init=False
        )

        # Sum the read counts for all molecules that map to the same unique umi.
        # These read_counts are than used as weights in the network based deduplication.
        read_counts[:] = np.bincount(
            self._db_to_molecule_idx,
            weights=data["read_count"],
            minlength=num_unique_umis,
        )

        yield db, read_counts

        self._memory.unlink_buffer("db")
        self._memory.unlink_buffer("read_counts")
        self._umi1_data = None
        self._umi2_data = None
        self._db_to_molecule_idx_umi1 = None
        self._db_to_molecule_idx_umi2 = None

    @cached_property
    def max_hamming_mismatches(self) -> int:
        """Return the maximum hamming distance for the number of allowed mismatches."""
        return self.max_mismatches * 2

    def process_file(self, path: Path, output: Path):
        """Collapse UMIs from a single parquet file containing molecular data."""
        logger = self._logger

        logger.info("Loading data from parquet: %s", str(path))
        df = pl.read_parquet(path)

        # Handle legacy data
        if "marker1" in df.columns or "marker2" in df.columns:
            df = df.rename({"marker1": "marker_1", "marker2": "marker_2"}, strict=False)

        self._stats.add_input_file(path, df.shape[0])

        logger.info("Partitioning dataset by marker pairs")

        partition_key = self._region_id_choice("marker_1", "marker_2")

        # Partition the data by the marker1 and marker2 columns and store each partition as a separate DataFrame
        partitions = df.partition_by(partition_key, as_dict=True, include_key=True)

        num_partitions = len(partitions)
        logger.info("Dataset partitioned into %s subsets", num_partitions)

        # Initialize the parquet writer
        writer = pyarrow.parquet.ParquetWriter(
            output, schema=self._output_schema, compression="zstd", compression_level=6
        )

        with writer as w:
            for idx, (group_by_keys, data) in enumerate(partitions.items()):
                # mypy cannot infer the type of group_by_keys
                marker_id = typing.cast(int, group_by_keys[0])
                stats = self._process_marker_group(
                    idx,
                    num_groups=num_partitions,
                    marker=marker_id,
                    data=data,
                    writer=w,
                )

                stats.region_id = self.region_id
                self._stats.add_marker_stats(stats)

    def _process_marker_group(
        self,
        idx: int,
        num_groups: int,
        marker: int,
        data: pl.DataFrame,
        writer: pa.parquet.ParquetWriter,
    ) -> MarkerCorrectionStats:
        """Process a group of reads from a single marker.

        :param idx: The index of the group in the partition
        :param num_groups: The total number of groups in the partition
        :param marker: The index of the marker in the panel
        :param data: The data for the group. A dataframe.
        :param writer: The parquet writer to stream output to.
        """
        starttime = time.time()
        logger = self._logger

        marker_name = self.panel.markers[marker]
        input_reads_count = int(data["read_count"].sum())

        local_stats = MarkerCorrectionStats(
            marker=marker_name,
            region_id=self.region_id,
            input_reads=input_reads_count,
            input_molecules=len(data),
        )

        logger.info(
            "Processing %s for %s reads from marker %s (%s/%s)",
            self.region_id,
            input_reads_count,
            marker_name,
            idx + 1,
            num_groups,
        )

        # Allocate shared memory for the binary vectors that make up the search space
        # and the read counts for each molecule. These are loaded with a context manager
        # to ensure proper cleanup
        with self._init_shared_memory(data) as shared_memory_data:
            db, read_counts = shared_memory_data

            local_stats.input_unique_umis = len(db)

            logger.info(
                "Building binary index of %s for %s from %s unique umis",
                self.region_id,
                marker_name,
                len(db),
            )

            # Build a binary index for given marker
            index = self._simsearch.build_index(db)

            logger.info(
                "Querying binary index of %s for %s for similar molecules",
                self.region_id,
                marker_name,
            )

            distances, indices = self._simsearch.search(index, db, min(16, len(db)))

            logger.info(
                "Building sparse adjacency matrix of %s for %s",
                self.region_id,
                marker_name,
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

            logger.info(
                f"Collapsing clustered reads ({self.algorithm}) of %s for %s",
                self.region_id,
                marker_name,
            )

            corrected_umis, cluster_stats = self._process_molecule_graph(
                adjacency, marker_name=marker_name, local_stats=local_stats
            )

            logger.info(
                "Preparing collapsed output for %s from markers %s (%s/%s)",
                self.region_id,
                marker_name,
                idx + 1,
                num_groups,
            )

            # Get the original 2bit encoded UEI
            uei_bytes = self._uei_region_extractor(data["molecule"])
            uei_data = self._compress_uei_embedding(uei_bytes)
            # Get the original 2bit encoded UMIs
            original_umi1_data = self._umi1_data[self._db_to_molecule_idx_umi1]
            original_umi2_data = self._umi2_data[self._db_to_molecule_idx_umi2]

            new_data = data.with_columns(
                marker_1=self._marker_idx_to_name(data["marker_1"].to_numpy()),
                marker_2=self._marker_idx_to_name(data["marker_2"].to_numpy()),
                original_umi1=original_umi1_data,
                original_umi2=original_umi2_data,
                uei=uei_data,
            ).drop(["molecule"])

            table = new_data.to_arrow()

            corrected_name = self._region_id_choice("corrected_umi1", "corrected_umi2")
            table = table.add_column(
                6,
                self._output_schema.field(corrected_name),
                corrected_umis,
            )
            table = table.cast(self._output_schema)

            logger.info("Streaming %s records to parquet", len(table))
            writer.write_table(table)

            logger.info(
                "Completed processing %s for %s reads from marker %s (%s/%s)",
                self.region_id,
                input_reads_count,
                marker_name,
                idx + 1,
                num_groups,
            )

        return cluster_stats

    def _process_molecule_graph(
        self, csgraph, marker_name: str, local_stats: MarkerCorrectionStats
    ) -> tuple[pa.Array, MarkerCorrectionStats]:
        """Determine connected components and collapse the UMIs for each component.

        :param csgraph: The sparse adjacency matrix of the connected components
        :param local_stats: The statistics object for this marker pair
        :return: A tuple with a pyarrow Table with for each read the corrected UMI
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

        memory_view = self._memory.read_only_view()
        embedding = self._embedding

        job_results: list[_DistributedResults]

        local_stats.input_unique_umis = len(self._umi_data)

        # Allocate output vectors in shared memory or as normal numpy arrays
        if self._effective_threads > 1 and n_components >= self.min_parallel_chunk_size:
            chunk_size = max(
                self.min_parallel_chunk_size, n_components // self._effective_threads
            )
            _logger.info(
                "Detecting unique %s from %s connected components for %s (parallel: [chunk_size=%s])",
                self.region_id,
                n_components,
                marker_name,
                chunk_size,
            )
            subranges = list(_split_chunks(n_components, chunk_size=chunk_size))

            job_results = self._parallel_worker(
                joblib.delayed(self._record_group_worker_fn)(
                    r, groups[slice(*r)], memory_view, embedding, n_molecules
                )
                for r in subranges
            )
        else:
            _logger.info(
                "Detecting unique %s from %s connected components for %s (serial)",
                self.region_id,
                n_components,
                marker_name,
            )

            job_results = [
                self._record_group_worker_fn(
                    (0, n_components), groups, memory_view, embedding, n_molecules
                )
            ]

        # An array of indices mapping the original unique reads into the corrected umi space
        # Initialize with the identity mapping
        corrected_umi_map = self._process_job_results(
            job_results, local_stats, n_molecules
        )

        _logger.info(
            "Corrected %s reads of %s (%.2f%%) or %s umis of %s unique umis (%.2f%%)",
            local_stats.corrected_reads,
            local_stats.input_reads,
            local_stats.corrected_reads_fraction * 100,
            local_stats.corrected_unique_umis,
            local_stats.input_unique_umis,
            local_stats.corrected_unique_umis_fraction * 100,
        )

        corrected_umis = self._build_corrected_umi_array(corrected_umi_map)

        return corrected_umis, local_stats

    def _process_job_results(
        self, job_results: list[_DistributedResults], local_stats, n_molecules: int
    ):
        """Process the results of the parallel job and build the corrected UMI map.

        Args:
            job_results: A list of _DistributedResults containing the results of the parallel job.
            local_stats: The statistics object for this marker pair.
            n_molecules: The number of molecules in the dataset.

        Returns:
            A numpy array with mapping input UMI indices to the corrected UMI.

        """
        # An array of indices mapping the original unique reads into the corrected umi space
        # Initialize with the identity mapping

        corrected_umi_map = np.zeros((n_molecules,), dtype=np.int64)

        for idx, r in enumerate(job_results):
            if r.start == r.stop:
                continue

            # Flatten the list of numpy indices that need to be corrected into a single array
            indices = np.concatenate(r.unique_molecule_ids, dtype=np.int64)

            # Repeat the unique umis for the number of umis that need to be corrected to that umi
            repeat_lengths = [len(x) for x in r.unique_molecule_ids]
            values = np.repeat(r.representative_indices, repeat_lengths)

            # Assign the values to the corrected_umi_map
            corrected_umi_map[indices] = values

            local_stats.corrected_reads += r.corrected_reads_count
            local_stats.corrected_unique_umis += r.collapsed_umi_count
            local_stats.output_unique_umis += r.output_umi_count

        return corrected_umi_map

    def _build_corrected_umi_array(
        self, corrected_umi_map: npt.NDArray[np.uint64]
    ) -> pyarrow.Array:
        """Build a table with the corrected UMI sequences.

        Args:
            corrected_umi_map: A numpy array containing the corrected UMI indices.

        Returns:
            A pyarrow Table with the original and corrected UMI encoded sequences.

        """
        # broadcast the corrected umi map to the original molecule indices using the reverse indices
        corrected_umi_indices = corrected_umi_map[self._db_to_molecule_idx]
        corrected_umis = self._umi_data[corrected_umi_indices]
        return pa.array(corrected_umis, type=pa.uint64())

    @staticmethod
    def _record_group_worker_fn(
        subrange: tuple[int, int],
        component_indices,
        memory: ReadOnlySharedMemoryRegistry,
        embedding: PNAEmbedding,
        n_molecules: int,
    ):
        """Process a batch of connected components.

        The database and read counts are loaded from shared memory to reduce
        python multiprocessing IPC overhead.

        :param subrange: The range of connected components to process.
            A tuple with the start and stop indices.
        :param component_indices: A list of lists containing the indices
            in the database and read counts vector for each connected component.
        :param db_shm: The shared memory buffer containing the binary vectors.
        :param read_counts_shm: The shared memory buffer containing the read counts.
        :param db_size: The size of the binary vectors memory buffer in bytes.
        :param embedding: The PNAEmbedding instance for encoding/decoding vectors.
        """
        db = memory.get_array("db")
        read_count = memory.get_array("read_counts")
        corrected_reads = 0
        corrected_umis = 0
        corrected_molecules = 0

        subrange_len = subrange[1] - subrange[0]
        unique_molecules_indices = []
        representative_indices: npt.NDArray[np.int64] = np.ndarray(
            shape=(subrange_len,), dtype=np.int64
        )

        for idx in range(subrange_len):
            this_cmp_indices = component_indices[idx]

            # Aggregate the read_count for this connected component
            molecule_read_counts = read_count[this_cmp_indices]
            component_reads = np.sum(molecule_read_counts)

            # Select the molecule with the most support as "representative" of the component
            # TODO: Voting on a representative molecule per nucleotide?
            representative_idx = np.argmax(molecule_read_counts)
            comp_idx = this_cmp_indices[representative_idx]
            corrected_reads += int(
                component_reads - molecule_read_counts[representative_idx]
            )

            representative_indices[idx] = comp_idx
            unique_molecules_indices.append(this_cmp_indices)
            corrected_umis += len(this_cmp_indices) - 1

        res = _DistributedResults(
            start=subrange[0],
            stop=subrange[1],
            representative_indices=representative_indices,
            unique_molecule_ids=unique_molecules_indices,
            corrected_reads_count=corrected_reads,
            collapsed_umi_count=corrected_umis,
            output_umi_count=len(unique_molecules_indices),
        )
        return res
