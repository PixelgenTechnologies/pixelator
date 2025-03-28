"""Common functions for collapsing data.

Copyright © 2024 Pixelgen Technologies AB.
"""

import dataclasses
import typing
from pathlib import Path
from typing import Generator, Iterable

import faiss
import numpy as np
import scipy
from numpy import typing as npt


def build_binary_index(data) -> faiss.IndexBinary:
    """Build a binary index for a batch of embeddings.

    The input data is a matrix of shape (n, d) where n is the number
    of embeddings and d is the dimensionality of each embedding.

    Depending on the size of the input data, either a flat or HNSW index is built.

    :param data: The data to build the index for.
    """
    """Build a binary index for the given data."""
    db_len = data.shape[0]
    d = data.shape[1] * 8

    if db_len < 5_000:
        index = faiss.IndexBinaryFlat(d)
    else:
        index = faiss.IndexBinaryHNSW(d, 16)

    # Add data all at once to utilise fast internal parallelization in FAISS
    index.add(data)

    return index


class FAISSBackend:
    """A backend for building and searching binary indexes using FAISS."""

    def __init__(self, threads: int | None = None):
        """Initialize the FAISS Backend.

        Args:
            threads: The number of threads to use. -1 or None will use all available threads

        """
        if threads is not None and threads >= 1:
            faiss.omp_set_num_threads(threads)

    def build_index(self, data: np.ndarray) -> faiss.IndexBinary:
        """Build a binary index for the given data.

        Args:
            data: The data to build the index for.

        Returns:
            The built index.

        """
        return build_binary_index(data)

    def search(self, index: faiss.IndexBinary, data: np.ndarray, k: int) -> np.ndarray:
        """Search the index for the k nearest neighbors of the given data.

        Args:
            index: The index to search.
            data: The data to search for.
            k: The number of nearest neighbors to find.

        Returns:
            The indices of the k nearest neighbors for each data point.
            Indices will be -1 when less than k neighbors are found.

        """
        return index.search(data, k)


def _collect_label_array_indices(
    labels: npt.NDArray[np.int32], n_components: int
) -> npt.NDArray[np.object_]:
    """Collect an array with a label for each item into an array of indices for each label.

    :param labels: the labels for each item
    :param n_components: Number of labels in the label array
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

    :param n_components: The size of the input range
    :param chunk_size: The size of each chunk
    :yield: A tuple of (start, stop) indices for each chunk
    """
    pos = 0
    complete_chunks = n_components // chunk_size

    for i in range(complete_chunks):
        yield (pos, pos + chunk_size)
        pos += chunk_size

    yield (pos, n_components)


class SplitFilesResult(typing.NamedTuple):
    """Tuple with a list per category of collapse input parquet files."""

    m1: list[Path]
    m2: list[Path]
    paired: list[Path]


def _split_files_per_marker_files(inputs: Iterable[Path | str]) -> SplitFilesResult:
    """Split the input files into separate lists based on the marker used.

    Args:
        inputs: The input files to split.

    Returns:
        A tuple of lists containing the input files for each marker.

    """
    umi1_files: list[Path] = []
    umi2_files: list[Path] = []
    paired_files: list[Path] = []

    for f in inputs:
        f = Path(f)
        if ".m1." in f.name:
            umi1_files.append(f)
        elif ".m2." in f.name:
            umi2_files.append(f)
        else:
            paired_files.append(f)

    return SplitFilesResult(umi1_files, umi2_files, paired_files)


def split_collapse_inputs(
    parquet_files: Iterable[Path | str],
) -> list[Path] | tuple[list[Path], list[Path]]:
    """Check the input parquet files to determine independent or paired collapsed data.

    Detection is simply based on the presence of ".m1." or ".m2." in the file name.

    :param parquet_files: The parquet files to check.
    :returns: Either a single list when the input data has been collapsed using the "paired" strategy,
        or a tuple of two lists when collapsed using "independent" strategy.
    :raises ValueError: If the input data contains a mix of independent and paired collapsed data.
    """
    # Check parquet files first
    umi1_files, umi2_files, paired_files = _split_files_per_marker_files(parquet_files)

    is_independent_collapsed = len(umi1_files) > 0 and len(umi2_files) > 0
    is_paired_collapsed = len(paired_files) > 0

    if is_paired_collapsed and is_independent_collapsed:
        raise ValueError(
            "Cannot combine from demux/collapse stage with different `--strategy` in the same run."
        )

    if is_paired_collapsed:
        return paired_files

    return umi1_files, umi2_files


MoleculeCollapserAlgorithm = typing.Literal["directional", "cluster"]


def _find_connected_components_directional(
    csgraph,
) -> tuple[int, npt.NDArray[np.int32]]:
    """Find connected components in a graph returned by the directional collapse algorithm.

    :param csgraph: The sparse adjacency matrix of the connected components
    """
    # Return the (weakly) connected components in a directed sparse graph
    n_components, labels = scipy.sparse.csgraph.connected_components(
        csgraph, directed=True, connection="weak", return_labels=True
    )
    return n_components, labels


def _find_connected_components_cluster(
    csgraph,
) -> tuple[int, npt.NDArray[np.int32]]:
    """Find connected components in a graph returned by the "cluster" collapse algorithm.

    :param csgraph: The sparse adjacency matrix of the connected components
    """
    # Return the (weakly) connected components in a directed sparse graph
    n_components, labels = scipy.sparse.csgraph.connected_components(
        csgraph, directed=False, return_labels=True
    )
    return n_components, labels


@dataclasses.dataclass(frozen=True, slots=True)
class CollapseInputs:
    """Output object for :func:`check_collapse_strategy_inputs`."""

    parquet: list[Path] | tuple[list[Path], list[Path]]
    reports: list[Path] | tuple[list[Path], list[Path]]


def check_collapse_strategy_inputs(
    parquet_files: Iterable[Path | str], report_files: Iterable[Path | str]
) -> CollapseInputs:
    """Check the input parquet files to determine independent or paired collapsed data.

    Detection is simply based on the presence of ".m1." or ".m2." in the file name.

    :param parquet_files: The parquet files to check.
    :returns: Either a single list when the input data has been collapsed using the "paired" strategy,
        or a tuple of two lists when collapsed using "independent" strategy.
    :raises ValueError: If the input data contains a mix of independent and paired collapsed data.
    """
    # Check parquet input files
    umi1_files, umi2_files, paired_files = _split_files_per_marker_files(parquet_files)

    # Check report input files
    umi1_report_files, umi2_report_files, paired_report_files = (
        _split_files_per_marker_files(report_files)
    )

    # Sanity checks
    if len(umi1_files) != len(umi1_report_files):
        raise ValueError(
            "Mismatched number of UMI1 parquet files and UMI1 report files."
        )

    if len(umi2_files) != len(umi2_report_files):
        raise ValueError(
            "Mismatched number of UMI2 parquet files and UMI2 report files."
        )

    if len(paired_files) != len(paired_report_files):
        raise ValueError("Mismatched number of parquet files and report files.")

    # Determine independent or paired strategy from the input files
    is_independent_collapsed = len(umi1_files) > 0 and len(umi2_files) > 0
    is_paired_collapsed = len(paired_files) > 0

    if is_paired_collapsed and is_independent_collapsed:
        raise ValueError(
            "Cannot combine from demux/collapse stage with different `--strategy` in the same run."
        )

    if is_paired_collapsed:
        return CollapseInputs(parquet=paired_files, reports=paired_report_files)

    return CollapseInputs(
        parquet=(umi1_files, umi2_files), reports=(umi1_report_files, umi2_report_files)
    )
