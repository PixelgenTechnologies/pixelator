"""Collapse reads into their underlying molecules.

This module contains functions for the collapse and error correction of MPX data
(from FASTQ)

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
import tempfile
from collections import Counter, defaultdict
from typing import (
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyfastx
from annoy import AnnoyIndex
from umi_tools._dedup_umi import edit_distance
from umi_tools.network import breadth_first_search

from pixelator.collapse.constants import SEED
from pixelator.exception import FileFqGzEmpty
from pixelator.types import PathType
from pixelator.utils import gz_size

logger = logging.getLogger(__name__)

np.random.seed(SEED)

UniqueFragment = str
UpiB = str
UniqueFragmentToUpiB = Dict[UniqueFragment, List[UpiB]]
UniqueFragmentAndCount = Tuple[str, int]


def build_annoytree(data: npt.NDArray[np.uint8], n_trees: int = 10) -> AnnoyIndex:
    """Build an Annoy tree index.

    Create an `Annoy` tree [1]_ using the given data matrix (an array of DNA sequences
    in binary format) and using hamming distance as the distance metric in the Annoy
    index.

    The number of trees (`n_trees`) determine the accuracy of the nearest neighbour
    estimations.

    .. [1] https://github.com/spotify/annoy

    :param data: the array of sequences (n_ele, n_features) in binary format
    :param n_trees: the number of Annoy trees to build. Default: 10
    :returns: an AnnoyIndex object
    :rtype: AnnoyIndex
    """
    logger.debug("Building annoy tree of shape %i,%i", data.shape[0], data.shape[1])

    tree = AnnoyIndex(data.shape[1], "hamming")
    tree.set_seed(SEED)
    for i in range(data.shape[0]):
        tree.add_item(i, data[i, :])
    tree.build(n_trees, n_jobs=-1)

    logger.debug("Annoy tree built")
    return tree


def build_binary_data(seqs: List[str]) -> npt.NDArray[np.uint8]:
    """Build a binary matrix from a list of sequences using two-bit encoding.

    Convert a list of DNA sequences (str) to binary sequences
    using two bit encoding, the converted sequences are returned
    as an `np.array` (n_ele, n_features).

    :param seqs: a list of DNA sequences
    :returns: a numpy array of binary sequences
    :rtype: npt.NDArray[np.uint8]
    """
    logger.debug("Transforming %i sequences to binary form", len(seqs))

    tr = {"A": "00", "C": "01", "G": "10", "T": "11"}
    n_feat = len(seqs[0])
    n_ele = len(seqs)
    data = np.zeros((n_ele, n_feat * 2), np.uint8)
    for i, seq in enumerate(seqs):
        data[i] = [int(b) for s in seq for b in tr[s]]

    logger.debug("Sequences transformed")
    return data


def get_representative_sequence_for_component(
    components: List[Set[UniqueFragment]], counts: Dict[UniqueFragment, int]
) -> Generator[Tuple[UniqueFragment, int], None, None]:
    """Take the representative sequence from a component based on its counts.

    Given a list of components (i.e. sequences that presumably belong to the
    same original molecule), and the counts of occurrences for each of these,
    return an iterator of tuples with the representative sequence for that
    component and the number of reads collapsed for that umi.

    The representative molecule is selected as the one with the most associated
    upib's, and if there is a tie the lexicographically smallest sequence is picked
    to make the results reproducible between runs.

    :param components: a list of components as produced from `get_connected_components`
    :param counts: a dictionary of the counts of each unique fragment
    :yields: the representative sequence and the size of the component as a tuple
    :rtype: Generator[Tuple[UniqueFragment, int], None, None]
    """
    for component in components:
        counts_and_sequences = defaultdict(list)
        for seq in component:
            counts_and_sequences[counts[seq]].append(seq)
        max_count = max(counts_and_sequences.keys())
        yield (
            sorted(counts_and_sequences[max_count])[0],
            len(component),
        )


# This code snipped has been obtained from:
# https://github.com/CGATOxford/umi-tools
def get_connected_components(
    graph: Dict[UniqueFragment, List[UniqueFragment]], counts: Dict[UniqueFragment, int]
) -> List[Set[UniqueFragment]]:
    """Get connected components from a graph of sequences.

    Find the connected components from the sequence graph (represented by an adjacency
    dictionary). Each node in the graph is a sequence and each edge represents that
    those sequences similar (e.g. have a low edit distance between each other). Thus,
    a connected component in this graph can be seen as a group of sequences deriving
    from the same original molecule.

    The function will return the list of connected components of each sequence
    (not counting sequences twice).

    :param graph: a dictionary of sequence to sequences (within same distance)
    :param counts: a dictionary of counts (copies) for each sequence in `graph`
    :returns: a list of sets of sequences where each set represents a
              connected component
    :rtype: List[Set[UniqueFragment]]
    """
    found = set()
    components = []
    for node in sorted(graph, key=lambda x: counts[x], reverse=True):
        if node not in found:
            component = breadth_first_search(node, graph)
            found.update(component)
            components.append(component)
    return components


def identify_fragments_to_collapse(
    seqs: List[UniqueFragment],
    min_dist: int,
    max_neighbours: int,
) -> Dict[UniqueFragment, List[UniqueFragment]]:
    """Identify fragments to collapse by approximate nearest neighbourhood search.

    Tries to identify all sequences in the given list of sequences
    (`seqs`) within a hamming distance `min_dist` of each other, and group them
    together.

    An `Annoy` index (approximate nearest neighbour) is used to avoid having to
    compute the distances between all sequence pairs. This means that this function is
    not guaranteed to return all sequences within a given hamming distance.

    The parameter `max_neighbours` is used to control the maximum number of
    neighbours to look for. The higher this number the more likely you are
    to find all sequences within a hamming distance of `min_dist`, but the search
    will also be slower. The number of actual neighbours searched is determined as
    follows: pick the minimum number of `max_neighbours` and 10% of the input data,
    but at least search 10 neighbours of each sequence.

    :param seqs: a list of sequences to be grouped
    :param min_dist: the hamming distance threshold (i.e. the mismatches
                     between two sequences)
    :param max_neighbours: the number of neighbours to use in the Annoy index
    :returns: a dictionary with fragments as keys and a list of their adjoining
              fragments as values
    :rtype: Dict[UniqueFragment, List[UniqueFragment]]
    """
    logger.debug("Computing adjancency sequences from %i elements", len(seqs))

    # use a nearest neighbours tree to reduce the search space
    # TODO this approach requires memory (n_ele * len(seq) * 2)
    # we could add an alternative approach using on binary data
    # but this has proven to be slower
    data = build_binary_data(seqs)
    tree = build_annoytree(data)

    neighbours = max(10, max_neighbours)
    logger.debug("The number of neighbours was set to %s", neighbours)

    # iterate the neighbours of each sequence to obtain
    # the adjacency list (dictionary of lists)
    adj_list = {seq: [] for seq in seqs}  # type: Dict[str, List[str]]
    for i, seq1 in enumerate(seqs):
        if i % 100000 == 0:
            logger.debug("Processed 100,000 reads...")
        idxs, estimated_distances = tree.get_nns_by_item(
            i, n=neighbours, search_k=-1, include_distances=True
        )
        for idx, dist in zip(idxs, estimated_distances):
            seq2 = seqs[idx]

            # No need to add yourself to the adjacency list
            if seq1 == seq2:
                continue

            # Escape early if estimated distance is more than twice the
            # minimum distance allowed, since there is a very low risk
            # that it will be a dist < min distance when we calculate the
            # true edit distance below.
            # This works since the distances returned from Annoy above
            # are returned on order of lowest to highest distance.
            # Please not that since we are using Hamming distance,
            # the min_dist * 2 is the equivalent of correcting min_dist * 2
            # base pairs.
            if dist > min_dist * 2:
                break

            # The`dist` provided by Annoy above is only an approximation of the
            # distance between the sequences. Because of this we need to double
            # check it here
            if edit_distance(seq1.encode("utf-8"), seq2.encode("utf-8")) <= min_dist:
                adj_list[seq1].append(seq2)

    logger.debug("Approximate sequence adjacency computed")
    return adj_list


def collapse_sequences_unique(
    seq_dict: UniqueFragmentToUpiB,
) -> Generator[UniqueFragmentAndCount, None, None]:
    """Get all fragments.

    Let each key in `seq_dict` represent it's own sequence. This is equivalent
    to not collapsing the sequences.

    :param seq_dict: the fragment to upib dict
    :yield UniqueFragmentAndCount: a unique fragment and the number of reads collapsed
    :rtype: Generator[UniqueFragmentAndCount, None, None]
    """
    logger.debug("Picking all unique sequences (i.e. no collapsing is carried out)")

    for seq in seq_dict.keys():
        yield (seq, len(seq_dict[seq]))


def collapse_sequences_adjacency(
    seq_dict: UniqueFragmentToUpiB,
    max_neighbours: int,
    min_dist: int,
) -> Iterator[UniqueFragmentAndCount]:
    """Collapse sequences based on their adjacency.

    Tries to identify all fragments that represent the same underlying
    molecule by trying to find sequences that are less than or equal to `min_distance`
    apart. The distance measure used is hamming distance.

    Each original collapsed molecule is represented by the fragment that is associated
    with the largest number of upib's.

    It does so by doing approximate nearest neighbour estimation, and then checking
    the exact distance between the neighbours. For more information on this see
    `identify_fragments_to_collapse`.

    :param seq_dict: a dictionary mapping unique fragments to their
                     corresponding upib's
    :param max_neighbours: the maximum number of neighbours to search in the approximate
                           nearest neighbour search
    :param min_dist: the hamming distance threshold (i.e. the mismatches
                     between two sequences)
    :returns: An iterator of the of collapsed molecules, and their original counts
    :rtype: Iterator[UniqueFragmentAndCount]
    """
    logger.debug("Collapsing %i sequences", len(seq_dict))

    counts = {k: len(v) for k, v in seq_dict.items()}
    seqs = list(counts.keys())

    adj_list = identify_fragments_to_collapse(
        seqs=seqs, max_neighbours=max_neighbours, min_dist=min_dist
    )
    full_components = get_connected_components(adj_list, counts)
    components = get_representative_sequence_for_component(full_components, counts)
    return components


def create_fragment_to_upib_dict(
    input_file: str,
    upia_start: int,
    upia_end: int,
    upib_start: int,
    upib_end: int,
    umia_start: Optional[int] = None,
    umia_end: Optional[int] = None,
    umib_start: Optional[int] = None,
    umib_end: Optional[int] = None,
) -> UniqueFragmentToUpiB:
    """Create a dict mapping fragments to upib's.

    Parses a fastq file with pixel data and extracts the UPIA, UPIB,
    UMIA and UMIB and then returns a dictionary of UMI+UPIA -> [UPIB].

    UMIA will be ignored if any of its positions is None.
    UMIB will be ignored if any of its positions is None.

    :param input_file: path to the file to read
    :param upia_start: the 0-based start position of UPIA
    :param upia_end: the 1-based end position of UPIA
    :param upib_start: the 0-based start position of UPIB
    :param upib_end: the 1-based end position of UPIB
    :param umia_start: the 0-based start position of UMIA
    :param umia_end: the 1-based end position of UMIA
    :param umib_start: the 0-based start position of UMIB
    :param umib_end: the 1-based end position of UMIB
    :returns: a dictionary of with the sequence of umi+upia
              as keys and the list of associated upibs as values
    :rtype: UniqueFragmentToUpiB
    :raises FileFqGzEmpty: when the file is empty
    :raises RuntimeError: when there is a error parsing the file
    """
    logger.debug("Extracting umi-upi sequences from %s", input_file)

    has_umia = umia_start is not None and umia_end is not None
    has_umib = umib_start is not None and umib_end is not None

    seqs_dict = defaultdict(list)
    try:
        for _, seq, _ in pyfastx.Fastq(input_file, build_index=False):
            upia = seq[upia_start:upia_end]
            upib = seq[upib_start:upib_end]
            if has_umia and has_umib:
                umi = seq[umia_start:umia_end] + seq[umib_start:umib_end]
            elif has_umia:
                umi = seq[umia_start:umia_end]
            elif has_umib:
                umi = seq[umib_start:umib_end]
            else:
                umi = ""
            seq = umi + upia
            seqs_dict[seq].append(upib)
    except Exception as exc:
        gzlen = gz_size(input_file)
        if gzlen == 0:
            raise FileFqGzEmpty(
                f"File {input_file} is empty", input_file, gzlen
            ) from exc
        raise RuntimeError(
            f"There was an error '{str(exc)}' parsing {input_file}"
        ) from exc

    logger.debug("umi-upi sequences were extracted from %s", input_file)
    return seqs_dict


def filter_by_minimum_upib_count(
    unique_reads: UniqueFragmentToUpiB, min_count: int
) -> UniqueFragmentToUpiB:
    """Filter fragment to upib's dictionary by minimum upib count.

    Filter reads from the input dictionary requiring at least `min_count` or more
    instances of the upib's to keep it.

    :param unique_reads: a dictionary of fragments and their corresponding upib's
    :param min_count: the minimum number of upib's per fragment required to keep it
    :returns: A filtered instance of the input dictionary
    :rtype: UniqueFragmentToUpiB
    """
    unique_reads = {k: v for k, v in unique_reads.items() if len(v) >= min_count}
    # in case there are no reads after filtering
    if not unique_reads:
        logger.warning(
            (
                "The input file %s does not any contain"
                "reads after filtering by count >= %i"
            ),
            input,
            min_count,
        )
    return unique_reads


def create_edgelist(
    clustered_reads: Iterable[UniqueFragmentAndCount],
    unique_reads: UniqueFragmentToUpiB,
    umia_start: Optional[int],
    umia_end: Optional[int],
    umib_start: Optional[int],
    umib_end: Optional[int],
    marker: str,
    sequence: str,
) -> pd.DataFrame:
    """Create an edgelist.

    Create an egdelist of the MPX graph based on the unique reads found,
    and their clustering.

    `clustered_reads` should be an iterable where each element is a tuple
    of reads that belong to the same cluster, i.e. are derived from the same
    unique fragment.

    `unique_reads` should contain a mapping from each of these unique fragments
    to their corresponding upib's.

    This will create an edgelist where each unique fragment is a row, with information
    about:
        - `upia`, the upia of the fragment
        - `upib`, the upib of the fragment
        - `umi`, the umi of the fragment
        - `count`, the number of upib's associated with the fragment
        - `umi_unique_count`, the number of unique molecules (based on upia+umi)
           associated with the fragment
        - `upi_unique_count`, the number of unique upib's associated with the fragment
        - `marker`, the marker associated with this fragment
        - `sequence`, the antibody DNA-oligo sequence of the marker associated
           with the fragment

    :param clustered_reads: An iterable of tuples of unique fragments
    :param unique_reads: A UniqueFragmentToUpiB dictionary
    :param umia_start: the start position of upia
    :param umia_end: the end position of upia
    :param umib_start: the start position of upib
    :param umib_end: the end position of upib
    :param marker: the marker
    :param sequence: the sequence of the marker:
    :returns: a dataframe representing the edgelist of the mpx graph
    :rtype: pd.DataFrame
    """
    # get the umi sizes to do the split
    if umia_start is not None and umia_end is not None:
        umia_size = umia_end - umia_start
    else:
        umia_size = 0
    if umib_start is not None and umib_end is not None:
        umib_size = umib_end - umib_start
    else:
        umib_size = 0
    umi_size = umia_size + umib_size

    # iterate each collapsed umi+upia to get the upib
    # with the highest count and store in a list
    def data():
        for cluster_representative_fragment, fragment_count in clustered_reads:
            # get all the upis from the sequence in the cluster
            upis = unique_reads[cluster_representative_fragment]

            # take the most abundant umi-upi
            umi_upia = cluster_representative_fragment
            umi = umi_upia[:umi_size]
            upia = umi_upia[umi_size:]

            # take the most common upib from the list
            unique_upis = Counter(upis)
            upib, _ = unique_upis.most_common(1)[0]

            # count (number of molecules) is the number
            # of upis in the umi-upia cluster
            count = len(upis)
            umi_unique_count = fragment_count
            upi_unique_count = len(unique_upis)

            # update data array
            yield (upia, upib, umi, count, umi_unique_count, upi_unique_count)

    # create an edge list (pd.DataFrame) with the collapsed sequences
    df = pd.DataFrame(
        data=data(),
        columns=[
            "upia",
            "upib",
            "umi",
            "count",
            "umi_unique_count",
            "upi_unique_count",
        ],
    )
    df.insert(3, "marker", marker)
    df.insert(4, "sequence", sequence)

    return df


def write_tmp_feather_file(df: pd.DataFrame) -> PathType:
    """Write the dataframe to a feather file in the OS tmpdir.

    :param df: the data frame to write
    :returns: path of the file written
    :rtype: PathType
    """
    # create a temporary edge list and save it to a temp file
    tmp_file = tempfile.mkstemp(suffix=".feather")[1]
    df.to_feather(tmp_file)
    return tmp_file


def collapse_fastq(
    input_file: str,
    algorithm: Literal["unique", "adjacency"],
    marker: str,
    sequence: str,
    upia_start: int,
    upia_end: int,
    upib_start: int,
    upib_end: int,
    umia_start: Optional[int],
    umia_end: Optional[int],
    umib_start: Optional[int],
    umib_end: Optional[int],
    max_neighbours: Optional[int] = None,
    mismatches: Optional[int] = None,
    min_count: Optional[int] = None,
) -> Optional[PathType]:
    """Collapses reads from a fastq file.

    Takes a fastq file as input and collapses its
    reads by UMI + UPIA (first) to then take the most abundant
    UPIB. The collapsed (error corrected) reads are saved as a
    `pd.DataFrame` to a temporary file with the following columns:

    upia,upib,umi,marker,count,umi_unique_count,upi_unique_count

    The function returns the path to the file or None if the input
    fastq file is empty or corrupted.

    The location (position) of the UPIs and UMIs must be provided.

    The function uses some of the functionalities from `umi-tools` [1]_.

    When `algorithm` is `unique` only the unique sequences will be returned
    (no error correction), otherwise sequences will be collapsed based
    on a hamming distance (`mismatches`), using an approximate nearest
    neighbour search based on Annoy [2]_.

    .. [1] Smith, Tom, Andreas Heger, and Ian Sudbery. 2017. “UMI-Tools: Modeling
           Sequencing Errors in Unique Molecular Identifiers to Improve
           Quantification Accuracy.” Genome Research 27 (3): 491–99.

    .. [2] https://github.com/spotify/annoy

    :param input_file: the path to the fastq file containing MPX amplicons
                       (must contain the UPI and UMI)
    :param algorithm: the collapsing algorithm to use (unique or adjacency)
    :param marker: the antibody tag to append to the output
    :param sequence: the barcode sequence to append to the output
    :param upia_start: the start position (0-based) of UPIA
    :param upia_end: the end position (1-based) of UPIA
    :param upib_start: the start position (0-based) of UPIB
    :param upib_end: the end position (1-based) of UPIB
    :param umia_start: the start position (0-based) of UMIA
                       (if None UMIA will be ignored)
    :param umia_end: the end position (1-based) of UMIA
                     (if None UMIA will be ignored)
    :param umib_start: the start position (0-based) of UMIB
                      (if None UMIB will be ignored)
    :param umib_end: the end position (1-based) of UMIB
                     (if None UMIB will be ignored)
    :param max_neighbours: the number of neighbours used in the approximate nearest
                           neighbour search
    :param mismatches: the number of mismatches allowed between sequences
    :param min_count: discard reads with a count lower than this
    :returns: a str containing the path to the edge list file
    :rtype: Optional[PathType]
    :raises AssertionError: invalid input
    :raises RuntimeError: raises an exception
    """
    if algorithm not in ["unique", "adjacency"]:
        raise AssertionError(f"Invalid value {algorithm} for algorithm")

    if algorithm == "adjacency":
        if not max_neighbours or not mismatches:
            raise AssertionError(
                (
                    'When `algorithm` is "adjacency", `max_neighbours`'
                    "and `mismatches` must be set'"
                )
            )

    logger.debug("Collapsing reads from %s", input_file)

    # parse input fastq file to extract a dict of umi+upia -> [upib]
    # TODO: this dictionary is memory heavy, a possible solution is to
    # replace it by a disk-based dictionary but that will really affect
    # the running time. Another approach would be to create a class with
    # a look up table and store the sequences in a list of tuples
    try:
        unique_reads = create_fragment_to_upib_dict(
            input_file,
            upia_start,
            upia_end,
            upib_start,
            upib_end,
            umia_start,
            umia_end,
            umib_start,
            umib_end,
        )
    except FileFqGzEmpty as err:
        # we want to allow empty files
        logger.warning(str(err.msg))
        return None

    # In case there are no reads in the input file
    if not unique_reads:
        # we want to allow empty files
        logger.warning("The input file %s does not contain data", input_file)
        return None

    if min_count and min_count > 1:
        unique_reads = filter_by_minimum_upib_count(unique_reads, min_count)
        if not unique_reads:
            return None

    if algorithm == "adjacency":
        clustered_reads = collapse_sequences_adjacency(
            seq_dict=unique_reads,
            max_neighbours=max_neighbours,  # type: ignore
            min_dist=mismatches,  # type: ignore
        )
    if algorithm == "unique":
        clustered_reads = collapse_sequences_unique(
            seq_dict=unique_reads,
        )

    edgelist = create_edgelist(
        clustered_reads=clustered_reads,
        unique_reads=unique_reads,
        umia_start=umia_start,
        umia_end=umia_end,
        umib_start=umib_start,
        umib_end=umib_end,
        marker=marker,
        sequence=sequence,
    )
    tmp_file = write_tmp_feather_file(edgelist)
    logger.debug(
        "Saved temporary edge list for %s (%s) with %i rows in %s",
        marker,
        sequence,
        len(edgelist),
        tmp_file,
    )
    return tmp_file
