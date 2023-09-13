"""Tests for collapse.py module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
# pylint: disable=redefined-outer-name


from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pixelator.collapse.process import (
    build_annoytree,
    build_binary_data,
    collapse_fastq,
    collapse_sequences_adjacency,
    collapse_sequences_unique,
    create_edgelist,
    create_fragment_to_upib_dict,
    filter_by_minimum_upib_count,
    get_connected_components,
    get_representative_sequence_for_component,
    identify_fragments_to_collapse,
    write_tmp_feather_file,
)
from pixelator.config import get_position_in_parent
from pixelator.config.config_instance import config
from pixelator.test_utils.simulation import ReadSimulator


def test_create_fragment_to_upib_dict():
    """Test creating a fragment to upib dict."""
    with mock.patch("pixelator.collapse.process.pyfastx") as mock_fastq_reader:

        def mock_reads(*args, **kwargs):
            for read in ["ABCDEFGHIJKLMNO", "ZZZDEFGHIJKLMNO", "XXXDEFGHIJKLMNO"]:
                yield None, read, None

        mock_fastq_reader.Fastq.return_value = mock_reads()

        result = create_fragment_to_upib_dict(
            input_file="/foo/bar",
            upia_start=0,
            upia_end=3,
            upib_start=4,
            upib_end=6,
            umia_start=7,
            umia_end=10,
        )
        assert result == {"HIJABC": ["EF"], "HIJXXX": ["EF"], "HIJZZZ": ["EF"]}


def test_filter_by_minimum_upib_count():
    """Test filtering by a minimum upib count."""
    unique_reads = {
        "HIJNOABC": ["EF", "LL", "JJ"],
        "HIJNOXXX": ["EF", "LL"],
        "HIJNOZZZ": ["EF"],
    }
    result = filter_by_minimum_upib_count(unique_reads=unique_reads, min_count=2)
    assert result == {
        "HIJNOABC": ["EF", "LL", "JJ"],
        "HIJNOXXX": ["EF", "LL"],
    }


def test_create_edgelist():
    """Test creating an edgelist."""
    unique_reads = {
        "HIJNOABC": ["EF", "LL", "JJ"],
        "HIJNOABX": ["EF", "LL", "JJ"],
        "HIJNOXXX": ["EF", "LL"],
        "HIJNOZZZ": ["EF"],
    }
    clustered_sequences = [("HIJNOABC", 2), ("HIJNOXXX", 1), ("HIJNOZZZ", 1)]

    result = create_edgelist(
        clustered_reads=clustered_sequences,
        unique_reads=unique_reads,
        umia_start=0,
        umia_end=3,
        umib_start=4,
        umib_end=6,
        marker="CD4",
        sequence="AAAAA",
    )

    assert_frame_equal(
        result,
        pd.DataFrame.from_records(
            [
                {
                    "upia": "ABC",
                    "upib": "EF",
                    "umi": "HIJNO",
                    "marker": "CD4",
                    "sequence": "AAAAA",
                    "count": 3,
                    "umi_unique_count": 2,
                    "upi_unique_count": 3,
                },
                {
                    "upia": "XXX",
                    "upib": "EF",
                    "umi": "HIJNO",
                    "marker": "CD4",
                    "sequence": "AAAAA",
                    "count": 2,
                    "umi_unique_count": 1,
                    "upi_unique_count": 2,
                },
                {
                    "upia": "ZZZ",
                    "upib": "EF",
                    "umi": "HIJNO",
                    "marker": "CD4",
                    "sequence": "AAAAA",
                    "count": 1,
                    "umi_unique_count": 1,
                    "upi_unique_count": 1,
                },
            ]
        ),
    )


def test_write_tmp_feather_file():
    """Test writing to a temporary feather file."""
    df = pd.DataFrame([1, 2, 3, 4], columns=["monkey"])
    res = Path(write_tmp_feather_file(df))
    assert res.exists()
    assert not res.stat().st_size == 0


def test_get_connected_components():
    """Test getting a connected component."""
    adj_list = {"A": ["B", "C"], "B": ["A"], "C": ["A"], "D": ["E"], "E": ["D"]}
    counts = {"A": 10, "B": 1, "C": 1, "D": 6, "E": 2}
    result = get_connected_components(adj_list, counts)
    assert result == [{"A", "B", "C"}, {"D", "E"}]


def test_get_representative_sequence_for_component():
    """Test get representative sequence for a component."""
    counts = {"A": 10, "B": 1, "C": 1, "D": 2, "E": 6}
    components = [{"A", "B", "C"}, {"D", "E"}]
    result = get_representative_sequence_for_component(
        components=components, counts=counts
    )
    assert list(result) == [("A", 3), ("E", 2)]


def test_identify_fragments_to_collapse():
    """Test identify fragments to collapse."""
    dna_sequences = [
        "AAGTC",
        "AAGTA",
        "AAGTG",  # End first fragment and sequencing errors
        "TTTTT",
        "TTTTA",  # End second fragment and sequencing errors
        "GGGGG",  # End third fragement
    ]
    result = identify_fragments_to_collapse(
        dna_sequences, max_neighbours=10, min_dist=2
    )
    assert result == {
        "AAGTC": ["AAGTA", "AAGTG"],
        "AAGTA": ["AAGTC", "AAGTG"],
        "AAGTG": ["AAGTA", "AAGTC"],
        "TTTTT": ["TTTTA"],
        "TTTTA": ["TTTTT"],
        "GGGGG": [],
    }


def test_collapse_sequences_unique():
    """Test collapsing by unique, i.e. no collapsing."""
    fragments_to_upib = {
        "TATATA": ["GC", "GG", "CG"],
        "GCGCGC": ["AT", "TA", "TT"],
        "ATATAT": ["CG", "CC"],
        "GGGGGG": ["AA"],
    }

    result = collapse_sequences_unique(fragments_to_upib)
    assert list(result) == [("TATATA", 3), ("GCGCGC", 3), ("ATATAT", 2), ("GGGGGG", 1)]


def test_collapse_sequences_adjacency_no_errors():
    """Test collapsing with adjacency when there are no errors."""
    fragments_to_upib = {
        "TATATA": ["GC", "GG", "CG"],
        "GCGCGC": ["AT", "TA", "TT"],
        "ATATAT": ["CG", "CC"],
        "GGGGGG": ["AA"],
    }

    result = collapse_sequences_adjacency(
        fragments_to_upib, max_neighbours=10, min_dist=2
    )
    assert list(result) == [(x, 1) for x in list(fragments_to_upib.keys())]


def test_collapse_sequences_adjacency_with_errors():
    """Test collapsing sequences with errors in them."""
    fragments_to_upib = {
        "TATATA": ["GC", "GG", "CG"],
        "TATATT": ["GC", "GG", "CG"],
        "GCGCGC": ["AT", "TA", "TT"],
        "ATATAT": ["CG", "CC"],
        "GGGGGG": ["AA"],
    }
    expected = set(fragments_to_upib.keys())
    # Remove this since it will match "TATATA"
    # with an edit distance of 1.
    # The reason "TATATA" is picked over "TATATT"
    # is that "TATATA" is lexicographically smaller
    # than "TATATT".
    expected.remove("TATATT")

    result = collapse_sequences_adjacency(
        fragments_to_upib, max_neighbours=10, min_dist=2
    )
    result = sorted(list(result))

    assert result == [
        ("ATATAT", 1),
        ("GCGCGC", 1),
        ("GGGGGG", 1),
        ("TATATA", 2),
    ]


def test_collapse_sequences_adjacency_with_errors_picks_most_abundant():
    """Test that the most abundant sequence is picked when there are errors."""
    fragments_to_upib = {
        "TATATA": ["GC", "GG", "CG"],
        "TATATT": ["GC", "GG"],
        "GCGCGC": ["AT", "TA", "TT"],
        "ATATAT": ["CG", "CC"],
        "GGGGGG": ["AA"],
    }
    expected = set(fragments_to_upib.keys())
    expected.remove("TATATT")

    result = collapse_sequences_adjacency(
        fragments_to_upib, max_neighbours=10, min_dist=2
    )
    assert sorted(list(result)) == [
        ("ATATAT", 1),
        ("GCGCGC", 1),
        ("GGGGGG", 1),
        ("TATATA", 2),  # Note that since "TATATA" has more upib's
        # associated it will be picked
    ]


def test_build_binary_data():
    """Test building binary data from sequences."""
    dna_sequences = ["AAGTC"]
    result = build_binary_data(dna_sequences)
    assert_array_equal(result, np.array([[0, 0, 0, 0, 1, 0, 1, 1, 0, 1]]))


def test_collapse_fastq_algorithm_unique():
    """Small test for using unique algorithm, i.e. no collapsing of reads."""
    with mock.patch("pixelator.collapse.process.pyfastx") as mock_fastq_reader:

        def mock_reads(*args, **kwargs):
            for read in ["AAATTTGGG", "TTTAAAGGG", "GGGCCCAAA"]:
                yield None, read, None

        mock_fastq_reader.Fastq.return_value = mock_reads()

        result_file = collapse_fastq(
            input_file="/foo/bar",
            algorithm="unique",
            marker="CD4",
            sequence="AAAAAAAA",
            upia_start=0,
            upia_end=3,
            upib_start=3,
            upib_end=6,
            umia_start=6,
            umia_end=9,
            umib_start=None,
            umib_end=None,
        )
        result_data = pd.read_feather(result_file)
        assert_frame_equal(
            result_data,
            pd.DataFrame.from_records(
                [
                    {
                        "upia": "AAA",
                        "upib": "TTT",
                        "umi": "GGG",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                    {
                        "upia": "TTT",
                        "upib": "AAA",
                        "umi": "GGG",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                    {
                        "upia": "GGG",
                        "upib": "CCC",
                        "umi": "AAA",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                ]
            ),
        )


def test_collapse_fastq_algorithm_adjacency():
    """Small test of adjacency based collapsing."""
    with mock.patch("pixelator.collapse.process.pyfastx") as mock_fastq_reader:

        def mock_reads(*args, **kwargs):
            for read in ["AAATTTGGG", "TTTAAAGGG", "GGGCCCAAA"]:
                yield None, read, None

        mock_fastq_reader.Fastq.return_value = mock_reads()

        result_file = collapse_fastq(
            input_file="/foo/bar",
            algorithm="adjacency",
            marker="CD4",
            sequence="AAAAAAAA",
            upia_start=0,
            upia_end=3,
            upib_start=3,
            upib_end=6,
            umia_start=6,
            umia_end=9,
            umib_start=None,
            umib_end=None,
            max_neighbours=10,
            mismatches=2,
        )
        result_data = pd.read_feather(result_file)
        assert_frame_equal(
            result_data,
            pd.DataFrame.from_records(
                [
                    {
                        "upia": "AAA",
                        "upib": "TTT",
                        "umi": "GGG",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                    {
                        "upia": "TTT",
                        "upib": "AAA",
                        "umi": "GGG",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                    {
                        "upia": "GGG",
                        "upib": "CCC",
                        "umi": "AAA",
                        "marker": "CD4",
                        "sequence": "AAAAAAAA",
                        "count": 1,
                        "umi_unique_count": 1,
                        "upi_unique_count": 1,
                    },
                ]
            ),
        )


@pytest.mark.integration_test
def test_collapse_fastq_algorithm_adjacency_simulated_reads():
    """Integration test with simulated reads, checking that reads are collapsed."""
    assay = config.assays["D21"]

    read_simulator = ReadSimulator(
        assay=assay, upia_pool_size=1000, upib_pool_size=1000, random_seed=1337
    )
    read_generator = partial(
        read_simulator.simulated_reads,
        n_molecules=10_000,
        mean_reads_per_molecule=10,
        std_reads_per_molecule=5,
        prob_of_seq_error=0.0001,
    )
    upia_start, upia_end = get_position_in_parent(assay, "upi-a")
    upib_start, upib_end = get_position_in_parent(assay, "upi-b")
    umib_start, umib_end = get_position_in_parent(assay, "umi-b")

    with mock.patch("pixelator.collapse.process.pyfastx") as mock_fastq_reader:

        def mock_reads(*args, **kwargs):
            for read in read_generator():
                yield None, read, None

        mock_fastq_reader.Fastq.return_value = mock_reads()

        result_file = collapse_fastq(
            input_file="/foo/bar",
            algorithm="adjacency",
            marker="CD4",
            sequence="AAAAAAAA",
            upia_start=upia_start,
            upia_end=upia_end,
            upib_start=upib_start,
            upib_end=upib_end,
            umia_start=None,
            umia_end=None,
            umib_start=umib_start,
            umib_end=umib_end,
            max_neighbours=10,
            mismatches=1,
        )
        data = pd.read_feather(result_file)

        assert data["upia"].nunique() == 1001
        assert data["upib"].nunique() == 1003
        assert data["umi"].nunique() == 9580
        assert data["count"].describe()["mean"] == 9.857187370170337
        assert data["umi_unique_count"].describe()["mean"] == 1.0378063980058163
        assert data["upi_unique_count"].describe()["mean"] == 1.0250311591192356


def test_build_annoytree():
    """Test building the annoy tree."""
    data = np.random.choice([0, 1], size=(10, 100))
    result = build_annoytree(data)
    # Best hit is self
    assert result.get_nns_by_item(0, n=10, search_k=-1)[0] == 0
