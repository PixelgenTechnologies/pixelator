"""Test BK-tree classes.

Copyright © 2025 Pixelgen Technologies AB.
Copyright (c) Pixelgen Technologies AB
"""

import random

import numpy as np
import pytest

from pixelator.pna.config import pna_config
from pixelator.pna.demux.correction import BKTreeItem, build_bktree


def change_nucleotide(s: bytes) -> bytes:
    pos = random.randint(0, len(s) - 1)
    population = list(ord(i) for i in "ACGT")
    population.remove(s[pos])
    nuc = random.choices(population, k=1)[0]
    res = bytearray(s)
    res[pos] = nuc
    return bytes(res)


def helper_populations():
    res = dict()

    for i in list(b"ACGT"):
        tmp_pop = list(b"ACGT")
        tmp_pop.remove(i)
        res[i] = tmp_pop

    return res


_NUCLEOTIDES_POPULATION = helper_populations()


def change_2_nucleotides(s: bytes) -> bytes:
    positions = np.random.choice(range(0, len(s)), 2, replace=False)
    res = bytearray(s)

    for pos in positions:
        nuc = random.choices(_NUCLEOTIDES_POPULATION[s[pos]], k=1)[0]
        res[pos] = nuc

    return res


@pytest.mark.slow
def test_bktree_building():
    panel = pna_config.get_panel("proxiome-immuno-155")
    tree = build_bktree(panel, sequence_key="sequence_1")

    # Check that the tree is not empty
    assert tree is not None

    # Assert that we have twice the number of markers in the tree
    assert len(list(tree)) == panel.df.shape[0]

    # Assert that the matches identical sequences
    for index, row in panel.df.iterrows():
        seq = row["sequence_1"].encode("ascii")

        n1 = tree.find(BKTreeItem(row["marker_id"], seq), 1)
        best_score, best_item = n1[0]
        assert best_score == 0
        assert best_item.sequence == seq
        assert best_item.id == row["marker_id"]

    # Assert that one mismatch is always correctly found
    # Do this 1000 times to make sure since we are using random nucleotides
    for repeat in range(1000):
        for index, row in panel.df.iterrows():
            seq = row["sequence_1"].encode("ascii")
            one_mismatch_seq = change_nucleotide(seq)

            n1 = tree.find(BKTreeItem(row["marker_id"], one_mismatch_seq), 1)
            assert len(n1) > 0
            best_score, best_item = n1[0]
            assert best_score == 1
            assert best_item.sequence == seq
            assert best_item.id == row["marker_id"]
