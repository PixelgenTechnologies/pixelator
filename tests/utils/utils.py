"""
Test utility functions

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import copy
import random
from typing import List

import numpy as np
from umi_tools._dedup_umi import edit_distance


def add_mutations(
    sequences: List[str], n_sequences: int, n_mutations: int
) -> List[str]:
    """
    add mutations to a list of DNA sequences, n_mutations will be added
    to n_sequences in the list
    """
    new_list = copy.deepcopy(sequences)
    seqs = np.random.choice(new_list, size=n_sequences, replace=False)
    for seq in seqs:
        for _ in range(n_mutations):
            pos = np.random.choice(len(seq))
            list_seq = list(seq)
            list_seq[pos] = np.random.choice(_ERRORS_DICT[list_seq[pos]])
        new_list.append("".join(list_seq))
    return new_list


def dna_seqs(length: int, min_dist: int, n_sequences: int) -> List[str]:
    """
    create a list of n_sequences random DNA sequences of length (length)
    where a minimum hamming distance of min_dist is guarantee
    """
    seq_list: List[str] = []
    for _ in range(n_sequences):
        while True:
            new_seq = "".join(random.choice("CGTA") for _ in range(length))
            all_dist = True
            for seq in seq_list:
                dist = edit_distance(new_seq.encode("utf-8"), seq.encode("utf-8"))
                if dist <= min_dist:
                    all_dist = False
                    break
            if all_dist:
                seq_list.append(new_seq)
                break
    return seq_list


_ERRORS_DICT = {
    "A": ["C", "G", "T"],
    "C": ["A", "G", "T"],
    "G": ["C", "A", "T"],
    "T": ["C", "G", "A"],
}
