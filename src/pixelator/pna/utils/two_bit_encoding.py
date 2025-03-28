"""Two-bit encoding of DNA sequences.

This module provides functions to encode/decode DNA sequences using two bits per nucleotide.

Copyright © 2024 Pixelgen Technologies AB.
"""

import numpy as np

_comp = str.maketrans("ACGT", "TGCA")

_STR2INT: dict[int, np.uint64] = {
    ord("A"): np.uint64(0),
    ord("C"): np.uint64(1),
    ord("G"): np.uint64(2),
    ord("T"): np.uint64(3),
}
_int2str: bytes = b"ACGT"


_STR2INT_4: dict[int, np.uint64] = {
    ord("A"): np.uint64(0b0110),
    ord("C"): np.uint64(0b0101),
    ord("T"): np.uint64(0b0011),
    ord("G"): np.uint64(0b0000),
    ord("N"): np.uint64(0b1111),
}
_int2str_4: bytes = b"ACTG"


def _int2chars(value: np.uint64) -> bytes:
    res = np.zeros(4, dtype=np.uint8)

    res[0] = _int2str[value & 3]
    value >>= np.uint64(8)
    res[1] = _int2str[value & 3]
    value >>= np.uint64(8)
    res[2] = _int2str[value & 3]
    value >>= np.uint64(8)
    res[3] = _int2str[value & 3]

    return res.tobytes()


def pack_2bits(kmer: bytes) -> np.uint64:
    """Pack a kmer into an int using two bits per nucleotide.

    A kmer must be at most 32 nucleotides long.
    Any nucleotides beyond the 32nd are ignored.

    :param kmer: the kmer to pack
    :return: the packed kmer as an integer
    """
    # pack the kmer into an int
    assert len(kmer) <= 32

    value = np.uint64(0)
    for c in kmer[32::-1]:
        value <<= np.uint(2)
        value |= _STR2INT[c]

    return np.uint64(value)


def pack_4bits(kmer: bytes) -> np.uint64:
    """Pack a kmer into an int using four bits per nucleotide.

    A kmer must be at most 16 nucleotides long.
    Any nucleotides beyond the 16th are ignored.

    :param kmer: the kmer to pack
    :return: the packed kmer as an integer
    """
    assert len(kmer) <= 16

    value = np.uint64(0)
    for c in kmer[:16]:
        value <<= np.uint(4)
        value |= _STR2INT_4[c]
    return np.uint64(value)


def unpack_2bits(packed: int, k: int) -> bytes:
    """Unpack a kmer from an integer using two bits per nucleotides.

    :param packed: the packed kmer as an integer
    :param k: the length of the kmer
    """
    np_packed = np.uint64(packed)
    seq = bytearray(k)
    for i in range(k):
        seq[i] = _int2str[np_packed & np.uint64(3)]
        np_packed >>= np.uint64(2)
    return bytes(seq)


def unpack_4bits(packed: int, k: np.uint64) -> bytes:
    """Unpack a kmer from an integer using two bits per nucleotides.

    :param packed: the packed kmer as an integer
    :param k: the length of the kmer
    """
    seq = bytearray(k)
    for i in range(k):
        seq[i] = _int2str_4[packed & 15]
        packed >>= 4
    return bytes(seq)
