"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
import pytest

from pixelator.pna.demux.barcode_demuxer import PNAEmbedding
from pixelator.pna.utils import pack_2bits, unpack_2bits


@pytest.fixture(scope="module")
def pna2_embedding():
    from pixelator.pna.config import pna_config

    assay = pna_config.get_assay("pna-2")
    embedding = PNAEmbedding(assay)
    return embedding


def test_embedding_decode_from_uin8_array(pna2_embedding):
    m = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00m\xe7\x01\xae\x81v\x80\xe7zp\r\x00C\xd1\x02@1\x14\x00\x00"

    b = np.frombuffer(m, dtype=np.uint8, count=len(m))
    umi1, umi2, uei = pna2_embedding.decode(b)

    assert umi1 == b"AACTGCCATCTTTGTACCCCACAGTAAC"
    assert umi2 == b"CCCTATGGACAGGCCTGGATACATGACA"
    assert uei == b"TGCGCCGGGGCGTGC"


def test_embedding_decode_from_bytes(pna2_embedding):
    m = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00m\xe7\x01\xae\x81v\x80\xe7zp\r\x00C\xd1\x02@1\x14\x00\x00"

    umi1, umi2, uei = pna2_embedding.decode(m)

    assert umi1 == b"AACTGCCATCTTTGTACCCCACAGTAAC"
    assert umi2 == b"CCCTATGGACAGGCCTGGATACATGACA"
    assert uei == b"TGCGCCGGGGCGTGC"


def test_embedding_encode(pna2_embedding):
    umi1 = b"AACTGCCATCTTTGTACCCCACAGTAAC"
    umi2 = b"CCCTATGGACAGGCCTGGATACATGACA"
    uei = b"TGCGCCGGGGCGTGC"

    expected = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00m\xe7\x01\xae\x81v\x80\xe7zp\r\x00C\xd1\x02@1\x14\x00\x00"

    res = pna2_embedding.encode(umi1, umi2, uei).tobytes()
    assert res == expected


def test_encode_umi(pna2_embedding):
    umi1 = b"AACTGCCATCTTTGTACCCCACAGTAAC"
    expected = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00\x00\x00\x00\x00"
    assert pna2_embedding.encode_umi(umi1) == expected


def test_decode_umi(pna2_embedding):
    umi = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00\x00\x00\x00\x00"
    expected = b"AACTGCCATCTTTGTACCCCACAGTAAC"
    assert pna2_embedding.decode_umi(umi) == expected


def test_3bit_to_2bit_recoding(pna2_embedding):
    umi = b"AACTGCCATCTTTGTACCCCACAGTAAC"
    umi_3bit_expected = b"v\x87\xd6\xeb6\xccm\xeb\x1a\xb3\x0b\x00\x00\x00\x00\x00"

    umi_3bit = pna2_embedding.encode_umi(umi)
    assert umi_3bit == umi_3bit_expected

    umi_2bit = pack_2bits(umi)
    umi_2bit_from_3bit = pna2_embedding.compress_umi_embedding(umi_3bit)

    assert umi_2bit == umi_2bit_from_3bit

    # Round trip back again
    seq_from_2bits = unpack_2bits(int(umi_2bit), 28)
    seq_from_2bits_from_3bits = unpack_2bits(int(umi_2bit_from_3bit), 28)

    assert seq_from_2bits == seq_from_2bits_from_3bits


@pytest.mark.parametrize(
    "umi,packed",
    [
        (b"AACTGCCATCTTTGTACCCCACAGTAAC", 19004325052749520),
    ],
)
def test_unpack_2bits(umi, packed):
    unpacked = unpack_2bits(packed, 28)
    assert umi == unpacked


@pytest.mark.parametrize(
    "umi,packed",
    [
        (b"AACTGCCATCTTTGTACCCCACAGTAAC", 19004325052749520),
    ],
)
def test_pack_2bits(umi, packed):
    packed_umi = pack_2bits(umi)
    assert packed_umi == packed
