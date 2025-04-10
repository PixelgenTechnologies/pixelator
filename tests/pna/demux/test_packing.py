"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
from numpy.core.numeric import array_equal

from pixelator.pna.config import pna_config
from pixelator.pna.demux.barcode_demuxer import PNAEmbedding


def test_bitvector_roundtrip():
    umi1 = b"AAATAGTCTCCTCGGCAACAGGCCCCTT"
    umi2 = b"TACGGCCTACACCCCTCATTGACACTTT"
    uei = b"GATAAGATAGTGTGA"

    assay = pna_config.get_assay("pna-2")
    embedding = PNAEmbedding(assay)
    vector = embedding.encode(umi1, umi2, uei)

    assert array_equal(
        vector,
        np.array(
            [
                182,
                103,
                172,
                107,
                87,
                160,
                118,
                13,
                180,
                237,
                6,
                0,
                115,
                129,
                118,
                174,
                219,
                118,
                245,
                6,
                215,
                221,
                6,
                0,
                240,
                108,
                120,
                198,
                48,
                24,
                0,
                0,
            ],
            dtype=np.uint8,
        ),
    )

    umi1_rt, umi2_rt, uei_rt = embedding.decode(vector)

    assert umi1 == umi1_rt
    assert umi2 == umi2_rt
    assert uei == uei_rt
