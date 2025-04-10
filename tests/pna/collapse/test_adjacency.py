"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
import pytest

from pixelator.pna.collapse.adjacency import (
    build_network_cluster,
    build_network_directional,
)
from pixelator.pna.demux.correction import hamming_distance
from pixelator.pna.utils.two_bit_encoding import pack_4bits


@pytest.fixture
def umitools_example_matrix():
    """A very small demonstration matrix from UMI-tools examples.

    See: https://umi-tools.readthedocs.io/en/latest/the_methods.html
    """

    nodes = [
        b"TCGT",
        b"CCGT",
        b"ACGT",
        b"ACAT",
        b"AAAT",
        b"ACAG",
    ]

    counts = np.array(
        [
            2,
            2,
            456,
            72,
            90,
            1,
        ],
        dtype=np.int64,
    )

    packed_nodes = [pack_4bits(node) for node in nodes]
    distances = np.zeros((6, 6), dtype=np.int64)

    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            v = hamming_distance(packed_nodes[i], packed_nodes[j])
            distances[i, j] = v
            distances[j, i] = v

    i, j = np.where(distances > 0)
    distances = distances[i, j].reshape(6, 5)
    indices = j.reshape(6, 5)

    return distances, indices, counts


def test_build_network_directional(umitools_example_matrix):
    """Test the build_adjacency_matrix_directional function.

      TCGT                  CCGT
       2                      2
       |                      |
       +----<----+------>-----+
                 |
               456 (ACGT)
                 |
     +           +------>----+
     |                       |
    90 (AAAT)                72 (ACAT)
                              ↓
                            1 (ACAG)
    """
    distances, indices, counts = umitools_example_matrix

    adj = build_network_directional(distances, indices, counts, 2)

    links = [
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 5),
    ]

    for i in range(0, 5):
        for j in range(0, 5):
            if (i, j) in links:
                assert adj[i, j] == 1
            else:
                assert adj[i, j] == 0


def test_build_network_cluster(umitools_example_matrix):
    """Test the build_adjacency_matrix_directional function.

      (TCGT)                  (CCGT)
       2  ------------------- 2
       |                      |
       +------+     +- -------+
              |     |
          456 (ACGT)
                 |
                 |
            72 (ACAT)
     +-----------+-----------+
     |                       |
    90 (AAAT)             1 (ACAG)



    """
    distances, indices, counts = umitools_example_matrix

    adj = build_network_cluster(distances, indices, counts, 2)

    links = {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (3, 5),
        (4, 3),
        (5, 3),
    }

    for i in range(0, 5):
        for j in range(0, 5):
            if (i, j) in links:
                assert adj[i, j] == 1
            else:
                assert adj[i, j] == 0
