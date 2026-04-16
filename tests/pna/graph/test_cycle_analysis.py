import tempfile
from pathlib import Path

import polars as pl

from pixelator.pna.graph.cycle_analysis import process_component


def test_process_component():
    with tempfile.TemporaryDirectory() as tmpdir:
        edgelist = pl.DataFrame(
            {
                "umi1": [1, 3, 3, 5, 5, 1, 7, 7, 9, 9, 11, 11, 13, 13],
                "umi2": [2, 2, 4, 4, 6, 6, 6, 8, 8, 6, 8, 12, 12, 8],
                "component": ["test_component"] * 14,
            }
        )
        r"""GRAPH STRUCTURE:
        #           12
        #          / \
        #         13 11
        #          \ /
        #           8
        #          / \
        #         9   7
        #          \ /
        #           6
        #          / \
        #         5   1
        #         |   |
        #         4   2
        #          \ /
        #           3
        This graph contains one 6-cycle (1-2-3-4-5-6) and two 4-cycles (6-7-8-9 and 8-11-12-13).
        """
        edgelist_path = Path(tmpdir) / "edgelist.parquet"
        edgelist.write_parquet(edgelist_path)

        n_removed_edges, edge_cycle_length_dist = process_component(
            comp_name="test_component",
            edgelist_path=edgelist_path,
            tmpdir=tmpdir,
        )

        edge_cycle_length_dist = edge_cycle_length_dist.set_index("cycle_length")[
            "n_edges"
        ]

    assert n_removed_edges == 0
    assert edge_cycle_length_dist.loc[6] == 6
    assert edge_cycle_length_dist.loc[4] == 8
