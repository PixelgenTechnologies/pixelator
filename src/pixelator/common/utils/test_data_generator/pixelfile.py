"""PNA pxl file generation from populated cell edge lists.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pixelator.pna.config.panel import PNAAntibodyPanel


def write_pna_pxl(
    edgelist: pl.DataFrame,
    panel: PNAAntibodyPanel,
    path: str | Path,
    sample_name: str = "synthetic",
) -> Path:
    """Write a populated edge list to a single-cell-pna pxl file.

    The edge list is expected in the :func:`generate_edgelist` schema (``umi1``,
    ``marker_1``, ``umi2``, ``marker_2``, ``component``, ``read_count``). Crossing
    edges (rows with a null ``component``) are dropped before the AnnData is
    built, so only genuine per-cell components are aggregated. The edge list is
    selected down to the pxl schema, written to the file, and an AnnData with
    aggregate metrics is built and stored alongside it.

    Args:
        edgelist: populated edge list from :func:`generate_edgelist`.
        panel: antibody panel providing the marker metadata.
        path: output path for the ``.pxl`` file.
        sample_name: sample name recorded in the pxl metadata.

    Returns:
        The path to the written pxl file.
    """
    from pixelator import __version__
    from pixelator.common.annotate.aggregates import call_aggregates
    from pixelator.pna.anndata import pna_edgelist_to_anndata
    from pixelator.pna.pixeldataset.io import PixelFileWriter

    path = Path(path)

    # Drop crossing edges (null component) and select the pxl edge list schema.
    pxl_edgelist = edgelist.filter(pl.col("component").is_not_null()).select(
        "umi1", "umi2", "marker_1", "marker_2", "component", "read_count"
    )

    with PixelFileWriter(path) as writer:
        writer.write_metadata(
            {
                "sample_name": sample_name,
                "version": __version__,
                "technology": "single-cell-pna",
                "panel_name": panel.name,
                "panel_version": panel.version,
            }
        )
        writer.write_edgelist(pxl_edgelist)
        adata = pna_edgelist_to_anndata(writer.get_connection(), panel)
        call_aggregates(adata)
        writer.write_adata(adata)

    return path
