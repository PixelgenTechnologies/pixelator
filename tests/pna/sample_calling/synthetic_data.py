"""Synthetic hashed pool used by the sample calling tests.

The pool has ``N_CELLS`` cells spread evenly over four hashing indices. Only
indices 1-3 are mapped to samples, so the cells with index 4 are undetermined
by construction.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from pixelator import __version__
from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset.io import PixelFileWriter
from tests.common.data_generator import generate_edgelist

POOL_NAME = "synthetic_pool"
SAMPLE_NAMES = ["PBMC", "Raji", "Jurkat"]
HASH_INDICES = [1, 2, 3, 4]
HASHING_BASES = ["B2M", "CD29"]
ALL_HASHING_ANTIBODIES = {
    f"{base}-{index}" for base in HASHING_BASES for index in HASH_INDICES
}

N_PER_INDEX = 4
N_CELLS = N_PER_INDEX * len(HASH_INDICES)
N_UNDETERMINED = N_PER_INDEX
ENRICHMENT_THRESHOLD = 10.0

_REGULAR_MARKERS = ["B2M", "CD29", "MarkerA", "MarkerB", "MarkerC", "MarkerD"]


def _dna(i: int, prefix: str) -> str:
    """Build a unique length-10 dna sequence from an integer."""
    suffix = "".join("ACGT"[(i >> (2 * k)) & 3] for k in range(3))
    return f"{prefix}{suffix}"


def build_hashing_panel() -> PNAAntibodyPanel:
    """Build a panel with regular markers and indexed hashing markers."""
    markers = [(m, "no") for m in _REGULAR_MARKERS]
    markers += [
        (f"{base}-{index}", "yes") for base in HASHING_BASES for index in HASH_INDICES
    ]
    panel_df = pd.DataFrame(
        {
            "marker_id": [marker for marker, _ in markers],
            "control": [False] * len(markers),
            "sequence_1": [_dna(i, "ACGTACG") for i in range(len(markers))],
            "sequence_2": [_dna(i, "TGCATGC") for i in range(len(markers))],
            "sample_hashing": [hashing for _, hashing in markers],
        }
    ).set_index("marker_id")
    return PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(name="test-hashing-panel", version="0.1.0"),
    )


def write_synthetic_hashed_pxl(directory: Path) -> Path:
    """Write the synthetic pool pxl file to ``directory``."""
    panel = build_hashing_panel()
    edgelist = generate_edgelist(
        n_cells=N_CELLS,
        n_nodes=300,
        n_edges=1200,
        min_neighbors=20,
        panel=panel,
        n_crossing_edges=0,
        hashing_fraction=0.8,
        rng=0,
    )
    edgelist = edgelist.with_columns(uei_count=pl.lit(1, dtype=pl.UInt32)).select(
        "umi1", "umi2", "read_count", "uei_count", "marker_1", "marker_2", "component"
    )

    path = directory / f"{POOL_NAME}.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_metadata(
            {
                "sample_name": POOL_NAME,
                "version": __version__,
                "technology": "single-cell-pna",
                "panel_name": panel.name,
                "panel_version": panel.version,
            }
        )
        writer.write_edgelist(edgelist)
        writer.write_adata(pna_edgelist_to_anndata(writer.get_connection(), panel))
    return path


def write_synthetic_samplesheet(directory: Path) -> Path:
    """Write a samplesheet mapping the first three hash indices to samples."""
    path = directory / "samplesheet.csv"
    rows = [
        f"{POOL_NAME},{sample},{index}"
        for sample, index in zip(SAMPLE_NAMES, HASH_INDICES)
    ]
    path.write_text("pool,sample,hash_index\n" + "\n".join(rows) + "\n")
    return path
