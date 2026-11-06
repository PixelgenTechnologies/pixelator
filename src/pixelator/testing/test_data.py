"""Generate the synthetic PNA FASTQ test data for pipeline tests.

Writes paired-end ``<name>_R1/_R2.fastq.gz`` libraries and their single-end
``<name>_single_end_forward.fastq.gz`` and
``<name>_single_end_reverse.fastq.gz`` counterparts, each with a
``.metadata.json`` file recording its settings. A single-end library is read
from the same edge list as its paired-end library, so both hold the same cells,
but its reads are drawn with a seed of its own. The forward and reverse files
share that seed, so each forward read is the reverse complement of the
matching reverse read.

Usage::

    python -m pixelator.testing.test_data --output-dir <dir>

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pixelator.pna.config import pna_config
from pixelator.pna.config.assay import PNAAssay
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.testing.molecules import generate_edgelist
from pixelator.testing.reads import write_pna_fastq, write_single_end_fastq

# Settings shared by every library.
N_NODES = 200
N_EDGES = 600
MIN_NEIGHBORS = 20
HASHING_FRACTION = 0.2
N_CELL_TYPES = 3
CELL_TYPE_EFFECT = 10.0
N_SHARED_MARKERS = 3
QUALITY_STD = 0.0


@dataclass(frozen=True)
class EdgelistSpec:
    """Settings of one synthetic edge list."""

    panel: str
    assay: str
    seed: int
    n_cells: int
    n_crossing_edges: int
    hashing_indices: list[int] | None

    @property
    def n_edges_total(self) -> int:
        """Number of edges, crossing edges included."""
        return self.n_cells * N_EDGES + self.n_crossing_edges

    def load_panel(self) -> PNAAntibodyPanel:
        """Load the antibody panel from the pixelator configuration."""
        panel = pna_config.get_panel(self.panel)
        if panel is None:
            raise ValueError(f"Unknown panel {self.panel!r}.")
        return panel

    def load_assay(self) -> PNAAssay:
        """Load the assay from the pixelator configuration."""
        assay = pna_config.get_assay(self.assay)
        if assay is None:
            raise ValueError(f"Unknown assay {self.assay!r}.")
        return assay


@dataclass(frozen=True)
class Library:
    """One sequencing library of an edge list."""

    name: str
    edgelist: str
    fastq_seed: int
    single_end_seed: int


EDGELISTS = {
    "pool1": EdgelistSpec(
        "proxiome-v2-immuno-155-prerelease", "proxiome-v2", 1, 48, 200, [1, 5]
    ),
    "pool2": EdgelistSpec(
        "proxiome-v2-immuno-155-prerelease", "proxiome-v2", 5, 48, 200, [1, 5]
    ),
    "sample1": EdgelistSpec(
        "proxiome-v1-immuno-155-v1.1", "proxiome-v1", 3, 18, 100, None
    ),
    "sample2": EdgelistSpec(
        "proxiome-v1-immuno-155-v1.1", "proxiome-v1", 4, 18, 100, None
    ),
}

LIBRARIES = [
    Library("pool1", "pool1", fastq_seed=11, single_end_seed=111),
    Library("pool1_reseq", "pool1", fastq_seed=12, single_end_seed=112),
    Library("pool2", "pool2", fastq_seed=21, single_end_seed=121),
    Library("sample1", "sample1", fastq_seed=31, single_end_seed=131),
    Library("sample1_reseq", "sample1", fastq_seed=32, single_end_seed=132),
    Library("sample2", "sample2", fastq_seed=41, single_end_seed=141),
]


def build_edgelist(spec: EdgelistSpec) -> pl.DataFrame:
    """Generate the populated edge list described by ``spec``."""
    return generate_edgelist(
        spec.n_cells,
        N_NODES,
        N_EDGES,
        MIN_NEIGHBORS,
        spec.load_panel(),
        n_crossing_edges=spec.n_crossing_edges,
        hashing_fraction=HASHING_FRACTION,
        rng=spec.seed,
        hashing_indices=spec.hashing_indices,
        n_cell_types=N_CELL_TYPES,
        cell_type_effect=CELL_TYPE_EFFECT,
        n_shared_markers=N_SHARED_MARKERS,
    )


def _edgelist_metadata(spec: EdgelistSpec) -> dict:
    return {
        "n_edges_total": spec.n_edges_total,
        "n_cells": spec.n_cells,
        "n_nodes": N_NODES,
        "n_edges": N_EDGES,
        "min_neighbors": MIN_NEIGHBORS,
        "n_crossing_edges": spec.n_crossing_edges,
        "hashing_fraction": HASHING_FRACTION,
        "hashing_indices": spec.hashing_indices,
        "n_cell_types": N_CELL_TYPES,
        "cell_type_effect": CELL_TYPE_EFFECT,
        "n_shared_markers": N_SHARED_MARKERS,
    }


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def write_paired_library(
    library: Library, edgelist: pl.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    """Write the paired-end fastq files and metadata of ``library``."""
    spec = EDGELISTS[library.edgelist]
    n_reads = 2 * spec.n_edges_total
    paths = write_pna_fastq(
        library.name,
        n_reads,
        edgelist,
        spec.load_panel(),
        spec.load_assay(),
        output_dir=output_dir,
        rng=library.fastq_seed,
        quality_std=QUALITY_STD,
    )
    metadata = {
        "sample_name": library.name,
        "edgelist": library.edgelist,
        "panel": spec.panel,
        "assay": spec.assay,
        "edgelist_seed": spec.seed,
        "fastq_seed": library.fastq_seed,
        "n_reads": n_reads,
        "quality_std": QUALITY_STD,
        **_edgelist_metadata(spec),
    }
    _write_metadata(output_dir / f"{library.name}.metadata.json", metadata)
    return paths


def write_single_end_library(
    library: Library, edgelist: pl.DataFrame, output_dir: Path, reverse_read: bool
) -> Path:
    """Write a forward or reverse single-end fastq file and metadata of ``library``."""
    spec = EDGELISTS[library.edgelist]
    direction = "reverse" if reverse_read else "forward"
    name = f"{library.name}_single_end_{direction}"
    n_reads = 2 * spec.n_edges_total
    path = write_single_end_fastq(
        name,
        n_reads,
        edgelist,
        spec.load_panel(),
        spec.load_assay(),
        reverse_read=reverse_read,
        output_dir=output_dir,
        rng=library.single_end_seed,
        quality_std=QUALITY_STD,
    )
    metadata = {
        "sample_name": name,
        "edgelist": library.edgelist,
        "panel": spec.panel,
        "assay": spec.assay,
        "edgelist_seed": spec.seed,
        "fastq_seed": library.single_end_seed,
        "n_reads": n_reads,
        "quality_std": QUALITY_STD,
        "sequencing": "single_end",
        "reverse_read": reverse_read,
        **_edgelist_metadata(spec),
    }
    _write_metadata(output_dir / f"{name}.metadata.json", metadata)
    return path


def main(argv: list[str] | None = None) -> None:
    """Write the requested libraries to ``--output-dir``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequencing",
        nargs="+",
        choices=["paired", "single-end-forward", "single-end-reverse"],
        default=["paired", "single-end-forward", "single-end-reverse"],
        help="Which kinds of fastq files to write.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=[library.name for library in LIBRARIES],
        default=[library.name for library in LIBRARIES],
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edgelists: dict[str, pl.DataFrame] = {}
    for library in LIBRARIES:
        if library.name not in args.libraries:
            continue
        if library.edgelist not in edgelists:
            edgelists[library.edgelist] = build_edgelist(EDGELISTS[library.edgelist])
        edgelist = edgelists[library.edgelist]
        if "paired" in args.sequencing:
            write_paired_library(library, edgelist, args.output_dir)
        if "single-end-forward" in args.sequencing:
            write_single_end_library(
                library, edgelist, args.output_dir, reverse_read=False
            )
        if "single-end-reverse" in args.sequencing:
            write_single_end_library(
                library, edgelist, args.output_dir, reverse_read=True
            )


if __name__ == "__main__":
    main()
