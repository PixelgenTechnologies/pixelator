"""PNA Pixel File io.

This module handles writing and reading duckdb based pxl files.

Duckdb is an embedded database that supports storing multiple
tables in one portable file.

Pxl files should contain an edgelist table that contains
the edgelist. Futhermore it should contain a number of tables
that encode the adata object. All these table should be prefixed
with `__adata__`, e.g. the adata.obs table should be named
`__adata__obs`.

In addition to the adata tables, and the edgelist, the database
can contain a metadata table, a layouts table, and a proximity
data table.

A full pxl file has (at least) the following tables:

.. code-block:: none

    ┌─────────────────────┐
    │ name                │
    │ ---                 │
    │ str                 │
    ╞═════════════════════╡
    │ Mandatory           │
    │ ---------           │
    │ __adata__X          │
    │ __adata__obs        │
    │ __adata__var        │
    │ edgelist            │
    ├─────────────────────┤
    │ Optional            │
    │ --------            │
    │ (layouts)           │
    │ (metadata)          │
    │ (proximity)         │
    └─────────────────────┘

## Architecture (IO types)

- `QueryBuilder` builds immutable `Query` values (SQL text + bound parameters).
- `PixelDataViewer` maps sample names to `PxlFile` instances; and gives a unified
  view to run SQL queries over the edgelist, layouts and proximity tables. Use
  ``with viewer.open() as session:`` and run `Query` instances via
  `PixelDataViewerSession` (lazy or eager Polars).
- `AnnDataHelper` holds a `PixelDataViewer` and uses `QueryBuilder` to join the
  annotated data from all samples into a single object.
- `PixelFileWriter` writes edgelist, AnnData-backed tables, and related data
  to a `.pxl` path.
- `InplacePixelDataFilterer` owns a `PxlFile` and trims components in place,
  using `AnnDataHelper`, `PixelDataViewer`, and `PixelFileWriter` when
  rewriting AnnData.

### Class diagram (ASCII)

.. code-block:: none

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ pixeldataset.io — types and responsibilities                             │
    └──────────────────────────────────────────────────────────────────────────┘

    Query  (dataclass: sql: str, params: dict)
         ▲
         │ builds
    ┌────┴────────────────────────────────────────────────────────────────────┐
    │ QueryBuilder                                                            │
    │   adata_X / obs / var / uns / obsm_* queries, edgelist_query → Query    │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐   maps sample → file   ┌──────────────────────────────┐
    │ PixelDataViewer     │ ──────────────────────▶│ PxlFile                      │
    │ open() → session    │                        │ path, sample_name, metadata… │
    │ sample_names, …     │                        └──────────────────────────────┘
    └──────────┬──────────┘
               │
               │ PixelDataViewerSession.execute_* (uses Query)
               ▼
            [ Query ]

    ┌─────────────────────┐
    │ AnnDataHelper       │──uses──▶ PixelDataViewer
    │ read_adata(...)     │
    └──────────┬──────────┘
               │ builds queries via
               ▼
         QueryBuilder

    ┌──────────────────────────────┐  owns   ┌────────────────────────────┐
    │ InplacePixelDataFilterer     │────────▶│ PxlFile                    │
    │ pxl_file, filter_components  │         └────────────────────────────┘
    └──────────────┬───────────────┘
                   │ uses when rewriting
                   ├──▶ AnnDataHelper
                   ├──▶ PixelDataViewer
                   └──▶ PixelFileWriter
                              │
                              │ path; write_edgelist, write_adata
                              ▼
                         (.pxl file)

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pixelator.common.duckdb_utils import connect_duckdb

from .inplace_pixel_data_filterer import InplacePixelDataFilterer
from .pixel_data_viewer import PixelDataViewer, PixelDataViewerSession
from .pixel_file_writer import PixelFileWriter
from .pxl_file import (
    PXL_FILE_ADATA_TABLES,
    PXL_FILE_MANDATOR_TABLES,
    PXL_FILE_OTHER_TABLES,
    PxlFile,
)
from .query_builder import Query, QueryBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "InplacePixelDataFilterer",
    "PixelDataViewer",
    "PixelDataViewerSession",
    "PixelFileWriter",
    "PxlFile",
    "Query",
    "QueryBuilder",
]


def copy_databases(src_db: Path, target_db: Path) -> None:
    """Copy the contents of one PXL file to another.

    This is a trick that can be used to reclaim disk-space. See:
    https://duckdb.org/docs/stable/operations_manual/footprint_of_duckdb/reclaiming_space.html

    Args:
        src_db: The source PXL file.
        target_db: The target PXL file.
    """
    query = f"""
    ATTACH '{str(src_db)}' AS src (READ_ONLY);
    ATTACH '{str(target_db)}' AS target;
    COPY FROM DATABASE src TO target;
    """

    with connect_duckdb() as connection:
        connection.execute(query)
