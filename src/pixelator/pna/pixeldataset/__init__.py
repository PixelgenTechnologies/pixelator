"""PNA Pixel Dataset.

This package defines the high-level
:class:`~pixelator.pna.pixeldataset.dataset.PNAPixelDataset` API and the
modality helpers (:class:`~pixelator.pna.pixeldataset.edgelist.Edgelist`,
:class:`~pixelator.pna.pixeldataset.proximity.Proximity`,
:class:`~pixelator.pna.pixeldataset.precomputed_layouts.PreComputedLayouts`)
used to read filtered views of data stored in ``.pxl`` DuckDB files.

Low-level access to those files (``PxlFile``, ``Query``, ``PixelDataViewer``, writers, and
:class:`~pixelator.pna.pixeldataset.io.anndata_helper.AnnDataHelper`) lives under
``pixelator.pna.pixeldataset.io``; see that package’s module documentation for the IO-layer
diagrams.

## Architecture (dataset layer)

:class:`~pixelator.pna.pixeldataset.dataset.PNAPixelDataset` is the façade: it holds a
:class:`~pixelator.pna.pixeldataset.io.pixel_data_viewer.PixelDataViewer` that maps sample
names to on-disk ``PxlFile`` instances, and it constructs an
:class:`~pixelator.pna.pixeldataset.io.anndata_helper.AnnDataHelper` that shares that viewer.
The helper applies active component and marker filters (plus ``PixelDatasetConfig`` join
strategy) when materializing :class:`anndata.AnnData`.

- ``read()`` / ``from_pxl_files()`` build a ``PixelDataViewer`` from paths or ``PxlFile``
  objects, then wrap it in ``PNAPixelDataset``.
- ``adata()`` delegates to ``AnnDataHelper.read_adata()`` (which opens the viewer’s DuckDB
  session and runs queries built via ``QueryBuilder`` in the IO package).
- ``edgelist()``, ``proximity()``, and ``precomputed_layouts()`` take the same ``view`` and
  reuse the dataset’s ``AnnDataHelper`` so AnnData-backed joins stay consistent with the
  active filters.
- ``metadata()`` queries DuckDB through ``view.open()`` and ``Query`` directly.
- ``filter()`` returns a new ``PNAPixelDataset`` with a narrowed ``PixelDataViewer`` (when
  filtering samples) and updated active components/markers, rebuilding ``AnnDataHelper``
  accordingly.

### Dependency overview (ASCII)

.. code-block:: none

    pixeldataset
    ┌────────────────────┐         ┌────────────────────┐
    │ PNAPixelDataset    ├─owns───▶│ PixelDataViewer    │
    └─────────┬──────────┘         └─────────┬──────────┘
              │                              │
              │ builds                       │ maps to PxlFile; open() → session
              ▼                              │
    ┌────────────────────┐                   │
    │ AnnDataHelper      ├─uses──────────────┘
    └────────────────────┘

    ┌────────────────────┐
    │ Edgelist           ├──┐
    │ Proximity          │  ├── view + shared AnnDataHelper (from PNAPixelDataset)
    │ PreComputedLayouts ├──┘
    └────────────────────┘

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path

from pixelator.pna.pixeldataset.config import PixelDatasetConfig
from pixelator.pna.pixeldataset.dataset import PNAPixelDataset
from pixelator.pna.pixeldataset.edgelist import Edgelist
from pixelator.pna.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pna.pixeldataset.proximity import Proximity
from pixelator.pna.pixeldataset.saver import PixelDatasetSaver
from pixelator.pna.pixeldataset.types import Component


def read(paths: Path | list[Path] | str | list[str]) -> PNAPixelDataset:
    """Read a PNAPixelDataset from one or more provided .pxl file(s).

    :param path: path to the file to read
    :return: an instance of `PNAPixelDataset`
    """
    if not paths:
        raise ValueError(
            "No paths provided to read function, did you pass an empty list?"
        )
    if not isinstance(paths, list):
        paths = [paths]  # type: ignore
    normalized_paths = [Path(p) for p in paths]  # type: ignore
    return PNAPixelDataset.from_pxl_files(normalized_paths)


__all__ = [
    "Component",
    "Edgelist",
    "PixelDatasetConfig",
    "PixelDatasetSaver",
    "PNAPixelDataset",
    "PreComputedLayouts",
    "Proximity",
]
