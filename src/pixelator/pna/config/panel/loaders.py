"""Load PNA panels from AnnData, pixel datasets, and the pixelator config.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from anndata import AnnData

from pixelator.common.config.panel import AntibodyPanelMetadata
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.base import PNAPanel
from pixelator.pna.config.panel.combination import PNAAntibodyPanelCombination
from pixelator.pna.config.panel.dispatch import (
    get_panel_type_from_metadata,
    panel_from_csv,
)
from pixelator.pna.config.panel.utils import _resolve_panel_source_from_pxl

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


def panel_from_adata(
    adata: AnnData,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> PNAPanel:
    """Create a panel (or combination) from AnnData panel metadata.

    If ``adata.uns`` contains ``num_partial_panels``, returns a
    :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`.
    Otherwise returns a single typed panel whose class follows the stored
    ``panel_type``.

    Args:
        adata: AnnData with panel metadata in ``uns`` (and marker columns in
            ``var`` for single-panel data).
        file_name: Optional basename of the source file.
        filepath: Optional full path of the source file.

    Returns:
        A :class:`~pixelator.pna.config.panel.base.PNAPanel` — typically a typed
        :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`
        subclass, or a combination when multiple partial panels are stored.
    """
    if "num_partial_panels" in adata.uns:
        return PNAAntibodyPanelCombination.from_adata(
            adata, file_name=file_name, filepath=filepath
        )
    panel_type = get_panel_type_from_metadata(
        AntibodyPanelMetadata.from_adata(adata)[0]
    )
    return panel_type.from_adata(adata, file_name=file_name, filepath=filepath)


def panel_from_pxl_dataset(
    pxl_data: PNAPixelDataset,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> PNAPanel:
    """Create a panel (or combination) from a PNA pixel dataset.

    Equivalent to
    :func:`~pixelator.pna.config.panel.loaders.panel_from_adata` on
    ``pxl_data.adata()``. When ``file_name`` / ``filepath`` are omitted and
    ``pxl_data`` wraps a single ``.pxl`` file, those values are taken from
    that file path.

    Args:
        pxl_data: Dataset that embeds panel information in its AnnData.
        file_name: Optional basename of the source file.
        filepath: Optional full path of the source file.

    Returns:
        A typed single panel or
        :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`.
    """
    file_name, filepath = _resolve_panel_source_from_pxl(
        pxl_data, file_name=file_name, filepath=filepath
    )
    return panel_from_adata(pxl_data.adata(), file_name=file_name, filepath=filepath)


def load_antibody_panel(
    config: PNAConfig, requested_panels: PathType | list[PathType] | list[str]
) -> PNAAntibodyPanelCombination:
    """Load one or more panels from config names and/or CSV paths.

    Each entry is resolved from ``config`` when possible, otherwise loaded with
    :func:`~pixelator.pna.config.panel.dispatch.panel_from_csv`. All resolved
    panels are returned as a
    :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`
    (including when only one panel is requested).

    Args:
        config: PNA configuration that may already contain named panels.
        requested_panels: Panel path(s) and/or config panel name(s).

    Returns:
        Combination containing the loaded panel(s).

    Raises:
        ValueError: If the resulting panel list is empty or panels conflict.
        AssertionError: If a CSV path cannot be loaded.
    """
    return_panels = []
    for panel in (
        requested_panels if isinstance(requested_panels, list) else [requested_panels]
    ):
        panel_str = str(panel)
        logger.debug("Loading panel %s", panel_str)
        panel_from_config = config.get_panel(panel_str)

        if panel_from_config is not None:
            logger.info("Found panel in config file: %s", panel_from_config.name)
            return_panels.append(panel_from_config)
            continue

        panel_obj = panel_from_csv(panel)
        logger.info(
            "Loaded %s %s from CSV file: %s",
            panel_obj.__class__.__name__,
            panel_obj.name,
            panel_obj.filename,
        )
        return_panels.append(panel_obj)
    return PNAAntibodyPanelCombination(return_panels)
