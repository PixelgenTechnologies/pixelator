"""Differences between two PNA antibody panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pandas as pd
from anndata import AnnData

from pixelator.pna.config.panel.antibody_panel import PNAAntibodyPanel

_SEQUENCE_COLUMNS = ["sequence_1", "sequence_2"]


def _by_sequences(panel: PNAAntibodyPanel) -> pd.DataFrame:
    """Return the panel table keyed by its sequence pair, which identifies a marker."""
    return panel.df.reset_index().set_index(_SEQUENCE_COLUMNS, drop=False)


def changed_marker_ids(old: PNAAntibodyPanel, new: PNAAntibodyPanel) -> dict[str, str]:
    """Return ``old marker_id -> new marker_id`` for markers whose id changed.

    Markers are matched on their sequences. Markers present in only one panel
    are ignored.

    Args:
        old: The panel before the change.
        new: The panel after the change.
    """
    both = _by_sequences(old)[["marker_id"]].join(
        _by_sequences(new)[["marker_id"]], how="inner", rsuffix="_new"
    )
    renamed = both[both["marker_id"] != both["marker_id_new"]]
    return dict(
        zip(renamed["marker_id"].astype(str), renamed["marker_id_new"].astype(str))
    )


def upgrade_adata(adata: AnnData, new_panel: PNAAntibodyPanel) -> AnnData:
    """Replace the panel columns of ``adata.var`` with those of ``new_panel``.

    Markers are matched on their sequences and keep their row order, so the
    count matrix still lines up. Columns of ``var`` that are not panel columns
    are kept.

    Args:
        adata: An AnnData object containing panel information.
        new_panel: The single-source panel to upgrade to. It must have the same
            markers as the panel stored in ``adata``.

    Raises:
        ValueError: If the panel stored in ``adata`` and ``new_panel`` do not
            have the same markers.
    """
    old_keys = _by_sequences(PNAAntibodyPanel.from_adata(adata)).index
    new_rows = _by_sequences(new_panel)
    if len(old_keys) != len(new_rows) or not old_keys.isin(new_rows.index).all():
        raise ValueError("Cannot upgrade: the panels do not have the same markers.")

    panel_part = new_rows.loc[old_keys].set_index("marker_id")
    other_part = adata.var.drop(columns=adata.uns["panel_metadata"]["panel_columns"])
    other_part.index = panel_part.index
    adata.var = pd.concat([panel_part, other_part], axis=1)
    adata.uns["panel_metadata"] = {
        **new_panel.metadata.model_dump(),
        "panel_columns": list(new_panel.df.columns),
    }
    return adata
