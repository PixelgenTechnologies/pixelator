"""Differences between two PNA antibody panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pandas as pd

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
