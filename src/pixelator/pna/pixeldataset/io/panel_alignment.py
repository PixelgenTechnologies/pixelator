"""Work out how marker ids must be renamed so several pxl files line up.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import duckdb

from pixelator.common.utils import logger
from pixelator.pna.config.panel import (
    PanelConflictError,
    PNAAntibodyPanel,
    align_panels,
)
from pixelator.pna.pixeldataset.io.pxl_file import PxlFile


@dataclass(frozen=True)
class ViewAlignment:
    """How the markers of the files in a view are aligned.

    Attributes:
        renames: For each sample that needs it, ``stored marker id -> aligned
            marker id``.
        panel_rows: The aligned panel row of every marker, by its aligned marker
            id. Empty when the files were not aligned.
    """

    renames: dict[str, dict[str, str]] = field(default_factory=dict)
    panel_rows: dict[str, dict[str, Any]] = field(default_factory=dict)


def align_view_panels(
    pxl_files: Mapping[str, PxlFile], force: bool = False
) -> ViewAlignment:
    """Align the panels of the files of a view.

    Nothing is aligned when there is a single file, when the panel of a file is
    missing or cannot be read, or when the panels cannot be aligned. The last
    two cases are logged as warnings. Conflicting markers are not one of them:
    they raise, as the data of the files cannot be combined safely.

    Args:
        pxl_files: The files of a view, by sample name.
        force: Log conflicting markers as a warning instead of raising.

    Returns:
        The alignment, empty when nothing was aligned.

    Raises:
        PanelConflictError: When the panels have conflicting markers and
            ``force`` is not set.
    """
    if len(pxl_files) < 2:
        return ViewAlignment()

    panels: dict[str, PNAAntibodyPanel] = {}
    for sample_name, pxl_file in pxl_files.items():
        try:
            panel = pxl_file.read_panel()
        except (duckdb.Error, ValueError, KeyError, AssertionError) as err:
            # Aligning is best effort, and must not stop the view from opening.
            # These are the ways a stored panel can be unreadable: a damaged
            # file (duckdb.Error), malformed JSON or metadata (ValueError, which
            # covers the JSON, pydantic and version errors), missing panel
            # columns (KeyError), or a panel that fails validation
            # (AssertionError). Anything else is a bug and is let through.
            logger.warning(
                "Markers are not aligned across files because the antibody panel "
                "of %s could not be read: %s",
                pxl_file.path,
                err,
            )
            return ViewAlignment()
        if panel is None:
            logger.warning(
                "Markers are not aligned across files because %s does not store "
                "its antibody panel.",
                pxl_file.path,
            )
            return ViewAlignment()
        panels[sample_name] = panel

    try:
        alignment = align_panels(list(panels.values()), force=force)
    except PanelConflictError:
        raise
    except ValueError as err:
        logger.warning("Markers are not aligned across files: %s", err)
        return ViewAlignment()

    renames = {
        sample_name: alignment.marker_renames(panel)
        for sample_name, panel in panels.items()
    }
    return ViewAlignment(
        renames={sample_name: r for sample_name, r in renames.items() if r},
        panel_rows={row["marker_id"]: row for row in alignment.rows.values()},
    )
