"""Work out how marker ids must be renamed so several pxl files line up.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Mapping

import duckdb

from pixelator.common.utils import logger
from pixelator.pna.config.panel import PNAAntibodyPanel, align_panels
from pixelator.pna.pixeldataset.io.pxl_file import PxlFile


def marker_renames_by_sample(
    pxl_files: Mapping[str, PxlFile],
) -> dict[str, dict[str, str]]:
    """Return the marker renames that make the panels of several files agree.

    Alignment is skipped, and no marker is renamed, when there is a single file,
    when the panel of a file is missing or cannot be read, or when the panels
    cannot be aligned. The last two cases are logged as warnings.

    Args:
        pxl_files: The files of a view, by sample name.

    Returns:
        For each sample that needs it, ``stored marker id -> aligned marker id``.
    """
    if len(pxl_files) < 2:
        return {}

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
            return {}
        if panel is None:
            logger.warning(
                "Markers are not aligned across files because %s does not store "
                "its antibody panel.",
                pxl_file.path,
            )
            return {}
        panels[sample_name] = panel

    try:
        alignment = align_panels(list(panels.values()))
    except ValueError as err:
        logger.warning("Markers are not aligned across files: %s", err)
        return {}

    renames = {
        sample_name: alignment.marker_renames(panel)
        for sample_name, panel in panels.items()
    }
    return {sample_name: r for sample_name, r in renames.items() if r}
