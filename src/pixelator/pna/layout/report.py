"""Model for report data returned by the single-cell layout stage.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import Literal

from pixelator.pna.report.models.base import SampleReport


class LayoutSampleReport(SampleReport):
    """Model for report data returned by the single-cell layout stage."""

    report_type: Literal["layout"] = "layout"
