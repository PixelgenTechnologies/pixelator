"""Plugin for sample-calling report format.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pixelator.pna.report.models.base import SampleReport


class SampleCallingSampleReport(SampleReport):
    """Model for report data returned by the single-cell sample-calling stage."""

    report_type: str = "sample_calling"
    number_of_components: int
    number_of_incompatible_hashes_removed: int | None
