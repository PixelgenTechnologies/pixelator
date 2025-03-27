"""Model for report data returned by the single-cell analysis stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.mpx.report.models.base import SampleReport


class PolarizationReport(pydantic.BaseModel):
    """Model for report data returned by the polarization analysis."""

    pass


class ColocalizationReport(pydantic.BaseModel):
    """Model for report data returned by the colocalization analysis."""

    pass


class AnalysisSampleReport(SampleReport):
    """Model for report data returned by the single-cell analysis stage."""

    polarization: PolarizationReport | None
    colocalization: ColocalizationReport | None
