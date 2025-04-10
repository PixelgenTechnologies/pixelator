"""Model for report data returned by the single-cell analysis stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.pna.report.models.base import SampleReport


class ProximityReport(pydantic.BaseModel):
    """Model for report data returned by the proximity analysis."""

    pass


class KCoreReport(pydantic.BaseModel):
    """Model for report data returned by the k-core analysis."""

    median_average_k_core: float


class SvdReport(pydantic.BaseModel):
    """Model for report data returned by the k-core analysis."""

    median_variance_explained_3d: float


class AnalysisSampleReport(SampleReport):
    """Model for report data returned by the single-cell analysis stage."""

    report_type: str = "analysis"

    proximity: ProximityReport | None
    k_cores: KCoreReport | None
    svd: SvdReport | None
