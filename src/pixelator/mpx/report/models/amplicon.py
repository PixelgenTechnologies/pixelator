"""Model for report data returned by the single-cell amplicon stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.mpx.report.models.base import SampleReport


class AmpliconSampleReport(SampleReport):
    """Model for data returned by the amplicon stage."""

    fraction_q30: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30.",
    )

    fraction_q30_bc: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the barcode region.",
    )

    fraction_q30_pbs1: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the pixel binding site 1.",
    )

    fraction_q30_pbs2: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the pixel binding site 2.",
    )

    fraction_q30_umi: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UMI region.",
    )

    fraction_q30_upia: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UPIA region.",
    )

    fraction_q30_upib: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UPIB region.",
    )
