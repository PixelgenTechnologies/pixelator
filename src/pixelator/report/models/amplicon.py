from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Type, TypeVar

from pixelator.report.models.base import StageReport
from pixelator.utils import get_sample_name


try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class AmpliconStageReport(StageReport):
    """Model for data returned by the amplicon stage."""

    #: The number of reads with Phred score ≥ 30
    fraction_q30: float

    #: The number of reads with Phred score ≥ 30 in the barcode region
    fraction_q30_bc: float

    #: The number of reads with Phred score ≥ 30 in the pixel binding site 1
    fraction_q30_pbs1: float

    #: The number of reads with Phred score ≥ 30 in the pixel binding site 2
    fraction_q30_pbs2: float

    #: The number of reads with Phred score ≥ 30 in the UMI region
    fraction_q30_umi: float

    #: The number of reads with Phred score ≥ 30 in the UPIA region
    fraction_q30_upia: float

    #: The number of reads with Phred score ≥ 30 in the UPIB region
    fraction_q30_upib: float

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`AmpliconStageReport` from a report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        return cls(sample_id=sample_name, **json_data["phred_result"])
