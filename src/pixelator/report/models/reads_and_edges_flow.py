from __future__ import annotations

import pydantic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ReadsAndMoleculesDataflowReport(pydantic.BaseModel):
    """Model for flow of input/output counts through processing stages."""

    sample_id: str

    #: The number of input reads from the input fastq files
    input_read_count: int

    #: The number of input reads after basic QC filtering
    qc_filtered_read_count: int

    #: The number of input reads after QC filtering and with valid PBS1/2 regions
    valid_pbs_read_count: int

    #: The number of input reads after QC, with valid PBS1/2 regions and with a valid
    #: antibody barcode
    valid_antibody_read_count: int

    #: The number of reads that are attributed to a unique molecule in the sample
    unique_molecule_read_count: int

    #: The number of unique molecules in the sample
    unique_molecule_count: int

    #: The number of reads that are counted as unique molecule in a cell component
    unique_molecule_in_cells_read_count: int

    #: The number of unique molecules that are assigned to a cell component
    unique_molecule_in_cells_count: int

    @pydantic.computed_field(return_type=float)
    @property
    def fraction_valid_pbs_reads(self):
        """Return the fraction of total raw input reads that has a valid PBS region."""
        return self.valid_pbs_read_count / self.input_read_count

    @pydantic.computed_field(return_type=float)
    @property
    def fraction_valid_umi_reads(self):
        """Return the fraction of total raw input reads that was attributed to a unique molecule."""
        return self.unique_molecule_read_count / self.input_read_count
