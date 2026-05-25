"""Saving helpers for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pixelator.pna.pixeldataset.io import (
    InplacePixelDataFilterer,
    PxlFile,
    copy_databases,
)


class PixelDatasetSaver:
    """A class to save a PixelDataset to disk."""

    def __init__(self, pxl_dataset):
        """Create a new PixelDatasetSaver instance.

        Args:
        pxl_dataset: Pxl dataset.

        """
        self.pxl_dataset = pxl_dataset

    def save(
        self,
        sample_name: str,
        output_path: Path | str,
        optimize_disk_usage: bool = True,
    ) -> PxlFile:
        """Save a sample from a the PixelDataset to disk as a single pxl file with any component filters applied to it.

        NB: for the time being, no marker filters are applied to the saved file.

        This will copy the entire sample to a new file, applying any filters that have been set on the PxlFile
        on-disk.

        Args:
        sample_name: The name of the sample to save.
        output_path: The path to save the sample to.
        optimize_disk_usage: If True, the saved file will be optimized for disk usage. If this is active a temporary file will be written before the final file is written to disk.

        """
        try:
            input_sample = self.pxl_dataset.view.sample_to_file_mappings[sample_name]
        except KeyError:
            raise ValueError(
                f"Sample {sample_name} not found in the PixelDataset. Use one of: {self.pxl_dataset.sample_names()}"
            )

        if isinstance(output_path, str):
            output_path = Path(output_path)

        input_sample_pxl_file = PxlFile(input_sample)

        if optimize_disk_usage:
            with tempfile.NamedTemporaryFile() as temp_file:
                tmp_output_sample = PxlFile.copy_pxl_file(
                    input_sample_pxl_file, Path(temp_file.name)
                )
                InplacePixelDataFilterer(tmp_output_sample).filter_components(
                    self.pxl_dataset.components(),
                    metadata=input_sample_pxl_file.metadata(),
                )
                copy_databases(tmp_output_sample.path, output_path)
                return PxlFile(output_path, sample_name)

        output_sample = PxlFile.copy_pxl_file(input_sample_pxl_file, output_path)
        InplacePixelDataFilterer(output_sample).filter_components(
            self.pxl_dataset.components(),
            metadata=input_sample_pxl_file.metadata(),
        )
        return output_sample
