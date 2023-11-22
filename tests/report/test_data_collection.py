"""Test functions related to data collection for the QC report.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import pandas as pd
import io

from pixelator.report.qcreport.collect import collect_reads_per_umi_frequency


def test_collect_reads_per_umi_frequency(setup_basic_pixel_dataset):
    """Test if collect_reads_per_umi_frequency returns the correct headers."""
    dataset, *_ = setup_basic_pixel_dataset
    csv_data = collect_reads_per_umi_frequency(dataset)

    stringbuf = io.StringIO(csv_data)
    data = pd.read_csv(stringbuf)

    assert list(data.columns) == ["reads_per_umi", "count", "frequency"]
