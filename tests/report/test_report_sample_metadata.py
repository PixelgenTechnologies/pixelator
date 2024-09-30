"""Tests for the SampleMetadata class.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest

from pixelator.report.models.report_metadata import SampleMetadata, SampleMetadataRecord


@pytest.fixture()
def setup_sample_metadata(tmp_path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "uropod_control",
                "description": "uropod control test sample",
                "panel_version": "1.0",
                "panel_name": "human-sc-immunology-spatial-proteomics",
            }
        ]
    )

    df.to_csv(tmp_path / "metadata.csv")
    return tmp_path / "metadata.csv"


def test_sample_metadata_lookup():
    metadata = SampleMetadata(
        [
            SampleMetadataRecord(
                sample_id="uropod_control",
                description="uropod control test sample",
                panel_version="1.0",
                panel_name="human-sc-immunology-spatial-proteomics",
            ),
            SampleMetadataRecord(
                sample_id="pmcs_unstimulated",
                description="uropod control test sample",
                panel_version="1.0",
                panel_name="human-sc-immunology-spatial-proteomics",
            ),
        ]
    )

    uropod_control_record = metadata.get_by_id("uropod_control")

    assert uropod_control_record.sample_id == "uropod_control"
    assert uropod_control_record.description == "uropod control test sample"
    assert uropod_control_record.panel_version == "1.0"
    assert uropod_control_record.panel_name == "human-sc-immunology-spatial-proteomics"

    pmcs_unstimulated_record = metadata.get_by_id("pmcs_unstimulated")

    assert pmcs_unstimulated_record.sample_id == "pmcs_unstimulated"
    assert pmcs_unstimulated_record.description == "uropod control test sample"
    assert pmcs_unstimulated_record.panel_version == "1.0"
    assert (
        pmcs_unstimulated_record.panel_name == "human-sc-immunology-spatial-proteomics"
    )

    assert metadata.get_by_id("blah") is None


def test_sample_metadata_from_csv(setup_sample_metadata):
    metadata = SampleMetadata.from_csv(setup_sample_metadata)
    uropod_control_record = metadata.get_by_id("uropod_control")

    assert uropod_control_record.sample_id == "uropod_control"
    assert uropod_control_record.description == "uropod control test sample"
    assert uropod_control_record.panel_version == "1.0"
    assert uropod_control_record.panel_name == "human-sc-immunology-spatial-proteomics"


def test_sample_metadata_duplicate_sample():
    with pytest.raises(ValueError, match="Every sample must have a unique id"):
        SampleMetadata(
            [
                SampleMetadataRecord(
                    sample_id="uropod_control",
                    description="uropod control test sample",
                    panel_version="1.0",
                    panel_name="human-sc-immunology-spatial-proteomics",
                ),
                SampleMetadataRecord(
                    sample_id="uropod_control",
                    description="uropod control test sample",
                    panel_version="1.0",
                    panel_name="human-sc-immunology-spatial-proteomics",
                ),
            ]
        )
