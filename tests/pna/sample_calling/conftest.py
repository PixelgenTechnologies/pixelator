"""Shared fixtures for the sample calling tests.

Copyright © 2026 Pixelgen Technologies AB.
"""

import polars as pl
import pytest

from pixelator.pna.sample_calling.hash_antibodies import HashedAntibodyMapping
from tests.pna.sample_calling.synthetic_data import (
    ALL_HASHING_ANTIBODIES,
    POOL_NAME,
    write_synthetic_hashed_pxl,
    write_synthetic_samplesheet,
)


@pytest.fixture(name="synthetic_hashed_pxl", scope="module")
def synthetic_hashed_pxl_fixture(tmp_path_factory):
    """Path to a small synthetic hashed pool pxl file."""
    return write_synthetic_hashed_pxl(tmp_path_factory.mktemp("synthetic_pool"))


@pytest.fixture(name="synthetic_samplesheet", scope="module")
def synthetic_samplesheet_fixture(tmp_path_factory):
    """Path to a samplesheet mapping three samples of the synthetic pool."""
    return write_synthetic_samplesheet(tmp_path_factory.mktemp("synthetic_sheet"))


@pytest.fixture(name="synthetic_hashing_mapping")
def synthetic_hashing_mapping_fixture(synthetic_samplesheet):
    """Hashing antibody mapping for the synthetic pool."""
    return HashedAntibodyMapping.from_samplesheet(
        pl.read_csv(synthetic_samplesheet),
        all_hashing_antibodies=ALL_HASHING_ANTIBODIES,
        pool_name=POOL_NAME,
    )
