"""Shared fixtures for common tests.

Copyright © 2022 Pixelgen Technologies AB.
"""

from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).parent / "data"


@pytest.fixture(name="data_root", scope="session")
def data_root_fixture():
    """Return the data root directory."""
    return DATA_ROOT
