"""Tests for the pna plot module.

Copyright © 2025 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest

from pixelator.pna.plot import molecule_rank_plot


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="./snapshots/test_molecule_rank_plot",
)
def test_molecule_rank_plot():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "n_umi": np.round(10 ** np.random.normal(4, 0.3, 500)).astype(int),
            "group": np.random.choice(["A", "B"], 500),
        }
    )
    plot, _ = molecule_rank_plot(data, group_by="group")
    return plot
