"""Tests for the plot module.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest

from pixelator.mpx.plot import density_scatter_plot


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_density_scatter_plot/",
)
@pytest.mark.parametrize(
    "marker1, marker2, extra_params",
    [
        (
            "CD3",
            "CD8",
            {
                "facet_row": None,
                "facet_column": None,
                "gate": pd.Series(
                    [600, 10, 1000, 20], index=["xmin", "ymin", "xmax", "ymax"]
                ),
            },
        ),
        (
            "CD3",
            "CD8",
            {
                "facet_row": "mean_molecules_per_a_pixel",
                "facet_column": None,
                "gate": None,
            },
        ),
        (
            "CD3",
            "CD8",
            {
                "facet_row": None,
                "facet_column": "mean_molecules_per_a_pixel",
                "gate": pd.Series(
                    [600, 10, 1000, 20], index=["xmin", "ymin", "xmax", "ymax"]
                ),
            },
        ),
    ],
)
def test_density_scatter_plot(
    marker1, marker2, extra_params, density_scatter_plot_adata
):
    """Verify density scatter plot.

    Using the old fixture-derived AnnData.

    Args:
        marker1: marker1.
        marker2: marker2.
        extra_params: extra params.
        density_scatter_plot_adata: AnnData fixture for the plot input.
    """

    facet_row = extra_params["facet_row"]
    facet_column = extra_params["facet_column"]
    gate = extra_params["gate"]

    show_marginal = (facet_column is None) & (facet_row is None)
    fig, _ = density_scatter_plot(
        density_scatter_plot_adata,
        marker1=marker1,
        marker2=marker2,
        facet_row=facet_row,
        facet_column=facet_column,
        gate=gate,
        show_marginal=show_marginal,
    )
    return fig
