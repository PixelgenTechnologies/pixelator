"""Functions for creating plots that are useful with PNA data.

Copyright © 2025 Pixelgen Technologies AB.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def molecule_rank_plot(
    data: pd.DataFrame, group_by: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot molecule count (``n_umi``) per component against component rank (based on ``n_umi``).

    Each row of data represents one component. Components are ranked by
    descending ``n_umi`` (rank 1 corresponds to the highest count). When ``group_by`` is set,
    ranks are computed within each group and series are drawn in separate colors.

    Args:
        data: DataFrame with an ``n_umi`` column giving the molecule count per
            component.
        group_by: Optional column name used to rank within groups and color
            the plot.

    Returns:
        A representation of the log-log plot for number of molecules vs component rank, which can be used for quality control.

    Raises:
        AssertionError: If ``n_umi`` or ``group_by`` is missing from the columns in ``data``.
        ValueError: If the type of ``data[group_by]`` is invalid.
    """
    if "n_umi" not in data.columns:
        raise AssertionError("column 'n_umi' is missing from DataFrame")

    if group_by is not None:
        if group_by not in data.columns:
            raise AssertionError(f"group variable '{group_by}' not found in DataFrame")

        if data[group_by].dtype not in ["object", "category"]:
            raise ValueError(
                f"Invalid class '{data[group_by].dtype}' for column '{group_by}'. "
                f"Expected a string or categorical value"
            )
        else:
            molecule_rank_df = data[[group_by, "n_umi"]].copy()
            molecule_rank_df["rank"] = molecule_rank_df.groupby([group_by])[
                "n_umi"
            ].rank(ascending=False, method="first")
    else:
        molecule_rank_df = data[["n_umi"]].copy()
        molecule_rank_df["rank"] = molecule_rank_df["n_umi"].rank(
            ascending=False, method="first"
        )

    plot_grid = (
        sns.relplot(
            data=molecule_rank_df,
            x="rank",
            y="n_umi",
            hue=group_by,
            aspect=1.6,
        )
        .set(xscale="log", yscale="log")
        .set_xlabels("Component rank (by number of molecules)")
        .set_ylabels("Number of molecules")
    )

    return plot_grid.figure, plot_grid.ax
