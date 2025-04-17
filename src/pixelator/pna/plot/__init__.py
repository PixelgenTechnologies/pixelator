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
    """Plot the number of molecules (n_umi) per component against its n_umi rank.

    :param data: a pandas DataFrame with a column 'n_umi' containing antibody
    counts in components.
    :param group_by: a column in the DataFrame to group the plot by.

    :return: a plot showing the number of molecules per component against its
    molecule rank used for quality control.
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: AssertionError if the required column(s) are not present in the DataFrame
    :raises: ValueError if the data types are invalid
    """
    assert "n_umi" in data.columns, "column 'n_umi' is missing from DataFrame"

    if group_by is not None:
        assert group_by in data.columns, (
            f"group variable '{group_by}' not found in DataFrame"
        )

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

    (
        sns.relplot(
            data=molecule_rank_df, x="rank", y="n_umi", hue=group_by, aspect=1.6
        )
        .set(xscale="log", yscale="log")
        .set_xlabels("Component rank (by number of molecules)")
        .set_ylabels("Number of molecules")
    )

    return plt.gcf(), plt.gca()
