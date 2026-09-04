"""NMF protein weights for two labeled cell populations.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import NMF

from pixelator.common.utils import logger

ProteinWeightsMode = Literal["cell_abundance", "k_neighborhood"]

_NMF_MAX_ITER = 5000
_MIN_MARKERS_REQUIRED = 5
_MIN_MARKERS_WARN = 10
_MIN_CELLS_WARN = 20
_PLOT_TOP_N = 40
_POPULATION_PALETTE = ("#6588CF", "#F19DA7")


def cc_protein_weights(
    adata: AnnData,
    group_by: str,
    population_1: str,
    population_2: str,
    masked_markers: Sequence[str] | None = None,
    max_freq: float = 0.01,
    min_diff: float = 0.5,
    mode: ProteinWeightsMode = "cell_abundance",
    max_components_per_population: int = 100,
    neighborhoods_per_component: int = 500,
    min_neighborhood_size: int = 10,
    k: int = 2,
    show_plot: bool = False,
    verbose: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """Derive protein weights for two cell populations.

    Uses non-negative matrix factorization (NMF) on protein abundance to find
    markers that distinguish two labeled populations, then returns a weight
    for each protein in each population. Those weights are the usual input
    when segmenting a cell:cell conjugate into its two cell types.

    Markers that look similar in both populations are dropped first. Highly
    abundant proteins such as HLA-ABC often get high weights in both
    populations, which makes them poor cell-type markers; ``max_freq``
    removes proteins that are common in both. ``min_diff`` keeps proteins
    whose relative frequency differs enough between the two populations.

    If spatial structure (for example platelet contamination) dominates the
    abundance difference, pass those proteins as ``masked_markers`` so they
    are left out of the model. CD41, CD36, CD62P, and CD9 are common choices.

    Set ``show_plot=True`` to inspect the top weights. A useful model has
    high weights on proteins that are mutually exclusive between the two
    cell types (for example CD20 on B cells and CD3e on T cells).

    Pass ``random_state`` if you want the same weights on repeated runs.
    NMF can otherwise vary slightly because cells may be downsampled and
    the factorization is not fully deterministic.

    Currently only whole-cell abundance (``mode="cell_abundance"``) is
    available. Neighborhood-based weights (``mode="k_neighborhood"``) are
    not implemented yet.

    Args:
        adata: AnnData with protein counts in ``.X`` and population labels
            in ``obs[group_by]``, for example from ``dataset.adata()``.
        group_by: Column in ``adata.obs`` with the cell-type labels.
        population_1: Name of the first population. This becomes the first
            column of the result.
        population_2: Name of the second population. This becomes the
            second column of the result.
        masked_markers: Optional proteins to exclude from the model.
        max_freq: Drop a protein if its frequency in *both* populations is
            at least this value. Default 0.01 (1%).
        min_diff: Minimum relative frequency difference between the two
            populations, ``|(freq1 - freq2) / (freq1 + freq2)|``. Default
            0.5.
        mode: How to build the abundance matrix. ``"cell_abundance"``
            (default) uses whole-cell protein counts.
            ``"k_neighborhood"`` is not available yet.
        max_components_per_population: Maximum number of cells to use from
            each population. Default 100.
        neighborhoods_per_component: Neighborhoods to sample per cell when
            ``mode="k_neighborhood"``. Default 500.
        min_neighborhood_size: Minimum neighborhood size when
            ``mode="k_neighborhood"``. Default 10.
        k: Neighborhood hop distance when ``mode="k_neighborhood"``.
            Default 2.
        show_plot: If True, plot the top protein weights for each
            population. Default False.
        verbose: If True, log progress and warn when a population is small
            or few markers remain. Default True.
        random_state: Seed used when sampling cells and fitting NMF, so
            results can be reproduced. Default 0.

    Returns:
        A DataFrame with one row per protein kept in the model and two
        columns named after the two populations. Values are non-negative
        NMF weights.

    Raises:
        ValueError: If the labels are missing, a population is not present,
            or fewer than 5 markers remain after filtering.
        NotImplementedError: If ``mode="k_neighborhood"``.

    Examples:
        After cell types are in ``adata.obs["cell_type"]``, fit weights for
        monocytes vs CD4 T cells and inspect the result::

            from pixelator.pna.analysis import cc_protein_weights
            from pixelator.pna.pixeldataset import read

            adata = read("sample.pxl").adata()
            w = cc_protein_weights(
                adata,
                group_by="cell_type",
                population_1="Mono",
                population_2="CD4T",
            )
            w.head()

    See Also:
        ``cc_protein_weights`` in pixelatorR, the equivalent function for
        R users.

    """
    del neighborhoods_per_component, min_neighborhood_size, k

    _validate_cc_protein_weights_params(
        adata=adata,
        group_by=group_by,
        population_1=population_1,
        population_2=population_2,
        masked_markers=masked_markers,
        max_freq=max_freq,
        min_diff=min_diff,
        mode=mode,
        max_components_per_population=max_components_per_population,
        show_plot=show_plot,
        verbose=verbose,
        random_state=random_state,
    )

    labels = adata.obs[group_by].astype(str)
    obs_keep = _components_to_keep(
        labels=labels,
        population_1=population_1,
        population_2=population_2,
        max_components_per_population=max_components_per_population,
        random_state=random_state,
        verbose=verbose,
    )
    subset = adata[obs_keep]
    counts = _protein_by_cell_counts(subset)
    group_vec = subset.obs[group_by].astype(str)
    pop1_mask = (group_vec == population_1).to_numpy()
    pop2_mask = (group_vec == population_2).to_numpy()

    pop1_props = _marker_proportions(counts, pop1_mask)
    pop2_props = _marker_proportions(counts, pop2_mask)
    markers_keep = _filter_markers(
        pop1_props=pop1_props,
        pop2_props=pop2_props,
        marker_names=counts.index,
        min_diff=min_diff,
        max_freq=max_freq,
        masked_markers=masked_markers,
    )
    _check_marker_count(int(markers_keep.sum()), verbose=verbose)

    filtered = counts.loc[markers_keep]
    weights_array, cell_loadings = _fit_rank2_nmf(
        filtered.to_numpy(dtype=float), random_state=random_state
    )
    if _population_1_component(cell_loadings, pop1_mask) == 1:
        weights_array = weights_array[:, ::-1]

    weights = pd.DataFrame(
        weights_array,
        index=filtered.index,
        columns=pd.Index([population_1, population_2], name="population"),
    )
    weights.index.name = counts.index.name or "marker"
    weights.attrs["model_type"] = "cell_abundance"
    weights.attrs["random_state"] = random_state

    if show_plot:
        _plot_protein_weights(weights)

    return weights


def _validate_cc_protein_weights_params(
    *,
    adata: AnnData,
    group_by: str,
    population_1: str,
    population_2: str,
    masked_markers: Sequence[str] | None,
    max_freq: float,
    min_diff: float,
    mode: str,
    max_components_per_population: int,
    show_plot: bool,
    verbose: bool,
    random_state: int,
) -> None:
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object.")
    if group_by not in adata.obs.columns:
        raise ValueError(f"'{group_by}' not found in adata.obs.")
    if population_1 == population_2:
        raise ValueError("population_1 and population_2 must be different.")
    if not (0.0 <= max_freq <= 1.0):
        raise ValueError("max_freq must be between 0 and 1.")
    if not (0.0 <= min_diff <= 1.0):
        raise ValueError("min_diff must be between 0 and 1.")
    if max_components_per_population < 1:
        raise ValueError("max_components_per_population must be at least 1.")
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be a bool.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool.")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an int.")

    labels = adata.obs[group_by].astype(str)
    present = set(labels.unique())
    for population in (population_1, population_2):
        if population not in present:
            raise ValueError(
                f"Population '{population}' not found in adata.obs['{group_by}']."
            )

    if masked_markers is not None:
        unknown = set(masked_markers) - set(adata.var_names)
        if unknown:
            raise ValueError(
                f"masked_markers not present in adata.var_names: {sorted(unknown)}."
            )

    if mode == "k_neighborhood":
        raise NotImplementedError(
            "mode='k_neighborhood' is not implemented. Use "
            "mode='cell_abundance' (the default, and the path used by "
            "segment_cell)."
        )
    if mode != "cell_abundance":
        raise ValueError(
            f"mode must be 'cell_abundance' or 'k_neighborhood', got {mode!r}."
        )


def _components_to_keep(
    labels: pd.Series,
    population_1: str,
    population_2: str,
    max_components_per_population: int,
    random_state: int,
    verbose: bool,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    kept: list[np.ndarray] = []
    for population in (population_1, population_2):
        names = labels.index[labels == population].to_numpy()
        n = names.size
        if verbose and n < _MIN_CELLS_WARN:
            logger.warning(
                "Population '%s' has fewer than %s cells (%s), which may "
                "lead to unstable results.",
                population,
                _MIN_CELLS_WARN,
                n,
            )
        if n > max_components_per_population:
            names = rng.choice(names, size=max_components_per_population, replace=False)
        kept.append(names)
    return np.concatenate(kept)


def _protein_by_cell_counts(adata: AnnData) -> pd.DataFrame:
    counts = adata.to_df().T
    if counts.empty:
        raise ValueError(
            "AnnData has no observations or variables to build protein weights from."
        )
    return counts


def _marker_proportions(counts: pd.DataFrame, cell_mask: np.ndarray) -> pd.Series:
    totals = counts.loc[:, cell_mask].sum(axis=1)
    denom = float(totals.sum())
    if denom == 0.0:
        return pd.Series(np.nan, index=counts.index)
    return totals / denom


def _filter_markers(
    pop1_props: pd.Series,
    pop2_props: pd.Series,
    marker_names: pd.Index,
    min_diff: float,
    max_freq: float,
    masked_markers: Sequence[str] | None,
) -> pd.Series:
    diff = (pop1_props - pop2_props) / (pop1_props + pop2_props)
    diff = diff.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = diff.abs() > min_diff
    keep &= np.minimum(pop1_props, pop2_props) < max_freq
    if masked_markers is not None:
        keep &= ~marker_names.isin(list(masked_markers))
    return keep


def _check_marker_count(n_markers: int, verbose: bool) -> None:
    if n_markers < _MIN_MARKERS_REQUIRED:
        raise ValueError(
            f"{n_markers} markers are kept after filtering with max_freq and "
            "min_diff. Consider relaxing the filtering criteria and check that "
            "the populations are correctly specified. Note that segmentation "
            "is not an option for similar cell types."
        )
    if verbose and n_markers < _MIN_MARKERS_WARN:
        logger.warning(
            "%s markers are kept after filtering with max_freq and min_diff. "
            "This is a low number of markers and may lead to unstable results.",
            n_markers,
        )


def _fit_rank2_nmf(
    counts: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit rank-2 NMF on a proteins × cells count matrix.

    sklearn NMF factorizes X ≈ W @ H with X shaped (n_samples, n_features).
    Passing proteins × cells matches R ``RcppML::nmf`` so W is proteins × 2
    (``w``) and H is 2 × cells.

    L1 regularization is left at sklearn defaults (none), matching R
    ``L1 = c(0, 0)``. The internal draft used ``alpha_W=0.1`` / ``l1_ratio=0.2``;
    those hyperparameters are intentionally not carried over.
    """
    model = NMF(
        n_components=2,
        init="nndsvda",
        solver="cd",
        random_state=random_state,
        max_iter=_NMF_MAX_ITER,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    protein_weights = model.fit_transform(counts)
    return protein_weights, model.components_


def _population_1_component(cell_loadings: np.ndarray, pop1_mask: np.ndarray) -> int:
    """Return the NMF component index (0 or 1) that matches population 1.

    R uses ``which.max`` of the row-sums of H on population-1 cells.
    """
    return int(np.argmax(cell_loadings[:, pop1_mask].sum(axis=1)))


def _plot_protein_weights(weights: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_data = (
        weights.rename_axis(index="marker")
        .reset_index()
        .melt(id_vars="marker", var_name="population", value_name="score")
    )
    top_markers = (
        plot_data.sort_values("score", ascending=False)
        .groupby("population", sort=False)
        .head(_PLOT_TOP_N)["marker"]
        .unique()
    )
    plot_subset = plot_data[plot_data["marker"].isin(top_markers)]
    marker_order = (
        plot_subset.groupby("marker")["score"].max().sort_values().index.tolist()
    )
    populations = list(weights.columns)
    palette = {
        populations[0]: _POPULATION_PALETTE[0],
        populations[1]: _POPULATION_PALETTE[1],
    }

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.25 * len(marker_order))))
    sns.barplot(
        data=plot_subset,
        y="marker",
        x="score",
        hue="population",
        order=marker_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title("NMF model weights")
    ax.set_xlabel("Weights")
    ax.set_ylabel("Protein")
    fig.tight_layout()
    plt.show()
