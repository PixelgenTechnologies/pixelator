"""Differential abundance of markers across groups of components.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from pixelator.common.utils import logger
from pixelator.pna.utils.utils import normalize_input_to_list

# Names accepted from pixelatorR RunDAA / p.adjust, plus statsmodels aliases.
_P_ADJUST_METHOD_MAP = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "hochberg": "hochberg",
    "hommel": "hommel",
    "BH": "fdr_bh",
    "BY": "fdr_by",
    "fdr": "fdr_bh",
    "fdr_bh": "fdr_bh",
    "fdr_by": "fdr_by",
    "sidak": "sidak",
    "benjamini-hochberg": "fdr_bh",
}

PAdjustMethod = Literal[
    "bonferroni",
    "holm",
    "hochberg",
    "hommel",
    "BH",
    "BY",
    "fdr",
    "fdr_bh",
    "fdr_by",
    "sidak",
    "benjamini-hochberg",
]

_SCANPY_KEY = "_pixelator_rank_genes_groups"
_RESULT_COLUMNS = [
    "marker",
    "p",
    "p_adj",
    "difference",
    "pct_1",
    "pct_2",
    "target",
    "reference",
]


def differential_abundance(
    adata: AnnData,
    contrast_column: str,
    reference: str,
    targets: str | Sequence[str] | None = None,
    group_vars: str | Sequence[str] | None = None,
    features: str | Sequence[str] | None = None,
    layer: str | None = None,
    p_adjust_method: PAdjustMethod = "bonferroni",
) -> pd.DataFrame:
    """Compare marker abundance between a reference group and one or more targets.

    This is the Python analog of pixelatorR ``RunDAA``. For each ``target`` vs
    ``reference`` in ``contrast_column``, and optionally within each combination
    of ``group_vars`` (for example cell type), it calls
    ``scanpy.tl.rank_genes_groups`` with ``method="wilcoxon"``. It does not
    reimplement the Wilcoxon test.

    The result matches ``RunDAA``'s table more closely than raw scanpy
    output: effect size is a **mean difference** (``mean(target) -
    mean(reference)``), like Seurat ``FindMarkers`` with ``mean.fxn = rowMeans``
    and ``fc.name = "difference"``, not scanpy's log-fold change. P-values vs
    Seurat need not match.

    ``p_adj`` is computed once across every test this helper runs (all markers
    × targets × ``group_vars`` strata), replacing scanpy's per-call adjustment,
    the same as ``RunDAA``. A further **global FDR** across separately computed
    cell-type tables (PAT 05 notebook glue) is **not** applied here.

    Scanpy's Wilcoxon historically assumes non-negative values. Pixelator's
    default CLR (``clr_transformation(..., non_negative=True)``) is
    non-negative and is safe to pass via ``layer``. Signed CLR (negatives) does
    not crash the Wilcoxon call in current scanpy, but percent-expressed
    (values ``> 0``) is uninformative and log-fold changes would be invalid;
    this helper therefore reports mean difference on the original matrix.
    For signed CLR, pass counts in ``X``, a non-negative CLR layer (the
    pixelator default), or shift each marker so its minimum is 0.

    Args:
        adata: AnnData of components × markers. ``contrast_column`` and any
            ``group_vars`` must be columns of ``adata.obs``.
        contrast_column: ``obs`` column that defines the contrast, matching
            ``RunDAA(contrast_column=...)``. Passed to scanpy as ``groupby``.
        reference: Reference level of ``contrast_column``, matching
            ``RunDAA(reference=...)``. Passed to scanpy as ``reference``.
        targets: Target level(s) of ``contrast_column`` to compare against
            ``reference``, matching ``RunDAA(targets=...)``. If ``None``, every
            other level is used. Passed to scanpy as ``groups``.
        group_vars: Optional ``obs`` column(s) that split the data before each
            contrast, matching ``RunDAA(group_vars=...)`` (for example
            ``"cell_type"``). Each combination is tested separately and the
            grouping values are added as columns on the result.
        features: Optional marker names to test, matching Seurat
            ``FindMarkers(features=...)``. Default is all markers in
            ``adata.var_names``.
        layer: Matrix to test. ``None`` uses ``adata.X``. Otherwise the name is
            looked up in ``adata.layers``, then ``adata.obsm`` (PNA CLR is
            stored in ``adata.obsm["clr"]`` by default). Maps to
            ``RunDAA``'s assay/data slot and to scanpy ``layer``.
        p_adjust_method: Multiple-testing method applied to the collected raw
            p-values, matching ``RunDAA(p_adjust_method=...)``. RunDAA /
            ``p.adjust`` names (``bonferroni``, ``holm``, ``hochberg``,
            ``hommel``, ``BH``, ``BY``, ``fdr``) and statsmodels names
            (``fdr_bh``, ``fdr_by``, ``sidak``) are accepted. Defaults to
            ``"bonferroni"``, the ``RunDAA`` default.

    Returns:
        A DataFrame with one row per marker and contrast (and ``group_vars``
        stratum, if any). Columns:

        * ``marker`` — marker name
        * ``p`` — Wilcoxon p-value from ``scanpy.tl.rank_genes_groups``
        * ``p_adj`` — adjusted p-value (see above)
        * ``difference`` — mean(target) − mean(reference) on the selected matrix
        * ``pct_1`` — fraction of target cells with value ``> 0`` (scanpy
          ``pts`` / Seurat ``pct.1``)
        * ``pct_2`` — fraction of reference cells with value ``> 0`` (Seurat
          ``pct.2``)
        * ``target``, ``reference``
        * one column per ``group_vars`` entry, when given

    Raises:
        ValueError: If ``contrast_column`` / ``group_vars`` / ``layer`` /
            ``features`` / ``reference`` / ``targets`` / ``p_adjust_method``
            are invalid, or if no target-vs-reference comparison can be run.

    Examples:
        Compare stimulated vs resting CD8 T cells on a non-negative CLR
        layer::

            from pixelator.pna.analysis import differential_abundance

            da = differential_abundance(
                adata[adata.obs["cell_type"] == "CD8 T"],
                contrast_column="condition",
                reference="resting",
                targets="stimulated",
                layer="clr",
            )

    See Also:
        ``RunDAA`` in pixelatorR, which wraps Seurat ``FindMarkers``.
        ``scanpy.tl.rank_genes_groups`` (Wilcoxon) is the test used here.
    """
    group_var_list = normalize_input_to_list(group_vars) or []
    feature_list = normalize_input_to_list(features)
    adjust_method = _resolve_p_adjust_method(p_adjust_method)
    _validate_inputs(adata, contrast_column, reference, group_var_list, feature_list)
    target_list = _resolve_targets(adata, contrast_column, reference, targets)

    if group_var_list:
        logger.info("Splitting data by: %s", ", ".join(group_var_list))

    warned_negatives = False
    pieces: list[pd.DataFrame] = []
    for stratum_key, cells in _iter_group_strata(adata, group_var_list):
        for target in target_list:
            one, warned_negatives = _run_one_scanpy_wilcoxon(
                adata,
                cells,
                contrast_column=contrast_column,
                target=target,
                reference=reference,
                features=feature_list,
                layer=layer,
                warned_negatives=warned_negatives,
            )
            if one is None:
                continue
            for name, value in zip(group_var_list, stratum_key, strict=True):
                one[name] = value
            pieces.append(one)

    if not pieces:
        raise ValueError(
            "No target vs reference comparisons could be run. Check that "
            f"{contrast_column!r} contains both {reference!r} and the requested "
            "targets in every group_vars stratum."
        )

    result = pd.concat(pieces, ignore_index=True)
    result["p_adj"] = multipletests(result["p"].to_numpy(), method=adjust_method)[1]
    return result[[*_RESULT_COLUMNS, *group_var_list]]


def _validate_inputs(
    adata: AnnData,
    contrast_column: str,
    reference: str,
    group_vars: list[str],
    features: list[str] | None,
) -> None:
    """Raise if contrast, grouping, reference, or feature names are invalid."""
    if contrast_column not in adata.obs.columns:
        raise ValueError(
            f"contrast_column {contrast_column!r} is not a column of adata.obs."
        )
    missing_groups = [name for name in group_vars if name not in adata.obs.columns]
    if missing_groups:
        raise ValueError(
            f"group_vars must be columns of adata.obs. Missing: {missing_groups}."
        )
    if contrast_column in group_vars:
        raise ValueError(
            f"contrast_column {contrast_column!r} cannot also be one of group_vars."
        )
    if reference not in set(adata.obs[contrast_column].dropna()):
        raise ValueError(
            f"reference {reference!r} is not present in adata.obs[{contrast_column!r}]."
        )
    if features is not None:
        if len(features) == 0:
            raise ValueError("features must be a non-empty sequence of marker names.")
        missing_features = [name for name in features if name not in adata.var_names]
        if missing_features:
            raise ValueError(
                "features contains marker(s) that are not in adata.var_names: "
                f"{missing_features}."
            )


def _resolve_targets(
    adata: AnnData,
    contrast_column: str,
    reference: str,
    targets: str | Sequence[str] | None,
) -> list[str]:
    """Return target levels, defaulting to every non-reference contrast value."""
    if targets is None:
        levels = pd.unique(adata.obs[contrast_column].dropna())
        target_list = [level for level in levels if level != reference]
    else:
        target_list = normalize_input_to_list(targets) or []

    if not target_list:
        raise ValueError(
            "No targets to compare against the reference. Pass targets, or "
            f"ensure adata.obs[{contrast_column!r}] has levels besides {reference!r}."
        )
    if reference in target_list:
        raise ValueError(f"targets must not include the reference {reference!r}.")
    present = set(adata.obs[contrast_column].dropna())
    missing = [target for target in target_list if target not in present]
    if missing:
        raise ValueError(
            f"targets not found in adata.obs[{contrast_column!r}]: {missing}."
        )
    return target_list


def _resolve_p_adjust_method(p_adjust_method: str) -> str:
    """Map a RunDAA / scanpy p-adjust name to a statsmodels method string."""
    try:
        return _P_ADJUST_METHOD_MAP[p_adjust_method]
    except KeyError as exc:
        valid = ", ".join(sorted(_P_ADJUST_METHOD_MAP))
        raise ValueError(
            f"Unknown p_adjust_method {p_adjust_method!r}. Valid options: {valid}."
        ) from exc


def _iter_group_strata(
    adata: AnnData, group_vars: list[str]
) -> Iterable[tuple[tuple, pd.Index]]:
    """Yield ``(group key, cell index)`` for each ``group_vars`` combination."""
    if not group_vars:
        yield (), adata.obs_names
        return
    grouped = adata.obs.groupby(group_vars, observed=True, sort=False, dropna=False)
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        yield key, frame.index


def _run_one_scanpy_wilcoxon(
    adata: AnnData,
    cells: pd.Index,
    *,
    contrast_column: str,
    target: str,
    reference: str,
    features: list[str] | None,
    layer: str | None,
    warned_negatives: bool,
) -> tuple[pd.DataFrame | None, bool]:
    """Run scanpy Wilcoxon for one target vs reference and return a DataFrame."""
    in_stratum = np.asarray(adata.obs_names.isin(cells))
    contrast = adata.obs[contrast_column]
    is_target = contrast.eq(target).to_numpy()
    is_reference = contrast.eq(reference).to_numpy()
    n_target = int((in_stratum & is_target).sum())
    n_reference = int((in_stratum & is_reference).sum())
    if n_target == 0 or n_reference == 0:
        logger.warning(
            "Skipping contrast %s vs %s: need cells in both groups (found "
            "%d target, %d reference).",
            target,
            reference,
            n_target,
            n_reference,
        )
        return None, warned_negatives

    keep = in_stratum & (is_target | is_reference)
    marker_idx = features if features is not None else slice(None)
    work = adata[keep, marker_idx].copy()
    work, layer = _ensure_anndata_layer(work, layer)
    matrix = _values_matrix(work, layer)
    has_negatives = _has_negatives(matrix)
    if not warned_negatives and has_negatives:
        logger.warning(
            "Selected matrix contains negative values. scanpy Wilcoxon p-values "
            "are still computed, but percent-expressed (value > 0) is not "
            "meaningful on signed CLR. Mean difference is reported on the "
            "original values. Use counts, or non-negative CLR "
            "(clr_transformation(..., non_negative=True))."
        )
        warned_negatives = True

    # Scanpy still computes log-fold changes internally; they are unused here
    # (we report mean difference) and are NaN on signed CLR.
    with warnings.catch_warnings():
        if has_negatives:
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in log2",
                category=RuntimeWarning,
            )
        sc.tl.rank_genes_groups(
            work,
            groupby=contrast_column,
            groups=[target],
            reference=reference,
            method="wilcoxon",
            use_raw=False,
            layer=layer,
            n_genes=work.n_vars,
            key_added=_SCANPY_KEY,
        )
    ranked = sc.get.rank_genes_groups_df(work, group=target, key=_SCANPY_KEY)
    effects = _mean_difference_and_pct(work, contrast_column, target, reference, layer)
    result = pd.DataFrame(
        {
            "marker": ranked["names"].astype(str),
            "p": ranked["pvals"].to_numpy(),
        }
    )
    result = result.merge(effects, on="marker", how="left")
    result["target"] = target
    result["reference"] = reference
    return result, warned_negatives


def _ensure_anndata_layer(
    adata: AnnData, layer: str | None
) -> tuple[AnnData, str | None]:
    """Return ``(adata, layer)`` so the matrix is available on ``adata.layers``."""
    if layer is None:
        return adata, None
    if layer in adata.layers:
        return adata, layer
    if layer in adata.obsm:
        adata.layers[layer] = _obsm_as_array(adata, layer)
        return adata, layer
    raise ValueError(
        f"layer {layer!r} was not found in adata.layers or adata.obsm. "
        "PNA CLR is stored in adata.obsm['clr'] by default; pass layer='clr' "
        "or copy it to a layer."
    )


def _obsm_as_array(adata: AnnData, key: str) -> np.ndarray:
    """Align an ``obsm`` table to current observations and markers as a 2D array."""
    values = adata.obsm[key]
    if isinstance(values, pd.DataFrame):
        missing = [name for name in adata.var_names if name not in values.columns]
        if missing:
            raise ValueError(
                f"adata.obsm[{key!r}] is missing marker column(s): {missing}."
            )
        return values.loc[adata.obs_names, adata.var_names].to_numpy()
    array = np.asarray(values)
    expected = (adata.n_obs, adata.n_vars)
    if array.shape != expected:
        raise ValueError(
            f"adata.obsm[{key!r}] has shape {array.shape}, expected {expected}."
        )
    return array


def _values_matrix(adata: AnnData, layer: str | None):
    """Return ``X`` or the named layer used for the test and effect sizes."""
    if layer is None:
        return adata.X
    return adata.layers[layer]


def _has_negatives(matrix) -> bool:
    """Return True if any value in the matrix is negative."""
    minimum = matrix.min() if issparse(matrix) else np.nanmin(matrix)
    return bool(minimum < 0)


def _mean_difference_and_pct(
    adata: AnnData,
    contrast_column: str,
    target: str,
    reference: str,
    layer: str | None,
) -> pd.DataFrame:
    """Compute mean difference and percent-expressed on the tested matrix."""
    matrix = _values_matrix(adata, layer)
    target_mask = (adata.obs[contrast_column] == target).to_numpy()
    reference_mask = (adata.obs[contrast_column] == reference).to_numpy()
    return pd.DataFrame(
        {
            "marker": adata.var_names.astype(str),
            "difference": _column_means(matrix, target_mask)
            - _column_means(matrix, reference_mask),
            "pct_1": _pct_expressed(matrix, target_mask),
            "pct_2": _pct_expressed(matrix, reference_mask),
        }
    )


def _column_means(matrix, mask: np.ndarray) -> np.ndarray:
    """Return per-marker means for the cells selected by ``mask``."""
    return np.asarray(matrix[mask].mean(axis=0)).ravel()


def _pct_expressed(matrix, mask: np.ndarray) -> np.ndarray:
    """Return the fraction of selected cells with value ``> 0`` per marker."""
    subset = matrix[mask]
    dense = subset.toarray() if issparse(subset) else np.asarray(subset)
    return (dense > 0).mean(axis=0)
