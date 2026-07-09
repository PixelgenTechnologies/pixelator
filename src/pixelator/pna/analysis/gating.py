"""Functions for gating components based on marker abundance (CLR) distributions.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from pixelator.common.utils import logger

_GATE_REGEX = re.compile(r"^(?P<sign>[+-])(?P<marker>.+)$")


@dataclass
class MarkerThreshold:
    """The result of determining a positive/negative threshold for a marker.

    Attributes:
        marker: The name of the marker.
        threshold: The CLR value used to call a component positive for the marker.
            ``None`` if the distribution was found to be unimodal and no
            threshold could reliably be determined.
        separation_score: A measure of how well separated the two components
            of the fitted Gaussian mixture model are. Higher values indicate
            a more clearly bimodal distribution.

    """

    marker: str
    threshold: float | None
    separation_score: float


def parse_gate(gate: list[str]) -> list[tuple[str, str]]:
    """Parse a gating specification into a list of (marker, sign) tuples.

    Args:
        gate: A list of gating strings, e.g. ``["+CD3e", "+CD4", "-CD19"]``. Each
            entry must start with ``+`` (positive) or ``-`` (negative) followed
            by a marker name.

    Returns:
        A list of ``(marker, sign)`` tuples, e.g. ``[("CD3e", "+"), ("CD4", "+"), ("CD19", "-")]``.

    Raises:
        ValueError: If any entry in ``gate`` is not prefixed with ``+`` or ``-``.

    """
    parsed = []
    for entry in gate:
        match = _GATE_REGEX.match(entry)
        if not match:
            raise ValueError(
                f"Invalid gate specification: '{entry}'. Expected a marker name "
                "prefixed with '+' (positive) or '-' (negative), e.g. '+CD3e'."
            )
        parsed.append((match.group("marker"), match.group("sign")))
    return parsed


def determine_marker_threshold(
    values: pd.Series | np.ndarray,
    min_separation_score: float = 3.0,
    marker: str = "",
) -> MarkerThreshold:
    """Determine a positive/negative threshold for a marker from its CLR distribution.

    A 2-component Gaussian mixture model is fit to the values, and the threshold
    is set to the midpoint between the two component means. The separation
    between the components is quantified as ``|mean_1 - mean_0| / (var_0 + var_1)``.
    If this separation score is below ``min_separation_score``, the distribution
    is considered unimodal, a warning is issued, and no threshold is returned.

    Note that fitting a 2-component mixture to a genuinely unimodal
    distribution still tends to produce a separation score around 1-3
    (rather than 0), since the two components end up splitting the single
    mode. The default cutoff of 3.0 was chosen empirically to reliably
    separate unimodal distributions from clearly bimodal ones (separated by
    several standard deviations).

    Args:
        values: The CLR values (across components) for a single marker.
        min_separation_score: The minimum separation score required for the
            distribution to be considered bimodal. Defaults to 3.0.
        marker: The name of the marker, used only for the warning message.

    Returns:
        A `MarkerThreshold` instance with the determined threshold (or ``None``
        if the distribution is unimodal) and the separation score.

    """
    values_arr = np.asarray(values, dtype=float).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, max_iter=1000, random_state=0)
    gmm.fit(values_arr)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]

    separation_score = float(np.abs(means[1] - means[0]) / np.sum(variances))

    if separation_score < min_separation_score:
        logger.warning(
            "Marker '%s' has a unimodal CLR distribution (separation score "
            "%.3f < %.3f). It will be ignored for gating.",
            marker,
            separation_score,
            min_separation_score,
        )
        return MarkerThreshold(
            marker=marker, threshold=None, separation_score=separation_score
        )

    threshold = float(np.mean(means))
    return MarkerThreshold(
        marker=marker, threshold=threshold, separation_score=separation_score
    )


def gate_mask(
    clr: pd.DataFrame,
    gate: list[str],
    min_separation_score: float = 3.0,
) -> tuple[pd.Series, list[MarkerThreshold]]:
    """Compute a boolean mask of components passing a gating specification.

    For each marker in the gate, a positive/negative threshold is determined
    from its CLR distribution in ``clr`` (see `determine_marker_threshold`).
    Markers whose distribution is unimodal are ignored (i.e. they do not
    constrain the mask).

    Args:
        clr: A DataFrame of CLR values with components as rows and markers as
            columns.
        gate: A list of gating strings, e.g. ``["+CD3e", "+CD4", "-CD19"]``.
        min_separation_score: The minimum separation score required for a
            marker's distribution to be considered bimodal. Defaults to 3.0.

    Returns:
        A tuple whose first element is a boolean ``pd.Series`` (indexed like
        ``clr``) that is ``True`` for components passing all (non-ignored)
        gating criteria, and whose second element is a list of
        ``MarkerThreshold`` instances, one per marker in ``gate``.

    Raises:
        KeyError: If a marker in ``gate`` is not a column of ``clr``.

    """
    parsed_gate = parse_gate(gate)

    mask = pd.Series(True, index=clr.index)
    thresholds = []
    for marker, sign in parsed_gate:
        if marker not in clr.columns:
            raise KeyError(f"Marker '{marker}' not found in the CLR data.")

        marker_threshold = determine_marker_threshold(
            clr[marker], min_separation_score=min_separation_score, marker=marker
        )
        thresholds.append(marker_threshold)

        if marker_threshold.threshold is None:
            continue

        if sign == "+":
            mask &= clr[marker] >= marker_threshold.threshold
        else:
            mask &= clr[marker] < marker_threshold.threshold

    return mask, thresholds
