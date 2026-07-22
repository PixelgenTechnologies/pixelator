"""Pixelgen brand colors, gradients, and palettes shared across all plotting modules.

This module is the Python port of Pixelgen's internal ``themes_and_palettes.R``
helpers (originally written for ggplot2), adapted for use with matplotlib,
seaborn, and plotly.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from matplotlib.colors import LinearSegmentedColormap

#: Pixelgen accent colors. Each hue has 12 levels ordered from lightest (index
#: 0, level 1) to darkest (index 11, level 12).
PIXELGEN_ACCENT_COLORS: dict[str, list[str]] = {
    "purples": [
        "#F7F5FD",
        "#DED4F6",
        "#C6B6EE",
        "#AF98E4",
        "#987DD8",
        "#8263CC",
        "#7E54E4",
        "#6D39DA",
        "#5F29C9",
        "#3C2178",
        "#2D165D",
        "#1F0E41",
    ],
    "blues": [
        "#F3F6FD",
        "#CDDAF4",
        "#A8BEE9",
        "#86A3DD",
        "#6588CF",
        "#466EC0",
        "#2955AE",
        "#1E469B",
        "#143887",
        "#0C2B71",
        "#061F58",
        "#02143E",
    ],
    "cyans": [
        "#F2FBFA",
        "#CBEFE9",
        "#A6E0D7",
        "#82D0C4",
        "#60BFB0",
        "#3FAC9B",
        "#209785",
        "#168777",
        "#0F7567",
        "#086256",
        "#044D44",
        "#013630",
    ],
    "greens": [
        "#F4FBF3",
        "#D2ECD0",
        "#B0DCAD",
        "#90CA8C",
        "#71B96C",
        "#53A54D",
        "#369030",
        "#2A8025",
        "#206F1C",
        "#175C14",
        "#0F480D",
        "#093207",
    ],
    "pinks": [
        "#FDF7FB",
        "#FCF3F9",
        "#F9E9F5",
        "#F6D1EB",
        "#F0ADDB",
        "#E57AC0",
        "#D953B1",
        "#C63288",
        "#AB216D",
        "#8D205A",
        "#771E4D",
        "#46122E",
    ],
    "reds": [
        "#FDF5F6",
        "#FDF0F2",
        "#FBE2E3",
        "#F7CACD",
        "#F19DA7",
        "#E96978",
        "#DD4154",
        "#CB2539",
        "#A72030",
        "#9B1828",
        "#7E0E1C",
        "#4F0E16",
    ],
    "oranges": [
        "#FDFAF6",
        "#FDF6F0",
        "#FAEBDC",
        "#F5D6BA",
        "#EFC097",
        "#E69C6A",
        "#DD7C45",
        "#CE5E2E",
        "#AE4525",
        "#8D3823",
        "#6C2D1F",
        "#491F15",
    ],
    "yellows": [
        "#FEFDF2",
        "#FDFBEC",
        "#FCF6CD",
        "#FAEDA6",
        "#F7E188",
        "#EFC438",
        "#DDAA29",
        "#BE861F",
        "#955E1A",
        "#7B4C1C",
        "#693D1D",
        "#4E2E15",
    ],
    "greys": [
        "#F9F9F9",
        "#E7E7E7",
        "#D5D5D5",
        "#C2C2C2",
        "#B0B0B0",
        "#9D9D9D",
        "#8B8B8B",
        "#737373",
        "#5C5C5C",
        "#444444",
        "#2D2D2D",
        "#151515",
    ],
    "beiges": [
        "#FFF6EE",
        "#FCE9D6",
        "#F6DCC1",
        "#EDCFAF",
        "#E2C29F",
        "#D5B592",
        "#C4A788",
        "#AC9175",
        "#947B62",
        "#7B6550",
        "#624F3E",
        "#48392C",
    ],
    "standardblues": [
        "#F8F9FD",
        "#EEF1F8",
        "#E0E6EF",
        "#CBD5E5",
        "#B2C1D9",
        "#98ABCA",
        "#7C90B1",
        "#607291",
        "#465671",
        "#344157",
        "#242D3E",
        "#161D2B",
    ],
}

#: Number of shade levels defined for every hue in `PIXELGEN_ACCENT_COLORS`.
_N_ACCENT_LEVELS = 12

#: Named multi-color gradients (for `pixelgen_gradient`/`pixelgen_colormap`).
PIXELGEN_GRADIENTS: dict[str, list[str]] = {
    "BluesCherry": [
        "#1F395F",
        "#496389",
        "#728BB1",
        "#AABAD1",
        "#DFE5EE",
        "#FFFFFF",
        "#FFE0EA",
        "#E9AABF",
        "#CD6F8D",
        "#A23F5E",
        "#781534",
    ],
    "BluesGrayCherry": [
        "#1F395F",
        "#44628E",
        "#718BB2",
        "#A8B9D1",
        "#DBE1EA",
        "#F1EEE9",
        "#F0D7E0",
        "#E3A6B8",
        "#CB6E8B",
        "#A23F5E",
        "#781534",
    ],
    "GrayblueRose": [
        "#798AAC",
        "#93A1BD",
        "#C4CBDB",
        "#FFFFFF",
        "#E8BFCD",
        "#D190A4",
        "#C1728B",
    ],
    "Cherry": [
        "#F2F2F2",
        "#FFE0EA",
        "#E9AABF",
        "#CD6F8D",
        "#A23F5E",
        "#781534",
    ],
    "Blues": [
        "#F2F2F2",
        "#DFE5EE",
        "#AABAD1",
        "#728BB1",
        "#496389",
        "#1F395F",
    ],
    "Magenta": [
        "#F2F2F2",
        "#FDE0EF",
        "#F1B6DA",
        "#DE77AE",
        "#C51C7D",
        "#8E0152",
    ],
    "Cyan": [
        "#F2F2F2",
        "#C2E5E1",
        "#9FE5DD",
        "#7CD5D0",
        "#59C5C3",
        "#36B5B6",
    ],
    "NaturalBlue": [
        "#1F385A",
        "#25456F",
        "#234977",
        "#214D80",
        "#1F5188",
        "#1D5591",
        "#1C5A99",
        "#1C5F9E",
        "#1C65A3",
        "#1C6AA8",
        "#1C6FAD",
        "#1E74B1",
        "#2478B2",
        "#2A7CB3",
        "#3080B5",
        "#3684B6",
        "#3E8AB9",
        "#4792BF",
        "#519AC5",
        "#5AA2CB",
        "#63AAD1",
        "#69B0D5",
        "#6DB6D8",
        "#71BBDB",
        "#75C1DE",
        "#79C6E1",
        "#83C9E0",
        "#8FCCDF",
        "#9BCEDD",
        "#A6D1DB",
        "#B2D3DA",
        "#C1DBE0",
    ],
}

#: Named discrete palettes (for `pixelgen_palette`).
PIXELGEN_PALETTES: dict[str, list[str]] = {
    "Tint": ["#E0E6EF", "#BECCE0", "#D1C6BB", "#DAD6D7", "#C4C4C4"],
    "Pastel": [
        "#8197BD",
        "#637EA5",
        "#D887A0",
        "#C86584",
        "#D8BA98",
        "#E2A489",
        "#B4ADAF",
        "#978D89",
    ],
    "Semi-saturated": [
        "#4D988D",
        "#496389",
        "#1F395F",
        "#E05573",
        "#BF9871",
        "#918F8F",
    ],
    "Saturated": [
        "#1B9E8A",
        "#25C6F2",
        "#E24B7E",
        "#AA498D",
        "#FFC950",
        "#231F20",
    ],
    "Cells1": [
        "#E9CD98",
        "#E19DB0",
        "#526C92",
        "#BECCE0",
        "#9E9188",
        "#7E9EA3",
        "#DEBA95",
        "#C6C6C6",
        "#46616E",
        "#1F395F",
    ],
    "Cells2": [
        "#C89433",
        "#809EA2",
        "#85756C",
        "#48696E",
        "#BA9D80",
        "#556C92",
        "#EBCD97",
        "#21395E",
    ],
    "Cells3": [
        "#A3A9CC",
        "#8DBFB3",
        "#F2EBC0",
        "#F3B462",
        "#F06060",
        "#44D593",
        "#B99095",
        "#E5E6E6",
        "#5D4C52",
        "#07475A",
    ],
}

#: Pixelgen branded colors assigned to specific (immune) cell types, for use in
#: UMAPs, bar plots, or heatmaps that annotate cell type.
PIXELGEN_CELL_PALETTE: dict[str, str] = {
    "CD4 T": "#6D92D1",
    "CD4 Naive": "#B9CDED",
    "Naive CD4 T": "#B9CDED",
    "CD4 TSCM": "#92B0E0",
    "CD4 TCM": "#6D92D1",
    "CD4 TEM": "#4A73C0",
    "CD4 TEFF": "#224792",
    "CD4 TEMRA": "#1C3A76",
    "Treg": "#2955AE",
    "CD4 CTL": "#2955AE",
    "CD8 T": "#75C5B5",
    "CD8 Naive": "#D0EDE6",
    "Naive CD8 T": "#D0EDE6",
    "CD8 TSCM": "#A2DACE",
    "CD8 TCM": "#75C5B5",
    "CD8 TEM": "#4AAF9D",
    "CD8 TEFF": "#209785",
    "CD8 TEMRA": "#1B7E6F",
    "other T": "#1B7E9F",
    "Other T": "#1B7E9F",
    "MAIT": "#0D6D5E",
    "DPT": "#1B7E9F",
    "DNT": "#7B787F",
    "dnT": "#7B787F",
    "NK": "#A28EDB",
    "CD56dim NK": "#A28EDB",
    "CD56bright NK": "#866CCD",
    "NK_CD56bright": "#866CCD",
    "NKT": "#6C4ABD",
    "B": "#C7A989",
    "B naive": "#F0DAC4",
    "Naive B": "#F0DAC4",
    "B intermediate": "#F0DAC4",
    "Memory B": "#C7A989",
    "B memory": "#C7A989",
    "Plasma cells": "#DE9982",
    "Plasmablast": "#DE9982",
    "DC": "#DA94C1",
    "cDC1": "#DA94C1",
    "cDC2": "#BB5391",
    "pDC": "#9C4579",
    "Mono": "#F7DFA0",
    "Classical Mono": "#F7DFA0",
    "CD14 Mono": "#F7DFA0",
    "Intermediate Mono": "#E6BE4A",
    "CD16 Mono": "#BD9315",
    "Non-classical Mono": "#BD9315",
    "other": "#797979",
    "Neutrophils": "#CDCDCD",
    "Basophils": "#797979",
    "Platelet": "#DB6365",
    "Platelets": "#DB6365",
    "gdT": "#5A3E9E",
}

#: Ordering of hues as they appear as columns in the R `Pixelgen_accent_colors`
#: tibble. Used to reproduce the default discrete color ordering.
_ACCENT_HUE_ORDER = (
    "purples",
    "blues",
    "cyans",
    "greens",
    "pinks",
    "reds",
    "oranges",
    "yellows",
    "greys",
    "beiges",
    "standardblues",
)

#: Row (level) order used to maximize perceptual distinction between adjacent
#: discrete colors, ported from `color_discrete_pixelgen`'s default branch.
_DISCRETE_LEVEL_ORDER = (7, 6, 8, 5, 9, 4, 10, 3, 11, 2, 12, 1)

#: Column (hue) order (1-based positions into `_ACCENT_HUE_ORDER`), ported
#: from `color_discrete_pixelgen`'s default branch.
_DISCRETE_HUE_POSITION_ORDER = (1, 3, 5, 7, 2, 4, 6, 8, 9, 10, 11)


def _validate_positive_int(n: int, name: str = "n") -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"'{name}' must be an integer, got {type(n).__name__}.")
    if n < 1:
        raise ValueError(f"'{name}' must be >= 1, got {n}.")


def pixelgen_gradient(n: int, name: str) -> list[str]:
    """Get ``n`` colors interpolated along a Pixelgen branded gradient.

    Args:
        n: The number of colors to return.
        name: The name of the gradient. One of the keys of `PIXELGEN_GRADIENTS`
            (``"BluesCherry"``, ``"BluesGrayCherry"``, ``"GrayblueRose"``,
            ``"Cherry"``, ``"Blues"``, ``"Magenta"``, ``"Cyan"``, or
            ``"NaturalBlue"``).

    Returns:
        A list of ``n`` hex color strings interpolated along the gradient.

    Raises:
        ValueError: If ``name`` is not a known gradient, or ``n`` < 1.
        TypeError: If ``n`` is not an integer.
    """
    _validate_positive_int(n)
    if name not in PIXELGEN_GRADIENTS:
        raise ValueError(
            f"Invalid gradient name {name!r}. Options are: "
            f"{sorted(PIXELGEN_GRADIENTS)}."
        )
    cmap = LinearSegmentedColormap.from_list(name, PIXELGEN_GRADIENTS[name])
    positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [_to_hex(cmap(pos)) for pos in positions]


def pixelgen_palette(n: int, name: str) -> list[str]:
    """Get the first ``n`` colors of a Pixelgen branded discrete palette.

    Args:
        n: The number of colors to return.
        name: The name of the palette. One of the keys of `PIXELGEN_PALETTES`
            (``"Tint"``, ``"Pastel"``, ``"Semi-saturated"``, ``"Saturated"``,
            ``"Cells1"``, ``"Cells2"``, or ``"Cells3"``).

    Returns:
        A list of ``n`` hex color strings.

    Raises:
        ValueError: If ``name`` is not a known palette, if ``n`` exceeds the
            number of colors available in the palette, or if ``n`` < 1.
        TypeError: If ``n`` is not an integer.
    """
    _validate_positive_int(n)
    if name not in PIXELGEN_PALETTES:
        raise ValueError(
            f"Invalid palette name {name!r}. Options are: {sorted(PIXELGEN_PALETTES)}."
        )
    colors = PIXELGEN_PALETTES[name]
    if n > len(colors):
        raise ValueError(f"Palette {name!r} only has {len(colors)} colors.")
    return colors[:n]


def pixelgen_accent_colors(
    hue: str | Sequence[str] | None = None,
    level: int | Sequence[int] | None = None,
) -> dict[str, str]:
    """Get Pixelgen accent colors for the given hue(s) and/or level(s).

    If only ``hue`` is given, all 12 levels of that hue (or hues) are
    returned. If only ``level`` is given, all hues at that level (or levels)
    are returned. If both are given, the colors at that hue/level
    combination are returned.

    Args:
        hue: A hue name or list of hue names. Options are the keys of
            `PIXELGEN_ACCENT_COLORS` (``"purples"``, ``"blues"``, ``"cyans"``,
            ``"greens"``, ``"pinks"``, ``"reds"``, ``"oranges"``,
            ``"yellows"``, ``"greys"``, ``"beiges"``, and ``"standardblues"``).
            If None, all hues are used.
        level: A level (1-12) or list of levels. If None, all levels are used.

    Returns:
        A dict mapping ``"{hue}{level}"`` (e.g. ``"blues4"``) to a hex color
        string.

    Raises:
        ValueError: If both ``hue`` and ``level`` are None, if ``hue``
            contains an unknown hue name, or if ``level`` is out of the
            ``[1, 12]`` range.
    """
    if hue is None and level is None:
        raise ValueError("'hue' and 'level' cannot both be None.")

    hues = (
        [hue]
        if isinstance(hue, str)
        else (list(_ACCENT_HUE_ORDER) if hue is None else list(hue))
    )
    levels = (
        [level]
        if isinstance(level, int)
        else (list(range(1, _N_ACCENT_LEVELS + 1)) if level is None else list(level))
    )

    invalid_hues = [h for h in hues if h not in PIXELGEN_ACCENT_COLORS]
    if invalid_hues:
        raise ValueError(
            f"Invalid hue(s) {invalid_hues}. Options are: {sorted(PIXELGEN_ACCENT_COLORS)}."
        )
    invalid_levels = [lv for lv in levels if not (1 <= lv <= _N_ACCENT_LEVELS)]
    if invalid_levels:
        raise ValueError(
            f"Invalid level(s) {invalid_levels}. Levels must be between 1 and "
            f"{_N_ACCENT_LEVELS}."
        )

    return {
        f"{h}{lv}": PIXELGEN_ACCENT_COLORS[h][lv - 1] for h in hues for lv in levels
    }


def pixelgen_discrete_colors(
    hue: str | Sequence[str] | None = None,
    level: int | Sequence[int] | None = None,
    n: int | None = None,
    shuffle: bool = False,
    indices: Sequence[int] | None = None,
) -> list[str]:
    """Get a list of Pixelgen accent colors suitable for coloring discrete categories.

    When neither ``hue`` nor ``level`` is given, this returns colors picked
    from across the full accent color grid in an order chosen to maximize
    perceptual distinction between adjacent categories -- this is the
    recommended default discrete color cycle.

    Args:
        hue: Optional hue name or list of hue names to restrict the colors to.
            See `pixelgen_accent_colors`.
        level: Optional level or list of levels to restrict the colors to. See
            `pixelgen_accent_colors`.
        n: If given, only the first ``n`` colors are returned.
        shuffle: Whether to shuffle the resulting colors. Ignored if
            ``indices`` is given. Defaults to False.
        indices: Optional 0-based indices used to select and/or reorder the
            colors.

    Returns:
        A list of hex color strings.

    Raises:
        ValueError: If ``hue`` or ``level`` are invalid (see
            `pixelgen_accent_colors`), or if ``n`` < 1.
    """
    if hue is not None or level is not None:
        colors = list(pixelgen_accent_colors(hue, level).values())
        if indices is not None:
            colors = [colors[i] for i in indices]
        elif shuffle:
            colors = list(colors)
            random.shuffle(colors)
    else:
        hue_positions = list(_DISCRETE_HUE_POSITION_ORDER)
        if indices is not None:
            hue_positions = [hue_positions[i] for i in indices]
        elif shuffle:
            hue_positions = list(hue_positions)
            random.shuffle(hue_positions)

        colors = [
            PIXELGEN_ACCENT_COLORS[_ACCENT_HUE_ORDER[pos - 1]][lv - 1]
            for lv in _DISCRETE_LEVEL_ORDER
            for pos in hue_positions
        ]

    if n is not None:
        _validate_positive_int(n)
        colors = colors[:n]
    return colors


def create_discrete_palette(conditions: Sequence[str]) -> list[str]:
    """Create a discrete color palette sized to the number of unique conditions.

    Colors are selected from different accent color hues and levels to ensure
    good visual distinction between conditions. This mirrors the palette used
    for coloring samples/conditions in QC and comparison plots.

    Args:
        conditions: A non-empty sequence of condition labels (duplicates
            allowed; only the number of unique values matters).

    Returns:
        A list of hex color strings, repeated 10 times over so it is safe to
        index into for any reasonable number of conditions.

    Raises:
        ValueError: If ``conditions`` is empty.
    """
    if len(conditions) == 0:
        raise ValueError("'conditions' must not be empty.")

    n_distinct = len(set(conditions))

    if n_distinct <= 2:
        base = [
            PIXELGEN_ACCENT_COLORS["blues"][3],
            PIXELGEN_ACCENT_COLORS["beiges"][5],
        ]
    elif n_distinct <= 6:
        base = (
            list(pixelgen_accent_colors(hue="blues", level=[4, 6]).values())
            + list(pixelgen_accent_colors(hue="beiges", level=[6, 9]).values())
            + list(pixelgen_accent_colors(hue="pinks", level=[5, 6]).values())
        )
    else:
        base = (
            list(pixelgen_accent_colors(hue="blues", level=[4, 6]).values())
            + list(pixelgen_accent_colors(hue="beiges", level=[6, 9]).values())
            + list(pixelgen_accent_colors(hue="pinks", level=[5, 6]).values())
            + list(pixelgen_accent_colors(hue="cyans", level=[4, 6]).values())
            + list(pixelgen_accent_colors(hue="yellows", level=[5, 7]).values())
            + list(pixelgen_accent_colors(hue="purples", level=[4, 6]).values())
            + list(pixelgen_accent_colors(hue="greys", level=[4, 6]).values())
        )

    return base * 10


def pixelgen_colorscale(
    hue: str = "purples",
    direction: int = 1,
    min_level: int = 1,
    max_level: int = _N_ACCENT_LEVELS,
) -> list[str]:
    """Get a Pixelgen branded continuous color scale for a given hue.

    Args:
        hue: The accent color hue to build the scale from. See
            `pixelgen_accent_colors`.
        direction: ``1`` for light-to-dark (the default accent color
            ordering), ``-1`` for dark-to-light.
        min_level: The lightest accent color level (1-12) to include. Raise
            this above 1 to avoid the near-white end of a hue, e.g. for
            points plotted on a white background that would otherwise be
            hard to see.
        max_level: The darkest accent color level (1-12) to include.

    Returns:
        A list of hex color strings (``max_level - min_level + 1`` of them),
        ordered as a continuous color scale.

    Raises:
        ValueError: If ``hue`` is unknown, ``direction`` is not 1 or -1, or
            ``min_level``/``max_level`` are out of range or in the wrong order.
    """
    if hue not in PIXELGEN_ACCENT_COLORS:
        raise ValueError(
            f"Invalid hue {hue!r}. Options are: {sorted(PIXELGEN_ACCENT_COLORS)}."
        )
    if direction not in (1, -1):
        raise ValueError(f"'direction' must be 1 or -1, got {direction}.")
    if not (1 <= min_level <= _N_ACCENT_LEVELS) or not (
        1 <= max_level <= _N_ACCENT_LEVELS
    ):
        raise ValueError(
            f"'min_level' and 'max_level' must be between 1 and {_N_ACCENT_LEVELS}."
        )
    if min_level > max_level:
        raise ValueError(
            f"'min_level' ({min_level}) must not be greater than 'max_level' "
            f"({max_level})."
        )

    colors = PIXELGEN_ACCENT_COLORS[hue][min_level - 1 : max_level]
    if direction == -1:
        colors = colors[::-1]
    return colors


def pixelgen_sequential_colormap(
    hue: str = "purples",
    direction: int = 1,
    min_level: int = 1,
    max_level: int = _N_ACCENT_LEVELS,
    n: int = 256,
) -> LinearSegmentedColormap:
    """Get a Pixelgen branded sequential matplotlib colormap for a given hue.

    Args:
        hue: The accent color hue to build the colormap from. See
            `pixelgen_accent_colors`.
        direction: ``1`` for light-to-dark, ``-1`` for dark-to-light.
        min_level: The lightest accent color level (1-12) to include. Raise
            this above 1 to avoid the near-white end of a hue, e.g. for
            points plotted on a white background that would otherwise be
            hard to see.
        max_level: The darkest accent color level (1-12) to include.
        n: The number of discrete color levels to sample the colormap at.

    Returns:
        A `matplotlib.colors.LinearSegmentedColormap` usable as a ``cmap=``
        argument.

    Raises:
        ValueError: If ``hue`` is unknown, ``direction`` is not 1 or -1, or
            ``min_level``/``max_level`` are out of range or in the wrong order.
    """
    colors = pixelgen_colorscale(
        hue=hue, direction=direction, min_level=min_level, max_level=max_level
    )
    return LinearSegmentedColormap.from_list(f"pixelgen_sequential_{hue}", colors, N=n)


def pixelgen_divergent_colormap(
    hue_low: str = "blues", hue_high: str = "reds", n: int = 256
) -> LinearSegmentedColormap:
    """Get a Pixelgen branded divergent matplotlib colormap between two hues.

    The colormap goes from the dark end of ``hue_low``, through white, to the
    dark end of ``hue_high``. This is useful for signed values such as
    correlations or differential scores.

    Args:
        hue_low: The accent color hue for low (negative) values. See
            `pixelgen_accent_colors`.
        hue_high: The accent color hue for high (positive) values. See
            `pixelgen_accent_colors`.
        n: The number of discrete color levels to sample the colormap at.

    Returns:
        A `matplotlib.colors.LinearSegmentedColormap` usable as a ``cmap=``
        argument.

    Raises:
        ValueError: If ``hue_low`` or ``hue_high`` is unknown.
    """
    low = pixelgen_colorscale(hue=hue_low, direction=-1)
    high = pixelgen_colorscale(hue=hue_high, direction=1)
    return LinearSegmentedColormap.from_list(
        f"pixelgen_divergent_{hue_low}_{hue_high}", [*low, "#FFFFFF", *high], N=n
    )


def pixelgen_colormap(name: str, n: int = 256) -> LinearSegmentedColormap:
    """Get one of the named Pixelgen branded gradients as a matplotlib colormap.

    Args:
        name: The name of the gradient. See `pixelgen_gradient`.
        n: The number of discrete color levels to sample the colormap at.

    Returns:
        A `matplotlib.colors.LinearSegmentedColormap` usable as a ``cmap=``
        argument.

    Raises:
        ValueError: If ``name`` is not a known gradient.
    """
    if name not in PIXELGEN_GRADIENTS:
        raise ValueError(
            f"Invalid gradient name {name!r}. Options are: "
            f"{sorted(PIXELGEN_GRADIENTS)}."
        )
    return LinearSegmentedColormap.from_list(name, PIXELGEN_GRADIENTS[name], N=n)


def _to_hex(rgba: tuple[float, float, float, float]) -> str:
    return "#" + "".join(f"{round(c * 255):02X}" for c in rgba[:3])
