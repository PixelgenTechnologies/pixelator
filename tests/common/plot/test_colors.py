"""Tests for the Pixelgen brand colors/palettes module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import re

import pytest
from matplotlib.colors import Colormap

from pixelator.common.plot.colors import (
    PIXELGEN_ACCENT_COLORS,
    PIXELGEN_CELL_PALETTE,
    PIXELGEN_GRADIENTS,
    PIXELGEN_PALETTES,
    create_discrete_palette,
    pixelgen_accent_colors,
    pixelgen_colormap,
    pixelgen_colorscale,
    pixelgen_discrete_colors,
    pixelgen_divergent_colormap,
    pixelgen_gradient,
    pixelgen_palette,
    pixelgen_sequential_colormap,
)

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _assert_all_hex(colors):
    for color in colors:
        assert HEX_COLOR_RE.match(color), f"{color!r} is not a valid hex color"


class TestPixelgenGradient:
    """Tests for `pixelgen_gradient`."""

    @pytest.mark.parametrize("name", list(PIXELGEN_GRADIENTS))
    def test_valid_names_return_n_colors(self, name):
        """Every documented gradient name should work and return valid hex colors."""
        colors = pixelgen_gradient(5, name)
        assert len(colors) == 5
        _assert_all_hex(colors)

    def test_n_equals_one_returns_first_color(self):
        """n=1 should return the start of the gradient, not error out."""
        colors = pixelgen_gradient(1, "Cherry")
        assert len(colors) == 1
        _assert_all_hex(colors)

    def test_endpoints_match_named_anchor_colors(self):
        """The first/last of n>=2 colors should be the gradient's defined endpoints."""
        anchors = PIXELGEN_GRADIENTS["Blues"]
        colors = pixelgen_gradient(10, "Blues")
        assert colors[0].upper() == anchors[0].upper()
        assert colors[-1].upper() == anchors[-1].upper()

    def test_invalid_name_raises(self):
        """An unknown gradient name should raise with the valid options listed."""
        with pytest.raises(ValueError, match="Invalid gradient name"):
            pixelgen_gradient(5, "NotAGradient")

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_n_raises(self, n):
        """n must be >= 1."""
        with pytest.raises(ValueError):
            pixelgen_gradient(n, "Cherry")

    def test_non_integer_n_raises(self):
        """n must be an integer, not e.g. a float."""
        with pytest.raises(TypeError):
            pixelgen_gradient(5.5, "Cherry")


class TestPixelgenPalette:
    """Tests for `pixelgen_palette`."""

    @pytest.mark.parametrize("name", list(PIXELGEN_PALETTES))
    def test_valid_names_return_subset(self, name):
        """Every documented palette name should work and return a color subset."""
        colors = pixelgen_palette(2, name)
        assert colors == PIXELGEN_PALETTES[name][:2]
        _assert_all_hex(colors)

    def test_n_equal_to_full_length_is_allowed(self):
        """Requesting exactly all colors in a palette should succeed."""
        n = len(PIXELGEN_PALETTES["Tint"])
        assert pixelgen_palette(n, "Tint") == PIXELGEN_PALETTES["Tint"]

    def test_n_greater_than_available_raises(self):
        """Requesting more colors than a palette has should raise, not silently wrap."""
        n = len(PIXELGEN_PALETTES["Tint"]) + 1
        with pytest.raises(ValueError, match="only has"):
            pixelgen_palette(n, "Tint")

    def test_invalid_name_raises(self):
        """An unknown palette name should raise with the valid options listed."""
        with pytest.raises(ValueError, match="Invalid palette name"):
            pixelgen_palette(2, "NotAPalette")

    def test_non_positive_n_raises(self):
        """n must be >= 1."""
        with pytest.raises(ValueError):
            pixelgen_palette(0, "Tint")


class TestPixelgenAccentColors:
    """Tests for `pixelgen_accent_colors`."""

    def test_both_none_raises(self):
        """At least one of hue/level must be given, mirroring the R stopifnot check."""
        with pytest.raises(ValueError, match="cannot both be None"):
            pixelgen_accent_colors()

    def test_hue_only_returns_all_levels(self):
        """Requesting only a hue should return all 12 levels for it."""
        colors = pixelgen_accent_colors(hue="blues")
        assert len(colors) == 12
        assert colors["blues1"] == PIXELGEN_ACCENT_COLORS["blues"][0]
        assert colors["blues12"] == PIXELGEN_ACCENT_COLORS["blues"][11]

    def test_level_only_returns_all_hues(self):
        """Requesting only a level should return that level for every hue."""
        colors = pixelgen_accent_colors(level=3)
        assert len(colors) == len(PIXELGEN_ACCENT_COLORS)
        assert colors["reds3"] == PIXELGEN_ACCENT_COLORS["reds"][2]

    def test_hue_and_level_combination(self):
        """Combining hue(s) and level(s) should return the cartesian product."""
        colors = pixelgen_accent_colors(hue=["blues", "reds"], level=[4, 6])
        assert list(colors.keys()) == ["blues4", "blues6", "reds4", "reds6"]
        assert colors["blues4"] == PIXELGEN_ACCENT_COLORS["blues"][3]
        assert colors["reds6"] == PIXELGEN_ACCENT_COLORS["reds"][5]

    def test_invalid_hue_raises(self):
        """An unknown hue name should raise with the valid options listed."""
        with pytest.raises(ValueError, match="Invalid hue"):
            pixelgen_accent_colors(hue="not-a-hue")

    @pytest.mark.parametrize("level", [0, 13, -1])
    def test_out_of_range_level_raises(self, level):
        """Levels must fall within the defined [1, 12] range."""
        with pytest.raises(ValueError, match="Invalid level"):
            pixelgen_accent_colors(level=level)


class TestPixelgenDiscreteColors:
    """Tests for `pixelgen_discrete_colors`."""

    def test_default_returns_full_grid(self):
        """With no arguments, every hue/level combination should be represented."""
        colors = pixelgen_discrete_colors()
        assert len(colors) == 12 * len(PIXELGEN_ACCENT_COLORS)
        _assert_all_hex(colors)
        # Every returned color should be a genuine accent color.
        all_accent_colors = {
            c for shades in PIXELGEN_ACCENT_COLORS.values() for c in shades
        }
        assert set(colors) <= all_accent_colors

    def test_n_truncates(self):
        """Passing n should truncate to that many colors, keeping the same order."""
        colors = pixelgen_discrete_colors(n=5)
        assert colors == pixelgen_discrete_colors()[:5]

    def test_hue_restricts_to_that_hue(self):
        """Passing a hue should restrict the discrete colors to that hue only."""
        colors = pixelgen_discrete_colors(hue="greens")
        assert set(colors) <= set(PIXELGEN_ACCENT_COLORS["greens"])

    def test_indices_selects_hue_subset(self):
        """Indices select which hues participate; all 12 levels are still returned."""
        colors = pixelgen_discrete_colors(indices=[0])
        assert len(colors) == 12
        assert set(colors) == set(PIXELGEN_ACCENT_COLORS["purples"])

    def test_indices_reorders_hue_order(self):
        """Reordering indices should change the resulting color order, not just its set."""
        a = pixelgen_discrete_colors(indices=[0, 1])
        b = pixelgen_discrete_colors(indices=[1, 0])
        assert a != b
        assert set(a) == set(b)

    def test_shuffle_is_a_permutation(self):
        """Shuffling should not add, drop, or duplicate colors."""
        full = pixelgen_discrete_colors()
        shuffled = pixelgen_discrete_colors(shuffle=True)
        assert sorted(shuffled) == sorted(full)


class TestCreateDiscretePalette:
    """Tests for `create_discrete_palette`."""

    def test_empty_conditions_raises(self):
        """An empty condition list is not a meaningful request."""
        with pytest.raises(ValueError, match="must not be empty"):
            create_discrete_palette([])

    @pytest.mark.parametrize(
        "conditions, expected_base_len",
        [
            (["a"], 2),
            (["a", "b"], 2),
            (["a", "a", "b", "b"], 2),
            (["a", "b", "c", "d", "e", "f"], 6),
            (["a", "b", "c", "d", "e", "f", "g"], 14),
        ],
    )
    def test_palette_size_scales_with_distinct_count(
        self, conditions, expected_base_len
    ):
        """The base palette size should follow the <=2 / <=6 / >6 tiers."""
        colors = create_discrete_palette(conditions)
        assert len(colors) == expected_base_len * 10
        _assert_all_hex(colors[:expected_base_len])

    def test_result_covers_all_conditions(self):
        """The returned palette must have at least as many colors as distinct conditions."""
        conditions = [f"condition_{i}" for i in range(20)]
        colors = create_discrete_palette(conditions)
        assert len(colors) >= len(set(conditions))


def test_pixelgen_cell_palette_is_valid():
    """Every entry in the cell type palette should be a valid hex color."""
    assert len(PIXELGEN_CELL_PALETTE) > 0
    _assert_all_hex(PIXELGEN_CELL_PALETTE.values())


class TestColorscaleAndColormaps:
    """Tests for the continuous colorscale/colormap helpers."""

    def test_colorscale_direction_1_is_light_to_dark(self):
        """Direction 1 should match the accent color hue's natural (light->dark) order."""
        assert (
            pixelgen_colorscale("purples", direction=1)
            == PIXELGEN_ACCENT_COLORS["purples"]
        )

    def test_colorscale_direction_minus_1_is_reversed(self):
        """Direction -1 should reverse the natural order."""
        assert (
            pixelgen_colorscale("purples", direction=-1)
            == PIXELGEN_ACCENT_COLORS["purples"][::-1]
        )

    def test_colorscale_invalid_hue_raises(self):
        """An unknown hue should raise."""
        with pytest.raises(ValueError, match="Invalid hue"):
            pixelgen_colorscale("not-a-hue")

    def test_colorscale_invalid_direction_raises(self):
        """Direction must be 1 or -1."""
        with pytest.raises(ValueError, match="direction"):
            pixelgen_colorscale("purples", direction=2)

    def test_colorscale_min_level_excludes_near_white_end(self):
        """Raising min_level should drop the lightest (near-white) shades."""
        colors = pixelgen_colorscale("blues", min_level=5)
        assert colors == PIXELGEN_ACCENT_COLORS["blues"][4:]
        assert colors[0] not in PIXELGEN_ACCENT_COLORS["blues"][:4]

    def test_colorscale_max_level_excludes_darkest_end(self):
        """Lowering max_level should drop the darkest shades."""
        colors = pixelgen_colorscale("blues", max_level=8)
        assert colors == PIXELGEN_ACCENT_COLORS["blues"][:8]

    def test_colorscale_min_level_applies_before_direction_reversal(self):
        """direction=-1 should reverse the already-clamped [min_level, max_level] range."""
        colors = pixelgen_colorscale("blues", direction=-1, min_level=5, max_level=8)
        assert colors == PIXELGEN_ACCENT_COLORS["blues"][4:8][::-1]

    @pytest.mark.parametrize("kwargs", [{"min_level": 0}, {"max_level": 13}])
    def test_colorscale_level_out_of_range_raises(self, kwargs):
        """min_level/max_level must fall within the defined [1, 12] range."""
        with pytest.raises(ValueError):
            pixelgen_colorscale("blues", **kwargs)

    def test_colorscale_min_level_greater_than_max_level_raises(self):
        """min_level must not exceed max_level."""
        with pytest.raises(ValueError, match="min_level"):
            pixelgen_colorscale("blues", min_level=8, max_level=5)

    def test_sequential_colormap_is_matplotlib_colormap(self):
        """The sequential colormap should be usable directly as a matplotlib cmap."""
        cmap = pixelgen_sequential_colormap("blues")
        assert isinstance(cmap, Colormap)

    def test_sequential_colormap_min_level_avoids_near_white(self):
        """A colormap built with a min_level floor should never render near-white."""
        cmap = pixelgen_sequential_colormap("blues", min_level=5)
        r, g, b, _ = cmap(0.0)
        assert min(r, g, b) < 0.95

    def test_divergent_colormap_midpoint_is_near_white(self):
        """The divergent colormap should pass through white at its midpoint."""
        cmap = pixelgen_divergent_colormap("blues", "reds")
        r, g, b, _ = cmap(0.5)
        assert r > 0.9 and g > 0.9 and b > 0.9

    def test_named_colormap_matches_gradient_endpoints(self):
        """`pixelgen_colormap` should reproduce the same gradient as `pixelgen_gradient`."""
        cmap = pixelgen_colormap("Cherry")
        assert isinstance(cmap, Colormap)
        low = cmap(0.0)
        expected = pixelgen_gradient(2, "Cherry")[0]
        assert (
            "#{:02X}{:02X}{:02X}".format(*(round(c * 255) for c in low[:3]))
            == expected.upper()
        )
