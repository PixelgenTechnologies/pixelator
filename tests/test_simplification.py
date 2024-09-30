"""Tests for line simplification implementation.

Copyright © 2024 Pixelgen Technologies AB.
"""

import numpy as np
import pytest

from pixelator.utils.simplification import simplify_line_rdp


@pytest.mark.parametrize(
    "input,expected,epsilon",
    [
        (
            np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
            np.array([True, False, False, True], dtype=bool),
            0.0,
        ),
        (
            np.array([[0.0, 0.0], [5.0, 4.0], [11.0, 5.5], [17.3, 3.2], [27.8, 0.1]]),
            np.array([True, True, True, False, True], dtype=bool),
            1.0,
        ),
    ],
)
def test_simplification_mask(input, expected, epsilon):
    mask = simplify_line_rdp(input, epsilon, return_mask=True)
    assert np.array_equal(mask, expected)
