"""Copyright © 2023 Pixelgen Technologies AB."""

import pytest

from pixelator.utils import flatten


@pytest.mark.parametrize(
    "input,expected",
    (
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([1, 2, ["test"]], [1, 2, "test"]),
        ([1, 2, ("test", 3)], [1, 2, "test", 3]),
    ),
)
def test_flatten(input, expected):
    assert list(flatten(input)) == expected
