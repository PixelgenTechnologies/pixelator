"""Tests for the marks module.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from pixelator.marks import experimental

import pytest


def test_experimental():
    @experimental
    def my_func():
        return 3

    with pytest.warns() as w:
        result = my_func()
        assert result == 3
        assert len(w) == 1
        assert w[0].message.args[0] == (
            "The function `my_func` is experimental, "
            "it might be removed or the API might change without notice."
        )
