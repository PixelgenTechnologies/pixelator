"""Copyright © 2025 Pixelgen Technologies AB."""

import pytest

from pixelator.pna.utils.units import parse_size


def test_parse_size():
    """Verify parse size."""
    assert parse_size("123") == 123
    assert parse_size("123K") == 123000
    assert parse_size("123M") == 123000000
    assert parse_size("123G") == 123000000000
    assert parse_size("123k") == 123000
    assert parse_size("123m") == 123000000
    assert parse_size("123g") == 123000000000

    assert parse_size("123.3") == 123.3
    assert parse_size("123.3K") == 123300
    assert parse_size("123.3M") == 123300000
    assert parse_size("123.3G") == 123300000000

    with pytest.raises(ValueError):
        parse_size("123KB")

    with pytest.raises(ValueError):
        parse_size("123.4KB")

    with pytest.raises(ValueError):
        parse_size("123.4x")

    with pytest.raises(ValueError):
        parse_size("123.4 K")
