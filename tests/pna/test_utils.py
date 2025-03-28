"""Copyright © 2025 Pixelgen Technologies AB."""

import pytest

from pixelator.pna.utils import get_demux_filename_info
from pixelator.pna.utils.units import parse_size


def test_parse_size():
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


def test_get_demux_filename_info():
    f1 = "Sample01_ABC_blah.part_000.demux.parquet"
    assert get_demux_filename_info(f1) == ("Sample01_ABC_blah", 0)

    with pytest.raises(ValueError):
        f2_wrong_suffix = "Sample01_ABC_blah.part_000.collapse.parquet"
        get_demux_filename_info(f2_wrong_suffix)

    # Weird name with too many suffixes
    t = "PNA055_Sample07_filtered_S7.demux.m1.collapse.m1.part_010.parquet"
    res = get_demux_filename_info(t)

    assert res[0] == "PNA055_Sample07_filtered_S7"
    assert res[1] == 10
