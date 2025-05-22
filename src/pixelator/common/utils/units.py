"""Utility functions for parsing numbers with optional unit suffixes.

Copyright © 2024 Pixelgen Technologies AB.
"""

import re

units = {"K": 10**3, "M": 10**6, "G": 10**9}


def parse_size(s: str) -> int | float:
    """Parse a string as a number with optional unit suffix [K, M, G].

    :param s: The string to parse
    :return: The parsed number as a float
    """
    match = re.match(r"(?P<value>\d+(?:.\d+)?)(?P<unit>[KMGkmg])?$", s)
    if not match:
        raise ValueError(f"Invalid number: {s}")

    number = float(match.group("value"))
    unit = match.group("unit")

    unit_scale = 1
    if unit:
        unit_scale = units.get(unit.upper())  # type: ignore
        if unit_scale is None:
            raise ValueError(f"Invalid unit: {unit}")

    result = float(number) * unit_scale
    if result.is_integer():
        return int(result)
    return result
