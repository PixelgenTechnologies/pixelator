"""Copyright (c) 2023 Pixelgen Technologies AB."""

import os


def _check_if_true(key):
    value = os.getenv(key, default="")
    if value.lower() == "true":
        return True
    return False


ENABLE_ALTERNATIVE_POLARIZATION: bool = _check_if_true(
    "PIXELATOR_ENABLE_ALTERNATIVE_POLARIZATION"
)
