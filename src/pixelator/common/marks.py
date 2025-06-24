"""Module with functions that relate to marking functions in pixelator.

Copyright © 2023 Pixelgen Technologies AB.
"""

import warnings
from functools import wraps


def experimental(f):
    """Annotation for experimental pixelator functions."""

    @wraps(f)
    def wrapper(*args, **kwds):
        warnings.warn(
            f"The function `{f.__name__}` is experimental, it might be removed or "
            "the API might change without notice.",
            stacklevel=2,
        )
        return f(*args, **kwds)

    return wrapper
