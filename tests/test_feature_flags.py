"""Copyright © 2024 Pixelgen Technologies AB."""

import os

from pixelator.feature_flags import _load_flags_from_environment


def test_load_flags_from_environment():
    # Test when the environment variable is set
    os.environ["PIXELATOR_FLAG"] = "True"
    result = bool(_load_flags_from_environment("FLAG", default=False))
    assert result is True

    # Test when the environment variable is not set
    os.environ.pop("PIXELATOR_FLAG", None)
    result = _load_flags_from_environment("FLAG", default=False)
    assert result is False

    # Test when the environment variable is set to a different value
    os.environ["PIXELATOR_FLAG"] = "Hello"
    result = _load_flags_from_environment("FLAG", default=True)
    assert result == "Hello"

    # Test when the environment variable is not set and no default value is provided
    os.environ.pop("PIXELATOR_FLAG", None)
    result = _load_flags_from_environment("FLAG")
    assert result is None
