"""Module relating to pixelator feature flags.

A feature flags optionally turns on features in pixelator that are not
yet ready for production. This is useful for testing and development work.

To use a feature flag in the code you can use the following pattern:

```
from pixelator.feature_flags import MY_FEATURE_FLAG

if MY_FEATURE_FLAG:
    # Do something
    ...

```

The value of the feature flag is determined by an environment variable,
which should have the name `PIXELATOR_<NAME_OF_FLAG>`.

Copyright © 2024 Pixelgen Technologies AB.
"""

import os
from typing import Any


def _load_flags_from_environment(flag: str, default: Any | None = None) -> Any:
    """Load a feature flag from environment variables.

    Args:
    flag: Feature flag name without the ``PIXELATOR_`` prefix.
    default: Value returned when the environment variable is unset.

    """
    return os.environ.get(f"PIXELATOR_{flag}", default)


MY_FEATURE_FLAG = bool(_load_flags_from_environment("MY_FEATURE_FLAG", default=False))
