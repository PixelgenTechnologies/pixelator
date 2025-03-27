"""Tools useful in testing.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pandas as pd


def enforce_edgelist_types_for_tests(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Enforce the types of a edgelist dataframe, in testing."""
    type_dict = {
        "count": "uint16",
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
    }
    if "component" in edgelist.columns:
        type_dict["component"] = "category"

    return edgelist.astype(type_dict)
