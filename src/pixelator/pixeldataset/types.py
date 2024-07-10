import sys
from typing import Protocol

import pandas as pd
from typing_extensions import Self


class EdgeListProtocol(Protocol):
    @property
    def df(self) -> pd.DataFrame: ...

    def is_empty(self) -> bool: ...

    @staticmethod
    def empty() -> Self:
        return MpxEdgeList(pd.DataFrame())

    def copy(self) -> Self: ...


class MpxEdgeList(EdgeListProtocol):
    def __init__(self, df: pd.DataFrame):
        self._df = self._enforce_edgelist_types(df)

    @staticmethod
    def empty():
        return MpxEdgeList(pd.DataFrame())

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def is_empty(self) -> bool:
        return self._df.empty

    @staticmethod
    def _enforce_edgelist_types(edgelist: pd.DataFrame) -> pd.DataFrame:
        """Enforce the data types of the edgelist."""
        # Enforcing the types of the edgelist reduces the memory
        # usage by roughly 2/3s.

        required_types = {
            "count": "uint16",
            "upia": "category",
            "upib": "category",
            "umi": "category",
            "marker": "category",
            "sequence": "category",
            "component": "category",
        }

        # if the dataframe is empty just enforce the types.
        if edgelist.shape[0] == 0:
            edgelist = pd.DataFrame(columns=required_types.keys())

        # If we have the optional sample column, this should be
        # set to use a categorical type
        if "sample" in edgelist.columns:
            required_types["sample"] = "category"

        # If all of the prescribed types are already set, just return the edgelist
        type_dict = edgelist.dtypes.to_dict()
        if all(type_dict[key] == type_ for key, type_ in required_types.items()):
            return edgelist

        return edgelist.astype(
            required_types,
            # Do not copy here, since otherwise the memory usage
            # blows up
            copy=False,
        )

    def copy(self) -> Self:
        return MpxEdgeList(self._df.copy())


this_module = sys.modules[__name__]

this_module._edgelist_class = MpxEdgeList


def set_edgelist_class(cls):
    this_module._edgelist_class = cls


def get_edgelist_class():
    return this_module._edgelist_class


def EdgeList(df: pd.DataFrame | None) -> EdgeListProtocol:
    edgelist_class = this_module._edgelist_class

    if df is None:
        return edgelist_class.empty()
    return edgelist_class(df)
