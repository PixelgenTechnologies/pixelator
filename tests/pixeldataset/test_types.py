import pandas as pd
from pixelator.pixeldataset.types import MpxEdgeList

from pandas.testing import assert_frame_equal


def test_mpxedgelist_empty():
    edgelist = MpxEdgeList.empty()
    assert edgelist.is_empty()


def test_mpxedgelist_df():
    df = pd.DataFrame(
        {
            "count": [1, 2, 3],
            "upia": ["A", "B", "C"],
            "upib": ["D", "E", "F"],
            "umi": ["G", "H", "I"],
            "marker": ["M1", "M2", "M3"],
            "sequence": ["S1", "S2", "S3"],
            "component": ["C1", "C2", "C3"],
        }
    )
    edgelist = MpxEdgeList(df)
    assert_frame_equal(edgelist.df, df, check_dtype=False, check_categorical=False)


def test_mpxedgelist_is_empty():
    df_empty = pd.DataFrame()
    df_non_empty = pd.DataFrame(
        {
            "count": [1, 2, 3],
            "upia": ["A", "B", "C"],
            "upib": ["D", "E", "F"],
            "umi": ["G", "H", "I"],
            "marker": ["M1", "M2", "M3"],
            "sequence": ["S1", "S2", "S3"],
            "component": ["C1", "C2", "C3"],
        }
    )
    edgelist_empty = MpxEdgeList(df_empty)
    edgelist_non_empty = MpxEdgeList(df_non_empty)
    assert edgelist_empty.is_empty()
    assert not edgelist_non_empty.is_empty()


def test_mpxedgelist_copy():
    df = pd.DataFrame(
        {
            "count": [1, 2, 3],
            "upia": ["A", "B", "C"],
            "upib": ["D", "E", "F"],
            "umi": ["G", "H", "I"],
            "marker": ["M1", "M2", "M3"],
            "sequence": ["S1", "S2", "S3"],
            "component": ["C1", "C2", "C3"],
        }
    )
    edgelist = MpxEdgeList(df)
    edgelist_copy = edgelist.copy()
    assert_frame_equal(
        edgelist.df, edgelist_copy.df, check_dtype=False, check_categorical=False
    )
    assert edgelist_copy.df is not edgelist.df


def test_mpxedgelist_enforce_edgelist_types():
    df = pd.DataFrame(
        {
            "count": [1, 2, 3],
            "upia": ["A", "B", "C"],
            "upib": ["D", "E", "F"],
            "umi": ["G", "H", "I"],
            "marker": ["M1", "M2", "M3"],
            "sequence": ["S1", "S2", "S3"],
            "component": ["C1", "C2", "C3"],
        }
    )
    edgelist = MpxEdgeList(df)
    edgelist_enforced = edgelist._enforce_edgelist_types(df)

    expected_types = {
        "count": "uint16",
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
        "component": "category",
    }
    type_dict = edgelist_enforced.dtypes.to_dict()
    assert all(expected_types[key] == type_ for key, type_ in type_dict.items())


def test_mpxedgelist_enforce_edgelist_types_empty():
    df_empty = pd.DataFrame()
    edgelist_empty = MpxEdgeList(df_empty)
    edgelist_enforced_empty = edgelist_empty._enforce_edgelist_types(df_empty)
    assert edgelist_enforced_empty.empty


def test_mpxedgelist_enforce_edgelist_types_with_sample():
    df = pd.DataFrame(
        {
            "count": [1, 2, 3],
            "upia": ["A", "B", "C"],
            "upib": ["D", "E", "F"],
            "umi": ["G", "H", "I"],
            "marker": ["M1", "M2", "M3"],
            "sequence": ["S1", "S2", "S3"],
            "component": ["C1", "C2", "C3"],
            "sample": ["S1", "S2", "S3"],
        }
    )
    edgelist = MpxEdgeList(df)
    edgelist_enforced = edgelist._enforce_edgelist_types(df)

    expected_types = {
        "count": "uint16",
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
        "component": "category",
        "sample": "category",
    }
    type_dict = edgelist_enforced.dtypes.to_dict()
    assert all(expected_types[key] == type_ for key, type_ in type_dict.items())
