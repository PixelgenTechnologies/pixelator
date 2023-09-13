"""
Tests for the antibody aggregates detection module

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pixelator.annotate.aggregates import call_aggregates

NBR_OF_MARKERS = 100
NBR_OF_NON_ZERO_MARKERS = 10
NBR_OF_AGGREGATE_COMPONENTS = 10
NBR_OF_NORMAL_COMPONENTS = 100
NBR_OF_UNSPECIFIC_COMPONENTS = 10


@pytest.fixture(name="aggregates")
def aggregates_fixture():
    random_state = np.random.default_rng(seed=1)
    no_signal_markers = random_state.poisson(
        5, (NBR_OF_AGGREGATE_COMPONENTS, NBR_OF_MARKERS - 1)
    )
    aggregates_markers = random_state.normal(
        2000, 100, (NBR_OF_AGGREGATE_COMPONENTS, 1)
    )
    aggregates_markers[aggregates_markers < 0] = 0
    test_data = np.concatenate((no_signal_markers, aggregates_markers), axis=1)
    return test_data


@pytest.fixture(name="normals")
def normal_fixture():
    random_state = np.random.default_rng(seed=1)
    # Normal cells have expression for a handful of markers, and
    # the rest are around zero
    normal_markers = random_state.normal(
        1500, 200, (NBR_OF_NORMAL_COMPONENTS, NBR_OF_NON_ZERO_MARKERS)
    )
    no_signal_markers = random_state.poisson(
        5, (NBR_OF_NORMAL_COMPONENTS, NBR_OF_MARKERS - NBR_OF_NON_ZERO_MARKERS)
    )
    normal_markers[normal_markers < 0] = 0
    all_markers = np.concatenate((no_signal_markers, normal_markers), axis=1)
    return all_markers


@pytest.fixture(name="unspecifics")
def unspecific_fixture():
    random_state = np.random.default_rng(seed=1)
    no_signal_markers = random_state.poisson(
        5, (NBR_OF_UNSPECIFIC_COMPONENTS, NBR_OF_MARKERS)
    )
    no_signal_markers[no_signal_markers < 0] = 0
    return no_signal_markers


@pytest.fixture(name="aggregates_data")
def mixed_data_fixture(aggregates, normals, unspecifics):
    return np.concatenate((aggregates, normals, unspecifics), axis=0)


def generate_anndata(x):
    adata = AnnData(X=x)
    components = [f"CMP{idx}" for idx in range(len(x))]
    adata.obs = pd.DataFrame({"component": components})
    adata.obs_names = components
    return adata


@pytest.fixture(name="aggregates_adata")
def aggregate_adata_fixture(aggregates_data):
    return generate_anndata(aggregates_data)


@pytest.fixture(name="no_aggregates_adata")
def no_aggregates_adata_fixture(normals):
    return generate_anndata(normals)


def test_find_aggregates(aggregates_adata):
    results = call_aggregates(adata=aggregates_adata, inplace=False)
    assert np.sum(results.obs["tau_type"] == "high") == NBR_OF_AGGREGATE_COMPONENTS
    assert np.any(results.obs["tau"])


def test_find_aggregates_writes_limits_to_uns(aggregates_adata):
    results = call_aggregates(adata=aggregates_adata, inplace=False)
    assert np.sum(results.obs["tau_type"] == "high") == NBR_OF_AGGREGATE_COMPONENTS
    assert np.any(results.obs["tau"])

    assert results.uns["tau_thresholds"]["tau_upper_hard_limit"] == 0.995
    assert results.uns["tau_thresholds"]["tau_upper_iqr_limit"] == pytest.approx(
        0.94, rel=0.01
    )
    assert results.uns["tau_thresholds"]["tau_lower_iqr_limit"] == pytest.approx(
        0.88, rel=0.01
    )


def test_find_aggregates_inplace(aggregates_adata):
    call_aggregates(adata=aggregates_adata, inplace=True)
    assert (
        np.sum(aggregates_adata.obs["tau_type"] == "high")
        == NBR_OF_AGGREGATE_COMPONENTS
    )
    assert np.any(aggregates_adata.obs["tau"])


def test_find_in_data_with_no_aggregates(no_aggregates_adata):
    results = call_aggregates(adata=no_aggregates_adata, inplace=False)
    # Two false positives
    assert np.sum(results.obs["tau_type"] == "normal") == NBR_OF_NORMAL_COMPONENTS - 2


def test_find_in_data_with_no_counts_in_component(aggregates_adata):
    counts = np.concatenate((aggregates_adata.X, np.zeros((1, NBR_OF_MARKERS))), axis=0)
    x = generate_anndata(counts)
    results = call_aggregates(adata=x, inplace=False)
    # One false positives
    assert np.sum(results.obs["tau_type"] == "high") == NBR_OF_AGGREGATE_COMPONENTS


def test_find_unspecific(aggregates_adata):
    results = call_aggregates(adata=aggregates_adata, inplace=False)
    assert np.sum(results.obs["tau_type"] == "low") == NBR_OF_UNSPECIFIC_COMPONENTS
