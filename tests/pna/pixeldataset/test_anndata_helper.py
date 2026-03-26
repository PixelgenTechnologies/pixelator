"""Tests for AnnDataHelper wrapper behavior.

Copyright © 2025 Pixelgen Technologies AB.
"""

import pytest

from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper


class TestAnnDataHelper:
    def test_anndata_helper_matches_dataset_adata_no_transforms(
        self, pxl_dataset, adata_data
    ):
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"

        helper = AnnDataHelper(pxl_dataset.view)
        res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        adata_assert_equal(res, adata_data)

    def test_anndata_helper_respects_component_and_marker_filters(self, pxl_dataset):
        filtered = pxl_dataset.filter(
            components={"fc07dea9b679aca7"},
            markers={"MarkerA"},
        )

        helper = AnnDataHelper(
            pxl_dataset.view,
            components={"fc07dea9b679aca7"},
            markers={"MarkerA"},
        )
        res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)

        assert set(res.obs.index) == {"fc07dea9b679aca7"}
        assert set(res.var.index) == {"MarkerA"}

        adata_assert_equal(
            res,
            filtered.adata(add_clr_transform=False, add_log1p_transform=False),
        )

    def test_anndata_helper_does_not_mutate_original(self, pxl_dataset):
        helper = AnnDataHelper(pxl_dataset.view)

        adata = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
        adata.layers["new_layer"] = adata.X + 1

        assert "new_layer" in adata.layers.keys()
        assert (
            "new_layer"
            not in helper.read_adata(
                add_clr_transform=False, add_log1p_transform=False
            ).layers.keys()
        )


@pytest.mark.parametrize(
    "components,markers",
    [
        (None, None),
        ({"fc07dea9b679aca7"}, None),
        (None, {"MarkerA"}),
        ({"fc07dea9b679aca7"}, {"MarkerA"}),
    ],
)
def test_anndata_helper_basic_smoke(pxl_dataset, components, markers):
    helper = AnnDataHelper(pxl_dataset.view, components=components, markers=markers)
    res = helper.read_adata(add_clr_transform=False, add_log1p_transform=False)
    assert res.n_obs >= 0
    assert res.n_vars >= 0
