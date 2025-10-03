"""Copyright © 2025 Pixelgen Technologies AB."""

from io import StringIO

import polars as pl
import pytest
from anndata.tests.helpers import assert_equal as adata_assert_equal
from polars.testing import assert_frame_equal

from pixelator.pna.pixeldataset.io import (
    PixelDataQuerier,
    PixelDataViewer,
    PixelFileWriter,
    PxlFile,
)


class TestPixelFileWriter:
    def test_write_edgelist(self, tmp_path, edgelist_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_edgelist(edgelist_parquet_path)

    def test_write_adata(self, tmp_path, adata_data):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_adata(adata_data)

    def test_write_metadata(self, tmp_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_metadata({"sample": "test_sample", "version": "0.1.0"})

    def test_write_proximity(self, tmp_path, proximity_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_proximity(proximity_parquet_path)

    def test_write_layouts(self, tmp_path, layout_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_layouts(layout_parquet_path)


@pytest.fixture(name="pxl_view")
def pxl_view_fixture(pxl_file):
    return PixelDataViewer.from_files([PxlFile(pxl_file)])


def _add_sample_name_columns(df, sample_name):
    return df.with_columns(sample=pl.lit(sample_name))


class TestPixelFileReader:
    def test_read_edgelist(self, pxl_view, edgelist_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_edgelist()
        assert_frame_equal(
            results, _add_sample_name_columns(edgelist_dataframe, "test_sample")
        )

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_edgelist(components={"fc07dea9b679aca7"})
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                edgelist_dataframe.filter(pl.col("component") == "fc07dea9b679aca7"),
                "test_sample",
            ),
        )

    def test_read_edgelist_filter_str(self, pxl_view, edgelist_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_edgelist(components="fc07dea9b679aca7")
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                edgelist_dataframe.filter(pl.col("component") == "fc07dea9b679aca7"),
                "test_sample",
            ),
        )

    def test_read_adata(self, pxl_view, adata_data):
        reader = PixelDataQuerier(pxl_view)

        adata_data.obs["sample"] = "test_sample"
        adata_data.uns = {
            'my_key': {
                'with_nesting': ['and array', 'of values'],
                'another_key': 1.0
            }
        }


        results = reader.read_adata()
        adata_assert_equal(results, adata_data)

    def test_read_metadata(self, pxl_view):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_metadata()
        assert results == {
            "test_sample": {"sample_name": "test_sample", "version": "0.1.0"}
        }

    def test_read_proximity(self, pxl_view, proximity_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_proximity()
        assert_frame_equal(
            results, _add_sample_name_columns(proximity_dataframe, "test_sample")
        )

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_proximity(components={"fc07dea9b679aca7"})
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                proximity_dataframe.filter(pl.col("component") == "fc07dea9b679aca7"),
                "test_sample",
            ),
        )

    def test_read_layouts(self, pxl_view, layout_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_layouts(add_marker_counts=False)
        assert_frame_equal(
            results,
            _add_sample_name_columns(layout_dataframe, "test_sample"),
            check_row_order=False,
        )

    def test_read_layouts_add_marker_counts(self, snapshot, pxl_view):
        reader = PixelDataQuerier(pxl_view)

        results_df = reader.read_layouts(add_marker_counts=True)
        results_df = results_df.select(sorted(results_df.columns))

        result = StringIO()
        results_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")

    def test_read_layouts_filter(self, pxl_view, layout_dataframe):
        reader = PixelDataQuerier(pxl_view)

        results = reader.read_layouts(
            components={"040b1570c7d0f28f"}, add_marker_counts=False
        )
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                layout_dataframe.filter(pl.col("component") == "040b1570c7d0f28f"),
                "test_sample",
            ),
            check_row_order=False,
        )


class TestPixelDataViewer:
    def test_sample_names(self, pxl_view):
        assert pxl_view.sample_names() == ["test_sample"]

    def test_read_adata_from_sample(self, pxl_view: PixelDataViewer, adata_data):
        res = pxl_view.read_adata_from_sample("test_sample")
        adata_assert_equal(res, adata_data)

    def test_read_adata(self, pxl_view, adata_data):
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"
        adata_data.uns = {
            'my_key': {
                'with_nesting': ['and array', 'of values'],
                'another_key': 1.0
            }
        }

        res = pxl_view.read_adata()
        adata_assert_equal(res, adata_data)

    def test_view_has_all_tables(self, pxl_view):
        with pxl_view as view:
            result = view.sql("SHOW ALL TABLES").pl()

        assert_frame_equal(
            result.filter(pl.col("database") == "memory").select("name").sort("name"),
            pl.DataFrame(
                {"name": ["edgelist", "proximity", "layouts", "metadata"]}
            ).sort("name"),
        )


class TestPxlDataQuerier:
    def test_read_edgelist(self, pxl_view, edgelist_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_edgelist()
        assert_frame_equal(
            result, edgelist_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_edgelist(components={"fc07dea9b679aca7"})
        assert_frame_equal(
            result,
            edgelist_dataframe.filter(
                pl.col("component") == "fc07dea9b679aca7"
            ).with_columns(sample=pl.lit("test_sample")),
        )

    def test_read_edgelist_len(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_edgelist_len()
        assert result == 57

    def test_read_edgelist_len_filter(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_edgelist_len(components={"fc07dea9b679aca7"})
        assert result == 23

    def test_read_edgelist_stream(self, pxl_view, edgelist_dataframe):
        querier = PixelDataQuerier(pxl_view)
        # Turn the stream into a DataFrame for comparison
        result = pl.from_arrow(querier.read_edgelist_stream())
        assert_frame_equal(
            result, edgelist_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_metadata(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_metadata()
        assert result == {
            "test_sample": {"sample_name": "test_sample", "version": "0.1.0"}
        }

    def test_read_layouts(self, pxl_view, layout_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_layouts().sort(["component", "index"])
        assert_frame_equal(
            result,
            layout_dataframe.with_columns(sample=pl.lit("test_sample")).sort(
                ["component", "index"]
            ),
        )

    def test_read_layouts_add_marker_counts(self, snapshot, pxl_view, layout_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result_df = querier.read_layouts(add_marker_counts=True)
        result_df = result_df.sort("component").select(sorted(result_df.columns))

        result = StringIO()
        result_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "read_view_layouts.csv")

    def test_read_layouts_len(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_layouts_len()
        assert result == 34

    def test_read_layouts_len_filter(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_layouts_len(components={"fc07dea9b679aca7"})
        assert result == 11

    def test_read_proximity(self, pxl_view, proximity_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_proximity()
        assert_frame_equal(
            result, proximity_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_proximity(components={"fc07dea9b679aca7"})
        assert_frame_equal(
            result,
            proximity_dataframe.filter(
                pl.col("component") == "fc07dea9b679aca7"
            ).with_columns(sample=pl.lit("test_sample")),
            check_column_order=False,
        )

    def test_read_proximity_len(self, pxl_view):
        querier = PixelDataQuerier(pxl_view)
        result = querier.read_proximity_len()
        assert result == 3
