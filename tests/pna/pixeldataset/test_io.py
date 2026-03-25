"""Copyright © 2025 Pixelgen Technologies AB."""

from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pixelator.common.utils.testing import adata_assert_equal
from pixelator.pna.pixeldataset.io import (
    PixelDataViewer,
    PixelFileWriter,
    PxlFile,
    QueryBuilder,
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


def _pivot_marker_table(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot joined marker counts into marker columns (same logic as layouts+marker counts)."""
    return (
        df.select(pl.col("*"), val=pl.lit(1))
        .pivot(
            on="marker",
            index=None,
            values="val",
            aggregate_function=pl.len().cast(pl.UInt8),
        )
        .fill_null(0)
    )


class TestPixelFileReader:
    def test_read_edgelist(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(connection, builder.edgelist_query(None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results, _add_sample_name_columns(edgelist_dataframe, "test_sample")
        )

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(connection, builder.edgelist_query(components))
            results = lazy.collect()
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                edgelist_dataframe.filter(pl.col("component") == components[0]),
                "test_sample",
            ),
        )

    def test_read_edgelist_filter_str(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(connection, builder.edgelist_query(components))
            results = lazy.collect()
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                edgelist_dataframe.filter(pl.col("component") == components[0]),
                "test_sample",
            ),
        )

    def test_read_adata(self, pxl_view, adata_data):
        adata_data.obs["sample"] = "test_sample"
        results = pxl_view.read_adata()
        adata_assert_equal(results, adata_data)

    def test_read_metadata(self, pxl_view):
        results = pxl_view.read_metadata()
        assert results == {
            "test_sample": {
                "sample_name": "test_sample",
                "version": "0.1.0",
                "panel_name": "custom_panel",
            }
        }

    def test_read_proximity(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection, builder.proximity_query(None, None)
            )
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results, _add_sample_name_columns(proximity_dataframe, "test_sample")
        )

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection, builder.proximity_query(components, None)
            )
            results = lazy.collect()
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                proximity_dataframe.filter(pl.col("component") == components[0]),
                "test_sample",
            ),
        )

    def test_read_layouts(self, pxl_view, layout_dataframe):
        builder = QueryBuilder()

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection,
                builder.layouts_query(components=None, add_marker_counts=False),
            )
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results,
            _add_sample_name_columns(layout_dataframe, "test_sample"),
            check_row_order=False,
        )

    def test_read_layouts_add_marker_counts(self, snapshot, pxl_view):
        builder = QueryBuilder()

        with pxl_view as connection:
            results_df = pxl_view.execute_eager(
                connection,
                builder.layouts_query(components=None, add_marker_counts=True),
            )
        results_df = _pivot_marker_table(results_df).drop(
            ["umi", "marker"], strict=False
        )
        results_df = results_df.select(sorted(results_df.columns))

        result = StringIO()
        results_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")

    def test_read_layouts_filter(self, pxl_view, layout_dataframe):
        builder = QueryBuilder()
        components = ["040b1570c7d0f28f"]

        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection,
                builder.layouts_query(components=components, add_marker_counts=False),
            )
            results = lazy.collect()
        assert_frame_equal(
            results,
            _add_sample_name_columns(
                layout_dataframe.filter(pl.col("component") == components[0]),
                "test_sample",
            ),
            check_row_order=False,
        )


class TestPixelDataViewer:
    def test_sample_names(self, pxl_view: PixelDataViewer):
        assert pxl_view.sample_names() == ["test_sample"]

    def test_read_adata_from_sample(self, pxl_view: PixelDataViewer, adata_data):
        with pxl_view as connection:
            res = pxl_view.read_adata_from_sample(connection, "test_sample")
        adata_assert_equal(res, adata_data)

    def test_read_adata(self, pxl_view: PixelDataViewer, adata_data):
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "test_sample"

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
        builder = QueryBuilder()
        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(connection, builder.edgelist_query(None))
            result = lazy.collect()
        assert_frame_equal(
            result, edgelist_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(connection, builder.edgelist_query(components))
            result = lazy.collect()
        assert_frame_equal(
            result,
            edgelist_dataframe.filter(
                pl.col("component") == components[0]
            ).with_columns(sample=pl.lit("test_sample")),
        )

    def test_read_edgelist_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view as connection:
            result = pxl_view.execute_scalar(
                connection, builder.edgelist_len_query(None)
            )
        assert result == 57

    def test_read_edgelist_len_filter(self, pxl_view):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view as connection:
            result = pxl_view.execute_scalar(
                connection, builder.edgelist_len_query(components)
            )
        assert result == 23

    def test_read_edgelist_stream(self, pxl_view, edgelist_dataframe):
        # Turn the stream into a DataFrame for comparison
        builder = QueryBuilder()
        with pxl_view as connection:
            result = pl.from_arrow(
                pxl_view.execute_arrow_reader(
                    connection,
                    builder.edgelist_query(None),
                    batch_size=1_000_000,
                )
            )
        assert_frame_equal(
            result, edgelist_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_metadata(self, pxl_view):
        result = pxl_view.read_metadata()
        assert result == {
            "test_sample": {
                "sample_name": "test_sample",
                "version": "0.1.0",
                "panel_name": "custom_panel",
            }
        }

    def test_read_layouts(self, pxl_view, layout_dataframe):
        builder = QueryBuilder()
        with pxl_view as connection:
            result = (
                pxl_view.execute_lazy(
                    connection, builder.layouts_query(None, add_marker_counts=False)
                )
                .collect()
                .sort(["component", "index"])
            )
        assert_frame_equal(
            result,
            layout_dataframe.with_columns(sample=pl.lit("test_sample")).sort(
                ["component", "index"]
            ),
        )

    def test_read_layouts_add_marker_counts(self, snapshot, pxl_view, layout_dataframe):
        builder = QueryBuilder()
        with pxl_view as connection:
            result_df = pxl_view.execute_eager(
                connection,
                builder.layouts_query(None, add_marker_counts=True),
            )
        result_df = _pivot_marker_table(result_df).drop(["umi", "marker"], strict=False)
        result_df = result_df.sort("component").select(sorted(result_df.columns))

        result = StringIO()
        result_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "read_view_layouts.csv")

    def test_read_layouts_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view as connection:
            result = pxl_view.execute_scalar(
                connection, builder.layouts_len_query(None)
            )
        assert result == 34

    def test_read_layouts_len_filter(self, pxl_view):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view as connection:
            result = pxl_view.execute_scalar(
                connection, builder.layouts_len_query(components)
            )
        assert result == 11

    def test_read_proximity(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()
        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection, builder.proximity_query(None, None)
            )
            result = lazy.collect()
        assert_frame_equal(
            result, proximity_dataframe.with_columns(sample=pl.lit("test_sample"))
        )

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view as connection:
            lazy = pxl_view.execute_lazy(
                connection, builder.proximity_query(components, None)
            )
            result = lazy.collect()
        assert_frame_equal(
            result,
            proximity_dataframe.filter(
                pl.col("component") == components[0]
            ).with_columns(sample=pl.lit("test_sample")),
            check_column_order=False,
        )

    def test_read_proximity_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view as connection:
            result = pxl_view.execute_scalar(
                connection, builder.proximity_len_query(None, None)
            )
        assert result == 3
