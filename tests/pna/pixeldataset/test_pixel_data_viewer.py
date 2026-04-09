"""Copyright © 2025 Pixelgen Technologies AB."""

from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pixelator.pna.pixeldataset.io import (
    PixelDataViewer,
    PixelDataViewerSession,
    PxlFile,
    Query,
    QueryBuilder,
)


@pytest.fixture(name="pxl_view")
def pxl_view_fixture(pxl_file):
    return PixelDataViewer.from_files([PxlFile(pxl_file)])


def _with_sample(df: pl.DataFrame, sample_name: str = "test_sample") -> pl.DataFrame:
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


class TestPixelDataViewerQueries:
    """Session query paths: lazy frames, scalars, Arrow streaming, and snapshots."""

    def test_read_edgelist(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.edgelist_query(None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(results, _with_sample(edgelist_dataframe))

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.edgelist_query(components))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results,
            _with_sample(
                edgelist_dataframe.filter(pl.col("component") == components[0])
            ),
        )

    def test_read_edgelist_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.edgelist_len_query(None))
        assert result == 57

    def test_read_edgelist_len_filter(self, pxl_view):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.edgelist_len_query(components))
        assert result == 23

    def test_read_edgelist_stream(self, pxl_view, edgelist_dataframe):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = pl.from_arrow(
                session.execute_arrow_reader(
                    builder.edgelist_query(None),
                    batch_size=1_000_000,
                )
            )
        assert_frame_equal(result, _with_sample(edgelist_dataframe))

    def test_read_proximity(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.proximity_query(None, None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(results, _with_sample(proximity_dataframe))

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.proximity_query(components, None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results,
            _with_sample(
                proximity_dataframe.filter(pl.col("component") == components[0])
            ),
            check_column_order=False,
        )

    def test_read_proximity_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.proximity_len_query(None, None))
        assert result == 3

    def test_read_layouts(self, pxl_view, layout_dataframe):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(
                builder.layouts_query(components=None, add_marker_counts=False),
            )
            assert isinstance(lazy, pl.LazyFrame)
            result = lazy.collect().sort(["component", "index"])
        assert_frame_equal(
            result,
            _with_sample(layout_dataframe).sort(["component", "index"]),
        )

    def test_read_layouts_add_marker_counts(self, snapshot, pxl_view):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result_df = session.execute_eager(
                builder.layouts_query(components=None, add_marker_counts=True),
            )
        result_df = _pivot_marker_table(result_df).drop(["umi", "marker"], strict=False)
        result_df = result_df.sort("component").select(sorted(result_df.columns))

        result = StringIO()
        result_df.write_csv(result)
        snapshot.assert_match(result.getvalue(), "layouts.csv")

    def test_read_layouts_filter(self, pxl_view, layout_dataframe):
        builder = QueryBuilder()
        components = ["040b1570c7d0f28f"]
        with pxl_view.open() as session:
            lazy = session.execute_lazy(
                builder.layouts_query(components=components, add_marker_counts=False),
            )
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(
            results,
            _with_sample(layout_dataframe.filter(pl.col("component") == components[0])),
            check_row_order=False,
        )

    def test_read_layouts_len(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.layouts_len_query(None))
        assert result == 34

    def test_read_layouts_len_filter(self, pxl_view):
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.layouts_len_query(components))
        assert result == 11


class TestPixelDataViewer:
    def test_sample_names(self, pxl_view: PixelDataViewer):
        assert pxl_view.sample_names() == ["test_sample"]

    def test_view_has_all_tables(self, pxl_view):
        with pxl_view.open() as session:
            result = session.execute_eager(Query("SHOW ALL TABLES", {}))

        assert_frame_equal(
            result.filter(pl.col("database") == "memory").select("name").sort("name"),
            pl.DataFrame(
                {"name": ["edgelist", "proximity", "layouts", "metadata"]}
            ).sort("name"),
        )

    def test_open_returns_session_context_manager(self, pxl_view):
        unopened = pxl_view.open()
        assert isinstance(unopened, PixelDataViewerSession)

        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.edgelist_query(None))
            assert isinstance(lazy, pl.LazyFrame)
            _ = lazy.collect()

    def test_nested_open_sessions_are_independent(self, pxl_view):
        builder = QueryBuilder()
        with pxl_view.open() as outer:
            outer_count = outer.execute_scalar(builder.edgelist_len_query(None))
            with pxl_view.open() as inner:
                inner_count = inner.execute_scalar(builder.edgelist_len_query(None))
            assert outer.execute_scalar(builder.edgelist_len_query(None)) == outer_count
        assert inner_count == outer_count == 57

    def test_execute_without_open_session_raises(self, pxl_view):
        session = pxl_view.open()
        with pytest.raises(RuntimeError, match="not open"):
            session.execute_eager(Query("SELECT 1", {}))
