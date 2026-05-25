"""Copyright © 2025 Pixelgen Technologies AB."""

import shutil
from io import StringIO
from pathlib import Path

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
    """Pxl view fixture.

    Args:
        pxl_file: pxl file.
    """
    return PixelDataViewer.from_files([PxlFile(pxl_file)])


def _expected_normalized_db_name(sample_name: str) -> str:
    """Mirror PixelDataViewer sample-name to DuckDB attach name normalization.

    Args:
        sample_name: Sample name.
    """
    return f"db_{sample_name.replace('-', '_').replace(' ', '_')}"


def _with_sample(df: pl.DataFrame, sample_name: str = "test_sample") -> pl.DataFrame:
    return df.with_columns(sample=pl.lit(sample_name))


def _pivot_marker_table(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot joined marker counts into marker columns (same logic as layouts+marker counts).

    Args:
        df: Df.
    """
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
        """Verify read edgelist.

        Args:
            pxl_view: pxl view.
            edgelist_dataframe: edgelist dataframe.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.edgelist_query(None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(results, _with_sample(edgelist_dataframe))

    def test_read_edgelist_filter(self, pxl_view, edgelist_dataframe):
        """Verify read edgelist filter.

        Args:
            pxl_view: pxl view.
            edgelist_dataframe: edgelist dataframe.
        """
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
        """Verify read edgelist len.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.edgelist_len_query(None))
        assert result == 57

    def test_read_edgelist_len_filter(self, pxl_view):
        """Verify read edgelist len filter.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.edgelist_len_query(components))
        assert result == 23

    def test_read_edgelist_stream(self, pxl_view, edgelist_dataframe):
        """Verify read edgelist stream.

        Args:
            pxl_view: pxl view.
            edgelist_dataframe: edgelist dataframe.
        """
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
        """Verify read proximity.

        Args:
            pxl_view: pxl view.
            proximity_dataframe: proximity dataframe.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.proximity_query(None, None))
            assert isinstance(lazy, pl.LazyFrame)
            results = lazy.collect()
        assert_frame_equal(results, _with_sample(proximity_dataframe))

    def test_read_proximity_filter(self, pxl_view, proximity_dataframe):
        """Verify read proximity filter.

        Args:
            pxl_view: pxl view.
            proximity_dataframe: proximity dataframe.
        """
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
        """Verify read proximity len.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.proximity_len_query(None, None))
        assert result == 3

    def test_read_layouts(self, pxl_view, layout_dataframe):
        """Verify read layouts.

        Args:
            pxl_view: pxl view.
            layout_dataframe: layout dataframe.
        """
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
        """Verify read layouts add marker counts.

        Args:
            snapshot: snapshot.
            pxl_view: pxl view.
        """
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
        """Verify read layouts filter.

        Args:
            pxl_view: pxl view.
            layout_dataframe: layout dataframe.
        """
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
        """Verify read layouts len.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.layouts_len_query(None))
        assert result == 34

    def test_read_layouts_len_filter(self, pxl_view):
        """Verify read layouts len filter.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        components = ["fc07dea9b679aca7"]
        with pxl_view.open() as session:
            result = session.execute_scalar(builder.layouts_len_query(components))
        assert result == 11


class TestPixelDataViewerSession:
    """Represent test pixel data viewer session."""

    def test_session_from_source_tuples_matches_viewer_open_scalar(
        self, pxl_file, pxl_view
    ):
        """Verify session from source tuples matches viewer open scalar.

        Args:
            pxl_file: pxl file.
            pxl_view: pxl view.
        """
        sample = "test_sample"
        sources = [(sample, Path(pxl_file), _expected_normalized_db_name(sample))]
        builder = QueryBuilder()
        with PixelDataViewerSession(sources) as from_sources:
            with pxl_view.open() as from_viewer:
                assert from_sources.execute_scalar(
                    builder.edgelist_len_query(None)
                ) == from_viewer.execute_scalar(builder.edgelist_len_query(None))

    def test_session_from_source_tuples_matches_viewer_open_eager(
        self, pxl_file, pxl_view
    ):
        """Verify session from source tuples matches viewer open eager.

        Args:
            pxl_file: pxl file.
            pxl_view: pxl view.
        """
        sample = "test_sample"
        sources = [(sample, Path(pxl_file), _expected_normalized_db_name(sample))]
        builder = QueryBuilder()
        with PixelDataViewerSession(sources) as from_sources:
            with pxl_view.open() as from_viewer:
                assert_frame_equal(
                    from_sources.execute_eager(builder.edgelist_query(None)),
                    from_viewer.execute_eager(builder.edgelist_query(None)),
                )


class TestPixelDataViewerSessionSqlInjection:
    """Malicious-looking session inputs are rejected when the session is built."""

    def test_rejects_sample_name_with_quote_and_comment_payload(self, pxl_file):
        """Verify rejects sample name with quote and comment payload.

        Args:
            pxl_file: pxl file.
        """
        with pytest.raises(ValueError, match="sample name"):
            PixelDataViewerSession([("O'Brien' OR 1=1 --", Path(pxl_file), "db_safe")])

    def test_rejects_db_alias_with_semicolon_and_quotes(self, pxl_file):
        """Verify rejects db alias with semicolon and quotes.

        Args:
            pxl_file: pxl file.
        """
        with pytest.raises(ValueError, match="database alias"):
            PixelDataViewerSession(
                [("clean_sample", Path(pxl_file), 'db_evil"; SELECT 1 -- x')]
            )

    def test_rejects_path_with_apostrophe_in_filename(self, pxl_file, tmp_path):
        """Verify rejects path with apostrophe in filename.

        Args:
            pxl_file: pxl file.
            tmp_path: tmp path.
        """
        tricky_path = tmp_path / "evil'name.pxl"
        shutil.copy2(pxl_file, tricky_path)
        with pytest.raises(ValueError, match="PXL path"):
            PixelDataViewerSession([("s", tricky_path, "db_ok")])

    def test_rejects_db_alias_with_double_quote(self, pxl_file):
        """Verify rejects db alias with double quote.

        Args:
            pxl_file: pxl file.
        """
        with pytest.raises(ValueError, match="database alias"):
            PixelDataViewerSession([("s", Path(pxl_file), 'db_with_"quote')])

    def test_accepts_safe_sample_db_and_path(self, pxl_file):
        """Verify accepts safe sample db and path.

        Args:
            pxl_file: pxl file.
        """
        sources = [("test_sample", Path(pxl_file), "db_test_sample")]
        builder = QueryBuilder()
        with PixelDataViewerSession(sources) as session:
            assert session.execute_scalar(builder.edgelist_len_query(None)) == 57


class TestPixelDataViewer:
    """Represent test pixel data viewer."""

    def test_sample_names(self, pxl_view: PixelDataViewer):
        """Verify sample names.

        Args:
            pxl_view: Pxl view.
        """
        assert pxl_view.sample_names() == ["test_sample"]

    def test_view_has_all_tables(self, pxl_view):
        """Verify view has all tables.

        Args:
            pxl_view: pxl view.
        """
        with pxl_view.open() as session:
            result = session.execute_eager(Query("SHOW ALL TABLES", {}))

        assert_frame_equal(
            result.filter(pl.col("database") == "memory").select("name").sort("name"),
            pl.DataFrame(
                {"name": ["edgelist", "proximity", "layouts", "metadata"]}
            ).sort("name"),
        )

    def test_open_returns_open_session(self, pxl_view):
        """Verify open returns open session.

        Args:
            pxl_view: pxl view.
        """
        session = pxl_view.open()
        assert isinstance(session, PixelDataViewerSession)
        builder = QueryBuilder()
        lazy = session.execute_lazy(builder.edgelist_query(None))
        assert isinstance(lazy, pl.LazyFrame)
        _ = lazy.collect()
        session.close()

    def test_open_context_manager_same_as_manual_close(self, pxl_view):
        """Verify open context manager same as manual close.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        with pxl_view.open() as session:
            lazy = session.execute_lazy(builder.edgelist_query(None))
            assert isinstance(lazy, pl.LazyFrame)
            _ = lazy.collect()

    def test_nested_open_sessions_are_independent(self, pxl_view):
        """Verify nested open sessions are independent.

        Args:
            pxl_view: pxl view.
        """
        builder = QueryBuilder()
        with pxl_view.open() as outer:
            outer_count = outer.execute_scalar(builder.edgelist_len_query(None))
            with pxl_view.open() as inner:
                inner_count = inner.execute_scalar(builder.edgelist_len_query(None))
            assert outer.execute_scalar(builder.edgelist_len_query(None)) == outer_count
        assert inner_count == outer_count == 57

    def test_execute_after_close_raises(self, pxl_view):
        """Verify execute after close raises.

        Args:
            pxl_view: pxl view.
        """
        session = pxl_view.open()
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.execute_eager(Query("SELECT 1", {}))

    def test_close_idempotent(self, pxl_view):
        """Verify close idempotent.

        Args:
            pxl_view: pxl view.
        """
        session = pxl_view.open()
        session.close()
        session.close()
