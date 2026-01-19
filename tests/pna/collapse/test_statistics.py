"""Copyright © 2025 Pixelgen Technologies AB."""

from io import StringIO

import polars as pl
import pytest

from pixelator.pna.collapse.paired.statistics import CollapseSummaryStatistics


@pytest.fixture
def collapsed_lz_df() -> pl.LazyFrame:
    data = StringIO("""
        marker_1,marker_2,umi1,umi2,read_count,uei_count,corrected_read_count
        CD29,CD43,555638967335511,28226901571890329,6,2,0
        CD29,CD43,555638967335511,14976911110869850,2,3,0
        CD29,CD43,555638967335511,10588034069358815,2,2,0
        B2M,CD44,1592993435260922,33899519959851077,3,1,0
        B2M,CD5,1772702303642965,23228591926107523,6,4,1
        CD45,CD45,1905030979812053,14976911110869850,3,2,0
        CD82,CD82,1927887423101473,8679240914900276,1,1,0
        CD11a,CD43,2083036410634253,10588034069358815,2,1,0
        CD102,B2M,2092819761711040,43487857778022885,1,1,0
        B2M,CD5,3341233760855701,65064780908590549,3,3,0""")
    return pl.read_csv(data).lazy()


def test_collapse_summary_statistics_from_lazy_frame(
    collapsed_lz_df: pl.LazyFrame,
):
    summary_stats = CollapseSummaryStatistics.from_lazy_frame(collapsed_lz_df)

    assert summary_stats.degree_distribution == {1: 13, 2: 2, 3: 1}
    assert summary_stats.read_counts_stats.mean == 2.9
    assert summary_stats.uei_stats.mean == 2.0
