"""Copyright © 2024 Pixelgen Technologies AB."""

from io import StringIO

import polars as pl
import pytest

from pixelator.pna.graph import PNAGraph

EDGELIST_DATA = """umi1,umi2,count,uei,marker_1,marker_2,component
16718381540940362211,4765690112321800547,10,2210194,MarkerC,MarkerB,fc07dea9b679aca7
16718381540940362211,13496407243000087834,10,54146122,MarkerC,MarkerB,fc07dea9b679aca7
16718381540940362211,3120322361086630706,10,8039964,MarkerC,MarkerA,fc07dea9b679aca7
16718381540940362211,3381206478569230700,10,40323847,MarkerC,MarkerA,fc07dea9b679aca7
16718381540940362211,1906719288004785482,10,12428327,MarkerC,MarkerC,fc07dea9b679aca7
16718381540940362211,2986818696398050493,10,828430,MarkerC,MarkerB,fc07dea9b679aca7
16718381540940362211,11449696640771709155,10,67062441,MarkerC,MarkerB,fc07dea9b679aca7
16718381540940362211,10575684261142813337,10,52561776,MarkerC,MarkerB,fc07dea9b679aca7
16718381540940362211,11601060255102815255,10,64718951,MarkerC,MarkerC,fc07dea9b679aca7
16718381540940362211,5412574254597463076,10,25729979,MarkerC,MarkerC,fc07dea9b679aca7
4765690112321800547,13496407243000087834,10,76405491,MarkerB,MarkerB,fc07dea9b679aca7
4765690112321800547,3120322361086630706,10,38367755,MarkerB,MarkerA,fc07dea9b679aca7
4765690112321800547,3381206478569230700,10,46092159,MarkerB,MarkerA,fc07dea9b679aca7
4765690112321800547,11449696640771709155,10,99720993,MarkerB,MarkerB,fc07dea9b679aca7
4765690112321800547,10575684261142813337,10,80498917,MarkerB,MarkerB,fc07dea9b679aca7
4765690112321800547,11601060255102815255,10,98083533,MarkerB,MarkerC,fc07dea9b679aca7
4765690112321800547,5412574254597463076,10,37952334,MarkerB,MarkerC,fc07dea9b679aca7
13496407243000087834,3120322361086630706,10,95010031,MarkerB,MarkerA,fc07dea9b679aca7
13496407243000087834,3381206478569230700,10,65045927,MarkerB,MarkerA,fc07dea9b679aca7
13496407243000087834,11449696640771709155,10,38892142,MarkerB,MarkerB,fc07dea9b679aca7
3120322361086630706,3381206478569230700,10,84548057,MarkerA,MarkerA,fc07dea9b679aca7
3120322361086630706,2986818696398050493,10,52535432,MarkerA,MarkerB,fc07dea9b679aca7
3120322361086630706,11449696640771709155,10,37541664,MarkerA,MarkerB,fc07dea9b679aca7
3093013882117452616,17578746049728489641,10,31024187,MarkerA,MarkerA,e7d82bca9694eea7
3093013882117452616,5828273475010330184,10,42295915,MarkerA,MarkerB,e7d82bca9694eea7
4633887202036215820,17578746049728489641,10,48583535,MarkerA,MarkerA,e7d82bca9694eea7
4633887202036215820,5828273475010330184,10,71882171,MarkerA,MarkerB,e7d82bca9694eea7
4633887202036215820,6367193580528650492,10,88948783,MarkerA,MarkerB,e7d82bca9694eea7
17578746049728489641,6367193580528650492,10,93404351,MarkerA,MarkerB,e7d82bca9694eea7
12237051952843152705,6006701602935914176,10,53134349,MarkerA,MarkerC,4920229146151c29
12237051952843152705,1657631243467327470,10,35779519,MarkerA,MarkerA,4920229146151c29
12237051952843152705,8754864093431251381,10,67273937,MarkerA,MarkerC,4920229146151c29
14716292412758382347,8754864093431251381,10,57152983,MarkerC,MarkerC,4920229146151c29
17336050916343892849,1657631243467327470,10,32186939,MarkerC,MarkerA,4920229146151c29
17336050916343892849,8754864093431251381,10,71951047,MarkerC,MarkerC,4920229146151c29
1645819806523959612,1657631243467327470,10,50441862,MarkerC,MarkerA,4920229146151c29
1645819806523959612,8754864093431251381,10,33791122,MarkerC,MarkerC,4920229146151c29
17093482914600361331,1657631243467327470,10,76065938,MarkerB,MarkerA,4920229146151c29
17093482914600361331,8754864093431251381,10,39161900,MarkerB,MarkerC,4920229146151c29
17581986263961994225,1657631243467327470,10,89027435,MarkerA,MarkerA,4920229146151c29
17581986263961994225,8754864093431251381,10,22715759,MarkerA,MarkerC,4920229146151c29
10162086509614058846,1657631243467327470,10,71417252,MarkerC,MarkerA,4920229146151c29
10162086509614058846,8754864093431251381,10,62318714,MarkerC,MarkerC,4920229146151c29
2680297880380432217,11721705052923611698,10,4854648,MarkerA,MarkerC,3770519d30f36d18
2680297880380432217,3527325133011089308,10,8401534,MarkerA,MarkerA,3770519d30f36d18
9914041027802204156,11721705052923611698,10,78709830,MarkerB,MarkerC,3770519d30f36d18
9914041027802204156,3527325133011089308,10,31646617,MarkerB,MarkerA,3770519d30f36d18
9914041027802204156,4489093624312168814,10,23936944,MarkerB,MarkerA,3770519d30f36d18
11721705052923611698,4489093624312168814,10,79168886,MarkerC,MarkerA,3770519d30f36d18
4419594517623728635,10605176534183593997,10,87648423,MarkerB,MarkerC,4920229146151c29
4419594517623728635,8754864093431251381,10,7929971,MarkerB,MarkerC,4920229146151c29
17533628619506136151,4489093624312168814,10,67126331,MarkerC,MarkerA,3770519d30f36d18
10605176534183593997,1657631243467327470,10,57364873,MarkerC,MarkerA,4920229146151c29
10605176534183593997,8754864093431251381,10,15027946,MarkerC,MarkerC,4920229146151c29
6006701602935914176,1657631243467327470,10,86004090,MarkerC,MarkerA,4920229146151c29
6006701602935914176,8754864093431251381,10,45033936,MarkerC,MarkerC,4920229146151c29
1657631243467327470,8754864093431251381,10,23064220,MarkerA,MarkerC,4920229146151c29
"""


def test_graph_from_edgelist():
    input_data = pl.read_csv(
        StringIO(EDGELIST_DATA),
        has_header=True,
        schema={
            "umi1": pl.UInt64,
            "umi2": pl.UInt64,
            "read_count": pl.UInt64,
            "uei_count": pl.UInt64,
            "marker_1": pl.String,
            "marker_2": pl.String,
            "component": pl.String,
        },
    ).lazy()

    result = PNAGraph.from_edgelist(input_data)

    assert dict(result.raw.nodes(data=True)) == {
        2680297880380432217: {
            "marker": "MarkerA",
            "pixel_type": "A",
            "read_count": 20,
            "name": 2680297880380432217,
        },
        16718381540940362211: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 100,
            "name": 16718381540940362211,
        },
        3093013882117452616: {
            "marker": "MarkerA",
            "pixel_type": "A",
            "read_count": 20,
            "name": 3093013882117452616,
        },
        4419594517623728635: {
            "marker": "MarkerB",
            "pixel_type": "A",
            "read_count": 20,
            "name": 4419594517623728635,
        },
        17533628619506136151: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 10,
            "name": 17533628619506136151,
        },
        12237051952843152705: {
            "marker": "MarkerA",
            "pixel_type": "A",
            "read_count": 30,
            "name": 12237051952843152705,
        },
        3120322361086630706: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 60,
            "name": 3120322361086630706,
        },
        14716292412758382347: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 10,
            "name": 14716292412758382347,
        },
        4633887202036215820: {
            "marker": "MarkerA",
            "pixel_type": "A",
            "read_count": 30,
            "name": 4633887202036215820,
        },
        13496407243000087834: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 50,
            "name": 13496407243000087834,
        },
        1645819806523959612: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 20,
            "name": 1645819806523959612,
        },
        6006701602935914176: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 30,
            "name": 6006701602935914176,
        },
        9914041027802204156: {
            "marker": "MarkerB",
            "pixel_type": "A",
            "read_count": 30,
            "name": 9914041027802204156,
        },
        10162086509614058846: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 20,
            "name": 10162086509614058846,
        },
        17336050916343892849: {
            "marker": "MarkerC",
            "pixel_type": "A",
            "read_count": 20,
            "name": 17336050916343892849,
        },
        17581986263961994225: {
            "marker": "MarkerA",
            "pixel_type": "A",
            "read_count": 20,
            "name": 17581986263961994225,
        },
        17093482914600361331: {
            "marker": "MarkerB",
            "pixel_type": "A",
            "read_count": 20,
            "name": 17093482914600361331,
        },
        17578746049728489641: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 30,
            "name": 17578746049728489641,
        },
        1657631243467327470: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 90,
            "name": 1657631243467327470,
        },
        4765690112321800547: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 80,
            "name": 4765690112321800547,
        },
        10605176534183593997: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 30,
            "name": 10605176534183593997,
        },
        11721705052923611698: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 30,
            "name": 11721705052923611698,
        },
        11449696640771709155: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 40,
            "name": 11449696640771709155,
        },
        3527325133011089308: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 20,
            "name": 3527325133011089308,
        },
        4489093624312168814: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 30,
            "name": 4489093624312168814,
        },
        2986818696398050493: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 20,
            "name": 2986818696398050493,
        },
        3381206478569230700: {
            "marker": "MarkerA",
            "pixel_type": "B",
            "read_count": 40,
            "name": 3381206478569230700,
        },
        1906719288004785482: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 10,
            "name": 1906719288004785482,
        },
        5412574254597463076: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 20,
            "name": 5412574254597463076,
        },
        5828273475010330184: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 20,
            "name": 5828273475010330184,
        },
        8754864093431251381: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 110,
            "name": 8754864093431251381,
        },
        6367193580528650492: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 20,
            "name": 6367193580528650492,
        },
        10575684261142813337: {
            "marker": "MarkerB",
            "pixel_type": "B",
            "read_count": 20,
            "name": 10575684261142813337,
        },
        11601060255102815255: {
            "marker": "MarkerC",
            "pixel_type": "B",
            "read_count": 20,
            "name": 11601060255102815255,
        },
    }


@pytest.fixture
def graph():
    input_data = pl.read_csv(
        StringIO(EDGELIST_DATA),
        has_header=True,
        schema={
            "umi1": pl.UInt64,
            "umi2": pl.UInt64,
            "read_count": pl.UInt64,
            "uei_count": pl.UInt64,
            "marker_1": pl.String,
            "marker_2": pl.String,
            "component": pl.String,
        },
    ).lazy()
    return PNAGraph.from_edgelist(input_data)


def test_graph_can_get_marker_counts(graph):
    result = graph.node_marker_counts
    assert set(result.columns) == {"MarkerC", "MarkerB", "MarkerA"}
    assert result.shape == (34, 3)


def test_graph_can_get_local_g(graph):
    result = graph.local_g(k=1)
    assert set(result.columns) == {"MarkerC", "MarkerB", "MarkerA"}
    assert result.shape == (34, 3)
