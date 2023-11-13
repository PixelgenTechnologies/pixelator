"""Tests verifying that the graph protocols are implemented.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import igraph as ig
import networkx as nx
import pytest
from pixelator.graph.backends.implementations import (
    IgraphBasedEdge,
    IgraphBasedEdgeSequence,
    IgraphBasedVertex,
    IgraphBasedVertexClustering,
    IgraphBasedVertexSequence,
    NetworkxBasedEdge,
    NetworkxBasedEdgeSequence,
    NetworkxBasedVertex,
    NetworkxBasedVertexClustering,
    NetworkxBasedVertexSequence,
)


@pytest.fixture
def ig_graph():
    yield ig.Graph()


@pytest.fixture
def nx_graph():
    yield nx.Graph()


@pytest.fixture
def ig_vertex(ig_graph):
    yield IgraphBasedVertex(ig_graph.add_vertex(my_attr="a"))


@pytest.fixture
def nx_vertex():
    yield NetworkxBasedVertex(index=0, data={"my_attr": "a"})


@pytest.mark.parametrize("vertex", ["ig_vertex", "nx_vertex"])
class TestVertexClassesImplementVertexProtocol:
    def test_index(self, vertex, request):
        vertex_inst = request.getfixturevalue(vertex)
        assert vertex_inst.index == 0

    def test__getitem__(self, vertex, request):
        vertex_instance = request.getfixturevalue(vertex)
        assert vertex_instance["my_attr"] == "a"


@pytest.fixture
def ig_edge(ig_graph):
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    yield IgraphBasedEdge(ig_graph.add_edge(0, 1, my_attr="a"))


@pytest.fixture
def nx_edge(nx_graph):
    nx_graph.add_edge(0, 1, my_attr="a", index=0)
    yield NetworkxBasedEdge(list(nx_graph.edges(data=True))[0], nx_graph)


@pytest.mark.parametrize("edge", ["ig_edge", "nx_edge"])
class TestEdgeClassesImplementEdgeProtocol:
    def test_index(self, edge, request):
        edge = request.getfixturevalue(edge)
        assert edge.index == 0

    def test_vertex_tuple(self, edge, request):
        edge = request.getfixturevalue(edge)
        assert tuple(map(lambda x: x.index, edge.vertex_tuple)) == (0, 1)


@pytest.fixture
def ig_vertex_seq(ig_graph):
    ig_graph.add_vertex(my_attr="a", other_attr=1)
    ig_graph.add_vertex(my_attr="b", other_attr=2)
    yield IgraphBasedVertexSequence(ig_graph.vs, ig_graph)


@pytest.fixture
def nx_vertex_seq():
    yield NetworkxBasedVertexSequence(
        [
            NetworkxBasedVertex(0, {"my_attr": "a", "other_attr": 1}),
            NetworkxBasedVertex(1, {"my_attr": "b", "other_attr": 2}),
        ]
    )


@pytest.mark.parametrize("vertex_seq", ["ig_vertex_seq", "nx_vertex_seq"])
class TestVertexSequenceClassesImplementVertexSequenceProtocol:
    def test_vertices(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert list(map(lambda x: x.index, vertex_seq.vertices())) == [0, 1]

    def test__len__(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert len(vertex_seq) == 2

    def test__iter__(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        iterator = iter(vertex_seq)
        assert next(iterator).index == 0

    def test_attributes(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert vertex_seq.attributes() == {"my_attr", "other_attr"}

    def test_get_vertex(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        vertex = vertex_seq.get_vertex(0)
        assert vertex.index == 0
        assert vertex["my_attr"] == "a"

    def test_get_attributes(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert list(vertex_seq.get_attribute("my_attr")) == ["a", "b"]
        assert list(vertex_seq.get_attribute("other_attr")) == [1, 2]


@pytest.fixture
def ig_edge_seq(ig_graph):
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_edge(0, 1, my_attr="a")
    ig_graph.add_edge(1, 2, my_attr="b")
    ig_graph.add_edge(2, 0, my_attr="a")
    ig_graph.add_edge(2, 3, my_attr="c")
    yield IgraphBasedEdgeSequence(ig_graph.es)


@pytest.fixture
def nx_edge_seq(nx_graph):
    nx_graph.add_edge(0, 1, my_attr="a", index=0)
    nx_graph.add_edge(1, 2, my_attr="b", index=1)
    nx_graph.add_edge(2, 0, my_attr="a", index=2)
    nx_graph.add_edge(2, 3, my_attr="c", index=3)

    yield NetworkxBasedEdgeSequence(
        nx_graph,
        [NetworkxBasedEdge(edge, nx_graph) for edge in nx_graph.edges(data=True)],
    )


@pytest.mark.parametrize("edge_seq", ["ig_edge_seq", "nx_edge_seq"])
class TestEdgeSequenceClassesImplementEdgeSequenceProtocol:
    def test__len__(self, edge_seq, request):
        edge_seq = request.getfixturevalue(edge_seq)
        assert len(edge_seq) == 4

    def test__iter__(self, edge_seq, request):
        edge_seq = request.getfixturevalue(edge_seq)
        iterator = iter(edge_seq)
        assert {e.index for e in iterator} == {0, 1, 2, 3}

    def test_select_where(self, edge_seq, request):
        edge_seq = request.getfixturevalue(edge_seq)
        edges = edge_seq.select_where("my_attr", "a")
        assert {e.index for e in edges} == {0, 2}

    def test_select_within(self, edge_seq, request):
        edge_seq = request.getfixturevalue(edge_seq)
        edges = edge_seq.select_within({1, 2})
        assert [e.index for e in edges] == [1]


@pytest.fixture
def ig_vertex_clustering(ig_graph):
    # cluster 1
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_edge(0, 1, my_attr="a")
    ig_graph.add_edge(1, 2, my_attr="b")
    ig_graph.add_edge(2, 0, my_attr="a")

    # cluster 2
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_vertex()
    ig_graph.add_edge(3, 4, my_attr="c")
    ig_graph.add_edge(4, 5, my_attr="c")

    # Edge connecting the clusters to get a crossing edge
    ig_graph.add_edge(0, 5, my_attr="c", crossing_edge=True)

    yield IgraphBasedVertexClustering(
        ig.VertexClustering(graph=ig_graph, membership=[0, 0, 0, 1, 1, 1]), ig_graph
    )


@pytest.fixture
def nx_vertex_clustering(nx_graph):
    # cluster 1
    nx_graph.add_edge(0, 1, my_attr="a", index=0)
    nx_graph.add_edge(1, 2, my_attr="b", index=1)
    nx_graph.add_edge(2, 0, my_attr="a", index=2)

    # cluster 2
    nx_graph.add_edge(3, 4, my_attr="c", index=4)
    nx_graph.add_edge(4, 5, my_attr="c", index=5)

    # Edge connecting the clusters to get a crossing edge
    nx_graph.add_edge(0, 5, my_attr="c", index=6, crossing_edge=True)

    yield NetworkxBasedVertexClustering(nx_graph, [{0, 1, 2}, {3, 4, 5}])


@pytest.mark.parametrize(
    "vertex_clustering", ["ig_vertex_clustering", "nx_vertex_clustering"]
)
class TestVertexClusteringClassesImplementVertexClusteringProtocol:
    def test__len__(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        assert len(vertex_clustering) == 2

    def test__iter__(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        iterator = iter(vertex_clustering)
        assert list(map(lambda v: v.index, next(iterator))) == [0, 1, 2]

    def test_modularity(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        assert pytest.approx(vertex_clustering.modularity, abs=0.01) == 0.3194

    def test_crossing(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        crossing_edges = vertex_clustering.crossing()
        assert len(crossing_edges) == 1

    def test_giant(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        assert len(vertex_clustering.giant().vs) == 3
        assert len(vertex_clustering.giant().es) == 3

    def test_subgraphs(self, vertex_clustering, request):
        vertex_clustering = request.getfixturevalue(vertex_clustering)
        assert len(vertex_clustering.subgraphs()) == 2
