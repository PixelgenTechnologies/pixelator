"""Tests verifying that the graph protocols are implemented.

Copyright © 2023 Pixelgen Technologies AB.
"""

import networkx as nx
import pytest

from pixelator.graph.backends.implementations import (
    graph_backend,
    graph_backend_from_graph_type,
)
from pixelator.graph.backends.implementations._networkx import (
    NetworkxBasedEdge,
    NetworkxBasedEdgeSequence,
    NetworkxBasedVertex,
    NetworkxBasedVertexClustering,
    NetworkxBasedVertexSequence,
    NetworkXGraphBackend,
)


def test_graph_backend_request_networkx():
    result = graph_backend("NetworkXGraphBackend")
    assert isinstance(result(), NetworkXGraphBackend)


@pytest.mark.parametrize("enable_backend", ["networkx"], indirect=True)
def test_graph_backend_request_networkx_when_env_var_set(enable_backend):
    result = graph_backend()
    assert isinstance(result(), NetworkXGraphBackend)


def test_graph_backend_from_graph_type_networkx():
    result = graph_backend_from_graph_type(graph=nx.Graph())
    assert isinstance(result(), NetworkXGraphBackend)

    result = graph_backend_from_graph_type(graph=nx.MultiGraph())
    assert isinstance(result(), NetworkXGraphBackend)


def test_graph_backend_from_graph_type_unknown():
    with pytest.raises(ValueError):
        graph_backend_from_graph_type(graph="hello")


@pytest.fixture
def nx_graph():
    yield nx.Graph()


@pytest.fixture
def nx_vertex(nx_graph):
    nx_graph.add_node(0, my_attr="a")
    yield NetworkxBasedVertex(*list(nx_graph.nodes(data=True))[0], graph=nx_graph)


@pytest.mark.parametrize("vertex", ["nx_vertex"])
class TestVertexClassesImplementVertexProtocol:
    def test_index(self, vertex, request):
        vertex_inst = request.getfixturevalue(vertex)
        assert vertex_inst.index == 0

    def test__getitem__(self, vertex, request):
        vertex_instance = request.getfixturevalue(vertex)
        assert vertex_instance["my_attr"] == "a"

    def test__setitem__(self, vertex, request):
        vertex_instance = request.getfixturevalue(vertex)
        assert vertex_instance["my_attr"] == "a"
        vertex_instance["my_attr"] = "changed"
        assert vertex_instance["my_attr"] == "changed"


@pytest.fixture
def nx_edge(nx_graph):
    nx_graph.add_edge(0, 1, my_attr="a", index=0)
    yield NetworkxBasedEdge(list(nx_graph.edges(data=True))[0], nx_graph)


@pytest.mark.parametrize("edge", ["nx_edge"])
class TestEdgeClassesImplementEdgeProtocol:
    def test_index(self, edge, request):
        edge = request.getfixturevalue(edge)
        assert edge.index == 0

    def test_vertex_tuple(self, edge, request):
        edge = request.getfixturevalue(edge)
        assert tuple(map(lambda x: x.index, edge.vertex_tuple)) == (0, 1)


@pytest.fixture
def nx_vertex_seq(nx_graph):
    nx_graph.add_node(0, my_attr="a")
    nx_graph.add_node(0, my_attr="a", other_attr=1)
    nx_graph.add_node(1, my_attr="b", other_attr=2)
    nx_graph.add_node(2, my_attr="n", other_attr=2)

    yield NetworkxBasedVertexSequence(
        [
            NetworkxBasedVertex(node, data, graph=nx_graph)
            for node, data in nx_graph.nodes(data=True)
        ]
    )


@pytest.mark.parametrize("vertex_seq", ["nx_vertex_seq"])
class TestVertexSequenceClassesImplementVertexSequenceProtocol:
    def test_vertices(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert list(map(lambda x: x.index, vertex_seq.vertices())) == [0, 1, 2]

    def test__len__(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert len(vertex_seq) == 3

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

        vertex = vertex_seq.get_vertex(2)
        assert vertex.index == 2
        assert vertex["my_attr"] == "n"

    def test_get_vertex_raises_key_error(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        with pytest.raises(KeyError):
            vertex_seq.get_vertex(10)

    def test_get_attributes(self, vertex_seq, request):
        vertex_seq = request.getfixturevalue(vertex_seq)
        assert list(vertex_seq.get_attribute("my_attr")) == ["a", "b", "n"]
        assert list(vertex_seq.get_attribute("other_attr")) == [1, 2, 2]


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


@pytest.mark.parametrize("edge_seq", ["nx_edge_seq"])
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


@pytest.mark.parametrize("vertex_clustering", ["nx_vertex_clustering"])
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
