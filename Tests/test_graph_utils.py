import sys
import os
sys.path.append(os.path.realpath(os.curdir))

from graph.data_utils import create_graph
import graph
from graph.graph_utils import light_Graph

import numpy as np


edges_graph_test = [[2, 3], [1, 3], [2, 4]]


def test_light_graph_init():
    y = 2012
    g = light_Graph(y)

    assert g.year == y
    assert g.nVertices == graph.nVertices
    assert isinstance(g.adj_list, list)
    assert len(g.adj_list) == graph.nVertices

    u = 0
    g_true = create_graph(y)
    neigh_u = set()
    for e in g_true:
        i, j = e[0], e[1]
        if i != j:
            if i == u:
                neigh_u.add(j)
            if j == u:
                neigh_u.add(i)
    
    assert set.issubset(neigh_u, g.adj_list[u])
    assert set.issubset(g.adj_list[u], neigh_u)

    g.graph_sparse = np.array(edges_graph_test)
    g._create_graph(min_degs=2)
    assert g.adj_list[1] == {3}
    assert g.adj_list[3] == {1, 2}
    assert g.initial_nodes == [2, 3]


def test_graph_light_KHop():
    y = 2000
    g = light_Graph(y)
    g.graph_sparse = edges_graph_test
    g._create_graph(0)

    (vertices, edges, degs, global2local), n_init = g.subgraph_khop([1], 1, 1, 5)
    assert set(vertices) == {1, 2, 3}
    assert len(edges) == 4
    assert degs == [1, 2, 1]
    assert n_init == 2

    g = light_Graph(y)
    vertices_true = set()
    for e in g.graph_sparse:
        vertices_true.update(e)
    (vertices, edges, degs, global2local), n_init = g.subgraph_khop(list(vertices_true), 0, 0, 5)
    assert set(vertices) == vertices_true
    assert len(degs) == len(vertices_true) == n_init
    assert edges == g.subgraph_khop(list(vertices_true), 2, 2, 5)[0][1]


if __name__ == "__main__":
    test_light_graph_init()
    test_graph_light_KHop()