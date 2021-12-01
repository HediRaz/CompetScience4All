import sys
import os
sys.path.append(os.path.realpath(os.curdir))
import graph
from graph.load_data import unsupervised_dataLoader
from graph import graph_utils
from graph import load_data

import numpy as np


def test_loader_unsupervised_nodes_embedding():
    years = [2000, 2001, 2002]
    loader = unsupervised_dataLoader(years, 4)
    d = 4
    for method in ["random", "eigen", "eigen2017"]:
        for normalize in [True, False]:
            f = loader.init_nodes_embedding(method=method, initial_embedding_dim=d, normalize=normalize)
            assert f.shape == (len(years), graph.nVertices, d)


def test_loader_unsupervised_generator():
    loader = load_data.unsupervised_dataLoader([2011, 2012, 2013, 2014])

    generator = loader.Generator(iter_number=10, max_neighbors=1, starting_size=1, k=0, add_diag=True)

    sparse_adj_tensor, degs12, vertices, starting_size, graph_i = next(generator)
    assert sparse_adj_tensor.shape == (1, 1)
    assert degs12.shape == (1, 1)
    assert isinstance(vertices, list)
    assert len(vertices) == 1
    assert starting_size == 1


if __name__ == '__main__':
    test_loader_unsupervised_nodes_embedding()
