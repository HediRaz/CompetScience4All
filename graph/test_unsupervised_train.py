import graph.graph_utils
import graph.load_data

import numpy as np


def test_loader_init_embeddings():
    loader = graph.load_data.unsupervised_dataLoader([2011, 2012, 2013, 2014])

    f = loader.init_nodes_embedding('eigen', 10)
    assert f.shape[0] == 64719
    assert f.shape[1] == 40

    f = loader.init_nodes_embedding('eigen2017', 10, True)
    assert f.shape[0] == 64719
    assert f.shape[1] == 10

    print(np.min(f.numpy(), 0).shape)
    print(f[:10])


def test_loader_generator():
    loader = graph.load_data.unsupervised_dataLoader([2011, 2012, 2013, 2014])

    generator = loader.Generator(iter_number=10, max_neighbors=1, starting_size=1, k=0, add_diag=True)

    sparse_adj_tensor, degs12, vertices, starting_size, graph_i = next(generator)
    assert sparse_adj_tensor.shape == (1, 1)
    assert degs12.shape == (1, 1)
    assert isinstance(vertices, list)
    assert len(vertices) == 1
    assert starting_size == 1


if __name__ == '__main__':
    # test_loader_init_embeddings()
    test_loader_generator()
