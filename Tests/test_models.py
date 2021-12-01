import torch
from graph.gcn import GCN, GIN, GAN
from graph.classifier import Classifier_Dense, Classifier_RNN


def test_gnn(gnn_type):
    batch_size = 8
    embedding_size = 16
    dims = [embedding_size, 32, 64]
    x = torch.ones((batch_size, embedding_size))
    indices = torch.tensor([[0, 3, 4, 5, 6, 5, 1, 4, 2, 4], [6, 7, 3, 4, 2, 3, 4, 5, 7, 1]], dtype=torch.int64)
    values = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    m = torch.sparse_coo_tensor(indices, values, (batch_size, batch_size))

    gnn = gnn_type(dims)
    gnn.device = 'cpu'

    y = gnn(m, x)
    assert y.shape[0] == batch_size
    assert y.shape[1] == dims[-1]


if __name__ == "__main__":
    test_gnn(GAN)