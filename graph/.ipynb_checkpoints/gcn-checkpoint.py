import torch
import graph

class DenseBlock(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(DenseBlock, self).__init__()
        self.dense = torch.nn.Linear(n_in, n_out)
        self.acti = torch.nn.Tanh()
        # self.bn = torch.nn.BatchNorm1d(n_out)
        # self.drop = torch.nn.Dropout(0.1)

    def forward(self, x):
        y = self.dense(x)
        # y = self.bn(y)
        y = self.acti(y)
        return y

class MultipleDenseBlock(torch.nn.Module):
    def __init__(self, n, n_in, n_out):
        super(MultipleDenseBlock, self).__init__()
        self.blocks = torch.nn.ModuleList([])
        self.blocks.append(DenseBlock(n_in, n_out))
        for _ in range(n-1):
            self.blocks.append(DenseBlock(n_out, n_out))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class GCN(torch.nn.Module):

    def __init__(self, dims, depth_dense_block):
        super(GCN, self).__init__()
        self.depth = len(dims) -1
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(MultipleDenseBlock(depth_dense_block, dims[i], dims[i+1]))

    def forward(self, adjacency_matrix, x):
        for i in range(self.depth):
            x = self.layers[i](torch.sparse.mm(adjacency_matrix, x))
            x = x / torch.std(x, -1, keepdim=True)
        return x

def l2_normalization(x):
    return x / torch.sqrt(torch.sum(torch.square(x), 1, keepdim=True))

class GIN(torch.nn.Module):

    def __init__(self, dims, depth_dense_block=3):
        super(GIN, self).__init__()

        self.depth = len(dims) -1
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(MultipleDenseBlock(depth_dense_block, dims[i], dims[i+1]))
            self.layers.append(MultipleDenseBlock(depth_dense_block, dims[i], dims[i+1]))
            self.layers.append(MultipleDenseBlock(depth_dense_block, 2*dims[i+1], dims[i+1]))

    def forward(self, adjacency_matrix, x):
        for i in range(self.depth):
            x = self.layers[3 * i + 2](torch.cat((self.layers[3 * i + 1](x), torch.sparse.mm(adjacency_matrix, self.layers[3 * i + 0](x))), -1))
            x = l2_normalization(x)
        return x

class GIN_GRU(torch.nn.Module):

    def __init__(self, dims, depth_dense_block=3):
        super(GIN_GRU, self).__init__()

        self.depth = len(dims) -1
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(MultipleDenseBlock(depth_dense_block, dims[0], dims[0]))
            self.layers.append(MultipleDenseBlock(depth_dense_block, dims[0], dims[0]))
            self.layers.append(MultipleDenseBlock(depth_dense_block, 2*dims[0], dims[0]))

        self.gru = torch.nn.GRU(
            input_size=dims[0],
            hidden_size=dims[0],
            num_layers=1,
            batch_first=True
        )

    def forward(self, adjacency_matrix, x):
        x = self.layers[2](torch.cat((self.layers[1](x), torch.sparse.mm(adjacency_matrix, self.layers[0](x))), -1))
        x = torch.unsqueeze(x, 1)
        x, hn = self.gru(x)
        x = torch.squeeze(x)
        x = x / torch.std(x, -1, keepdim=True)
        for i in range(1, self.depth):
            x = self.layers[3 * i + 2](torch.cat((self.layers[3 * i + 1](x), torch.sparse.mm(adjacency_matrix, self.layers[3 * i + 0](x))), -1))
            x = torch.unsqueeze(x, 1)
            x, hn = self.gru(x, hn)
            x = torch.squeeze(x)
            x = x / torch.std(x, -1, keepdim=True)
        return x

def agg_mean(m, h):
    return torch.sparse.mm(m, h)

def agg_std(m, h, mu=None):
    if mu is None:
        mu = agg_mean(m, h)
    return torch.sparse.mm(torch.square(h-mu))

def agg_min(adj_list, h):
    res = (torch.min(h[torch.tensor(adj_list[u], dtype=torch.int64, device=graph.device)], 0) for u in range(graph.nVertices))
    return torch.cat(res, 0)

def agg_max(adj_list, h):
    res = (torch.max(h[torch.tensor(adj_list[u], dtype=torch.int64, device=graph.device)], 0) for u in range(graph.nVertices))
    return torch.cat(res, 0)

def Samp(d, alpha):
    delta = torch.mean(torch.log(d+1))
    return torch.log(d+1) / delta

class PNA(torch.nn.Module):

    def __init__(self, dims, degrees, adj_list, depth_dense_block=3):
        super(PNA, self).__init__()
        self.aggregators = [
            agg_mean,
            agg_std,
            agg_min,
            agg_max
        ]
        self.d = degrees
        self.adj_list = adj_list


    def forward(self, adjacency_matrix, adj_list, x):
        for i in range(self.depth):
            x = self.layers[4 * i + 2](torch.cat((self.layers[4 * i + 1](x), torch.sparse.mm(adjacency_matrix, self.layers[4 * i + 0](x))), -1))
            x = x / torch.std(x, -1, keepdim=True)
        return x


class GIN_shared_weights(torch.nn.Module):

    def __init__(self, dims, depth_dense_block=3):
        super(GIN_shared_weights, self).__init__()

        self.depth = len(dims) -1
        self.b1 = MultipleDenseBlock(depth_dense_block, 2*dims[0], dims[0])
        self.b2 = MultipleDenseBlock(depth_dense_block, dims[0], dims[0])
        self.b3 = MultipleDenseBlock(depth_dense_block, dims[0], dims[0])

    def forward(self, adjacency_matrix, x):
        for _ in range(self.depth):
            x = self.b1(torch.cat((self.b2(x), self.b3(torch.sparse.mm(adjacency_matrix, x))), -1))
            x = x / torch.std(x, -1, keepdim=True)
        return x


class GAT(torch.nn.Module):

    def __init__(self, dims):
        super(GAT, self).__init__()
        self.device = torch.device("cuda:0")

        self.l11 = DenseBlock(dims[0], dims[1])  # dims[0] = dim des features des noeuds
        self.l12 = DenseBlock(dims[1], 1)
        self.l13 = DenseBlock(dims[0], dims[1])

        self.l21 = DenseBlock(dims[1], dims[2])
        self.l22 = DenseBlock(dims[2], 1)
        self.l23 = DenseBlock(dims[1], dims[2])

    def forward(self, adj_mat, h):
        # values = features des edges
        # values = torch.sum(self.l11(h)[adj_mat._indices()], -2)
        # indices = adj_mat._indices()
        # print(self.l11(h).shape)
        # print(self.l11(h)[adj_mat._indices()].shape)
        # print(torch.sum(self.l11(h)[adj_mat._indices()], 0).shape)
        # print(self.l12(torch.sum(self.l11(h)[adj_mat._indices()], -2)).shape)

        # New adjacency matrix
        adj_mat = torch.sparse.softmax(
            torch.sparse_coo_tensor(
                adj_mat._indices(),
                torch.squeeze(self.l12(torch.sum(self.l11(h)[adj_mat._indices()], 0))),
                adj_mat.shape,
                dtype=torch.float32,
                device=self.device
                ),
            1)
        h = torch.sparse.mm(adj_mat, self.l13(h))
        h = h / torch.std(h, -1, keepdim=True)

        adj_mat = torch.sparse.softmax(
            torch.sparse_coo_tensor(
                adj_mat._indices(),
                torch.squeeze(self.l22(torch.sum(self.l21(h)[adj_mat._indices()], 0))),
                (adj_mat.shape[0], adj_mat.shape[1]),
                dtype=torch.float32,
                device=self.device
                ),
            1)
        h = torch.sparse.mm(adj_mat, self.l23(h))
        h = h / torch.std(h, -1, keepdim=True)

        return h
