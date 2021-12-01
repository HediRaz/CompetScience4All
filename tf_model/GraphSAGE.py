import tensorflow as tf
import data_utils
import time
from tqdm import tqdm
from timeit import timeit
import random


class GraphSAGE():

    def __init__(self, graph_sparse, embedding_dim, num_of_vertices):

        self.graph_sparse = graph_sparse
        self.embedding_dim = embedding_dim
        self.num_of_vertices = num_of_vertices

        print('Creating adjacency list')
        self.adj_list, self.deg_list = self._init_adjList()

        print('Initializing embeddings')
        self._init_embedding()
        self.nodes_em = []
        for _ in range(num_of_vertices):
            self.nodes_em.append(
                tf.Variable(
                    tf.random.normal((embedding_dim,), 0, 1, tf.dtypes.float32),
                    trainable=True,
                    name='nodes_embedding',
                    dtype=tf.dtypes.float32
                    )
            )
        self.zero_node = tf.constant(tf.zeros((embedding_dim,), tf.dtypes.float32), dtype=tf.dtypes.float32)

        print('Initializing graph')
        self.GNN = self._init_GNN()

    def _init_embedding(self):
        self.nodes_em = tf.Variable(
            tf.random.normal((self.num_of_vertices, self.embedding_dim), 0, stddev=1, dtype=tf.dtypes.float32),
            trainable=True,
            name='nodes_embeddings',
            dtype=tf.dtypes.float32
            )

    def _init_adjList(self):
        adj_list = [dict() for _ in range(self.num_of_vertices)]
        deg_list = [0 for _ in range(self.num_of_vertices)]

        for e in tqdm(self.graph_sparse):
            if e[1] in adj_list[e[0]]:
                adj_list[e[0]][e[1]] += 1
            else:
                adj_list[e[0]][e[1]] = 1

            if e[0] in adj_list[e[1]]:
                adj_list[e[1]][e[0]] += 1
            else:
                adj_list[e[1]][e[0]] = 1

        for i in range(self.num_of_vertices):
            for n in adj_list[i].values():
                deg_list[i] += n

        return adj_list, deg_list

    def get_neighbors(self, vertex):
        return self.adj_list[vertex].keys()

    def _init_GNN(self):
        pass

    def get_graph_embedding(self, vertex, k=2):
        if k == 0:
            return self.nodes_em[vertex]

        neigh = self.get_neighbors(vertex)
        if len(neigh) == 0:
            agg = self.zero_node
        else:            
            agg = tf.math.add_n([self.get_graph_embedding(i, k-1) * self.compute_weight(vertex, i) for i in neigh])
            # agg = tf.math.add_n([self.adj_list[vertex][i]*self.get_graph_embedding(i, k-1) for i in neigh])
            # agg *= (1/self.deg_list[vertex])


        # v = tf.concat([self.nodes_em[vertex], agg], axis=0)
        v = agg
        return v
        # return self.GNN(v)

    def test_grad(self, vertex, k=2):
        with tf.GradientTape() as tape:
            tape.watch(self.nodes_em[vertex])
            v = self.get_graph_embedding(vertex, k)
        print(tape.gradient(v, self.nodes_em[vertex]))

    def compute_weight(self, v, i):
        return self.adj_list[v][i]/self.deg_list[v]

    def test_compute_weight(self):
        v = random.randint(0, self.num_of_vertices-1)
        l = list(self.adj_list[v].keys())
        if len(l) > 0:
            i = random.choice(l)
            self.compute_weight(v, i)



if __name__ == '__main__':
    g = data_utils.load_data()[0]
    graph = GraphSAGE(g, 2, 64719)
    graph.test_compute_weight()
    print(timeit(lambda :graph.test_compute_weight(), number=100))
    

    """
    top = time.time()
    print(graph.nodes_em[:2])
    print(time.time()-top)
    print('')

    top = time.time()
    print(graph.get_graph_embedding(0, 0))
    print(time.time()-top)
    print('')

    top = time.time()
    print(graph.deg_list[0])
    print(graph.get_graph_embedding(0, 1))
    print(time.time()-top)

    top = time.time()
    print(graph.deg_list[0] + sum([graph.deg_list[i] for i in graph.get_neighbors(0)]))
    print(graph.get_graph_embedding(0, 2))
    print(time.time()-top)
    """
