import pickle
import random
import graph
from graph.data_utils import create_graph

import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
import networkx as nx

import torch



class Graph():

    def __init__(self, year, ponderated=False):
        self.year = year
        self.nVertices = graph.nVertices
        self.graph_sparse = create_graph(year)
        csr = sparse.csr_matrix(
            (
                np.ones(len(self.graph_sparse)),
                (self.graph_sparse[:, 0], self.graph_sparse[:, 1])
            ),
            shape=(self.nVertices, self.nVertices))
        self.nx_graph = nx.from_scipy_sparse_matrix(csr)

        self._getAdjList()
        self._ponderateGraph(ponderate=ponderated)

        self.getEdgeFeatures = dict()

    def load_centralities(self):
        # Nodes Centralities:
        centralities = []
        for name in ["degree.pkl", "degrees2.pkl", "eigenvector1.pkl", "neighsConnection.pkl", "pagerank.pkl"]:
            file = open(f"Features/{self.year}/{name}", "rb")
            d = pickle.load(file)
            if isinstance(d, tuple):
                d = d[0]
            file.close()
            centralities.append([d[u] for u in range(64719)]) 
        return torch.tensor(centralities, dtype=torch.float32).t()

    def _getAdjList(self):
        adj_dict = [dict() for _ in range(self.nVertices)]

        for e in self.graph_sparse:
            if e[1] in adj_dict[e[0]]:
                adj_dict[e[0]][e[1]] += 1
            else:
                adj_dict[e[0]][e[1]] = 1

            if e[0] in adj_dict[e[1]]:
                adj_dict[e[1]][e[0]] += 1
            else:
                adj_dict[e[1]][e[0]] = 1

        self.adj_dict = adj_dict
        self.adj_sets = [set(adj_dict[i].keys()) for i in range(self.nVertices)]

    def _ponderateGraph(self, ponderate):
        graph_sparse_ponderated = []
        deg_list = [0 for _ in range(self.nVertices)]
        for u in range(self.nVertices):
            for v in self.adj_dict[u].keys():
                if not ponderate:
                    self.adj_dict[u][v] = 1
                graph_sparse_ponderated.append(np.array([u, v, self.adj_dict[u][v]]))
                deg_list[u] += self.adj_dict[u][v]
        self.graph_sparse = np.array(graph_sparse_ponderated)
        self.deg_list = np.array(deg_list, dtype=np.float)

    def KHop(self, vertices_list: list, k, max_neighbors=20):
        n1 = len(vertices_list)
        subgraph_vertices = set(vertices_list)
        n = len(vertices_list)
        vertices_indexes = dict([[vertices_list[i], i] for i in range(n)])

        subgraph_edges = set()
        for u in vertices_list:
            neighbors_u = list(self.adj_sets[u].intersection(subgraph_vertices).difference({u}))
            m = len(neighbors_u)
            subgraph_edges.update(zip(  [vertices_indexes[u]]*m,
                                        (vertices_indexes[v] for v in neighbors_u),
                                        (self.adj_dict[u][v] for v in neighbors_u)))
            subgraph_edges.update(zip(
                (vertices_indexes[v] for v in neighbors_u),
                [vertices_indexes[u]]*m,
                (self.adj_dict[v][u] for v in neighbors_u)))

        next_gen = set(vertices_list)
        for i in range(k):
            current_gen = next_gen.copy()
            next_gen = set()
            for u in current_gen:
                neighbors_set = self.adj_sets[u].difference(subgraph_vertices)  # list(self.adj_sets[u].difference({u}))
                neighbors_list = list(neighbors_set)
                sampled_neighbors = random.sample(neighbors_list, k=min(len(neighbors_list), max_neighbors))

                #assert len(subgraph_vertices.intersection(sampled_neighbors)) == 0
                next_gen.update(sampled_neighbors)
                subgraph_vertices.update(sampled_neighbors)

                m = len(sampled_neighbors)
                vertices_list.extend(sampled_neighbors)
                vertices_indexes.update(zip(vertices_list[n:], range(n, n+m)))
                n += m

                for v in sampled_neighbors:
                    neighbors_v = list(self.adj_sets[v].intersection(subgraph_vertices).difference({v}))
                    mv = len(neighbors_v)
                    subgraph_edges.update(zip(
                        [vertices_indexes[v]]*mv,
                        (vertices_indexes[w] for w in neighbors_v),
                        (self.adj_dict[v][w] for w in neighbors_v)))
                    subgraph_edges.update(zip(
                        (vertices_indexes[w] for w in neighbors_v),
                        [vertices_indexes[v]]*mv,
                        (self.adj_dict[v][w] for w in neighbors_v)))

            if i == 0:
                n1 = n
        subgraph_edges = list(subgraph_edges)

        return subgraph_edges, n, n1, vertices_list, vertices_indexes

    def sparseAdjacencyTensor(self, subgraph=None, n=None, add_diag=False):
        if subgraph is None:
            subgraph = self.graph_sparse
            n = 64719

        subgraph = np.array(subgraph, dtype=np.int)

        if add_diag:
            diag1 = np.array([[i, i, 1] for i in range(n)])
            subgraph = np.concatenate((subgraph, diag1), 0)

        sparse_tensor = torch.sparse_coo_tensor(np.transpose(subgraph[:, :2]), subgraph[:, 2], (n, n), dtype=torch.float32, device=graph.device)
        return sparse_tensor


class light_Graph():

    def __init__(self, year, min_degs=None):
        self.year = year
        self.nVertices = graph.nVertices
        # self.nVertices = n
        self.graph_sparse = create_graph(year)
        # self.graph_sparse = graph

        self._create_graph(min_degs)
    
    def _create_graph(self, min_degs):
        self.adj_list = [set() for _ in range(self.nVertices)]
        if min_degs == None:
            for e in self.graph_sparse:
                if e[0] != e[1]:
                    self.adj_list[e[0]].add(e[1])
                    self.adj_list[e[1]].add(e[0])
        else:
            degs = np.zeros((self.nVertices), dtype=np.uintc)

            for e in self.graph_sparse:
                if e[0] != e[1]:
                    self.adj_list[e[0]].add(e[1])
                    self.adj_list[e[1]].add(e[0])
                    degs[e[0]] += 1
                    degs[e[1]] += 1
            self.initial_nodes = list(np.where(degs >= min_degs)[0])
        


    def subgraph_khop(self, vertices: list, k_init, k_max, max_neighbors):

        subgraph_vertices = set(vertices)
        next_layer = set(vertices)
        current_layer = set()
        subgraph_edges = set()

        vertices = vertices.copy()

        degrees = np.array([0 for _ in range(self.nVertices)])

        for vertex in next_layer:
            neighbors_set = (self.adj_list[vertex] & subgraph_vertices) - {vertex}
            
            subgraph_edges.update(zip([vertex]*len(neighbors_set), neighbors_set))
            degrees[vertex] = len(neighbors_set)

        for i in range(k_init):

            current_layer = next_layer.copy()
            next_layer.clear()
            assert len(next_layer) == 0

            for vertex in current_layer:

                neighbors_set = self.adj_list[vertex] - subgraph_vertices
                neighbors_list = list(neighbors_set)
                sampled_neighbors = random.sample(neighbors_list, k=min(len(neighbors_list), max_neighbors))

                vertices.extend(sampled_neighbors)
                next_layer.update(sampled_neighbors)
                subgraph_vertices.update(sampled_neighbors)

                degrees[vertex] += len(sampled_neighbors)

                for u in sampled_neighbors:
                    neighbors_u = list(self.adj_list[u] & subgraph_vertices)
                    mu = len(neighbors_u)
                    subgraph_edges.update(zip([u] * mu, neighbors_u))
                    subgraph_edges.update(zip(neighbors_u, [u] * mu))

                    degrees[u] = mu

        n_init = len(vertices)

        for i in range(k_max):

            current_layer = next_layer.copy()
            next_layer.clear()

            for vertex in current_layer:

                neighbors_set = self.adj_list[vertex] - subgraph_vertices
                neighbors_list = list(neighbors_set)
                sampled_neighbors = random.sample(neighbors_list, k=min(len(neighbors_list), max_neighbors))

                vertices.extend(sampled_neighbors)
                next_layer.update(sampled_neighbors)
                subgraph_vertices.update(sampled_neighbors)

                degrees[vertex] += len(sampled_neighbors)

                for u in sampled_neighbors:
                    neighbors_u = list(self.adj_list[u] & subgraph_vertices)
                    mu = len(neighbors_u)
                    subgraph_edges.update(zip([u] * mu, neighbors_u))
                    subgraph_edges.update(zip(neighbors_u, [u] * mu))

                    degrees[u] = mu

        return self.construct_subgraph(vertices, subgraph_edges, degrees), n_init

    def construct_subgraph(self, vertices, subgraph_edges, degrees):

        global2local = dict()

        local_degrees = []

        for c, vertex in enumerate(vertices):
            global2local[vertex] = c
            local_degrees.append(degrees[vertex])

        edges = []
        for edge in subgraph_edges:
            edges.append([global2local[edge[0]], global2local[edge[1]]])

        return vertices, edges, local_degrees, global2local


class graph_ultimate():

    def __init__(self, graph_sparse, min_degs=None):
        self.nVertices = graph.nVertices
        self._create_graph(min_degs, graph_sparse)
    
    def _create_graph(self, min_degs, graph_sparse):
        self.adj_list = [set() for _ in range(self.nVertices)]
        degs = np.zeros((self.nVertices), dtype=np.uintc)

        for e in graph_sparse:
            if e[0] != e[1]:
                self.adj_list[e[0]].add(e[1])
                self.adj_list[e[1]].add(e[0])
                degs[e[0]] += 1
                degs[e[1]] += 1
        self.degs = degs
        self.initial_nodes = list(np.where(degs >= min_degs)[0])

    def subgraph_khop(self, vertices: list, k_init, k_max, max_neighbors):

        subgraph_vertices = set(vertices)
        next_layer = set(vertices)
        current_layer = set()
        subgraph_edges = set()

        vertices = vertices.copy()

        degrees = np.array([0 for _ in range(self.nVertices)])

        for vertex in next_layer:
            neighbors_set = (self.adj_list[vertex] & subgraph_vertices) - {vertex}
            
            subgraph_edges.update(zip([vertex]*len(neighbors_set), neighbors_set))
            degrees[vertex] = len(neighbors_set)

        for i in range(k_init):

            current_layer = next_layer.copy()
            next_layer.clear()
            assert len(next_layer) == 0

            for vertex in current_layer:

                neighbors_set = self.adj_list[vertex] - subgraph_vertices
                neighbors_list = list(neighbors_set)
                sampled_neighbors = random.sample(neighbors_list, k=min(len(neighbors_list), max_neighbors))

                vertices.extend(sampled_neighbors)
                next_layer.update(sampled_neighbors)
                subgraph_vertices.update(sampled_neighbors)

                degrees[vertex] += len(sampled_neighbors)

                for u in sampled_neighbors:
                    neighbors_u = list(self.adj_list[u] & subgraph_vertices)
                    mu = len(neighbors_u)
                    subgraph_edges.update(zip([u] * mu, neighbors_u))
                    subgraph_edges.update(zip(neighbors_u, [u] * mu))

                    degrees[u] = mu

        n_init = len(vertices)

        for i in range(k_max):

            current_layer = next_layer.copy()
            next_layer.clear()

            for vertex in current_layer:

                neighbors_set = self.adj_list[vertex] - subgraph_vertices
                neighbors_list = list(neighbors_set)
                sampled_neighbors = random.sample(neighbors_list, k=min(len(neighbors_list), max_neighbors))

                vertices.extend(sampled_neighbors)
                next_layer.update(sampled_neighbors)
                subgraph_vertices.update(sampled_neighbors)

                degrees[vertex] += len(sampled_neighbors)

                for u in sampled_neighbors:
                    neighbors_u = list(self.adj_list[u] & subgraph_vertices)
                    mu = len(neighbors_u)
                    subgraph_edges.update(zip([u] * mu, neighbors_u))
                    subgraph_edges.update(zip(neighbors_u, [u] * mu))

                    degrees[u] = mu

        return self.construct_subgraph(vertices, subgraph_edges, degrees), n_init

    def construct_subgraph(self, vertices, subgraph_edges, degrees):

        global2local = dict()

        local_degrees = []

        for c, vertex in enumerate(vertices):
            global2local[vertex] = c
            local_degrees.append(degrees[vertex])

        edges = []
        for edge in subgraph_edges:
            edges.append([global2local[edge[0]], global2local[edge[1]]])

        return vertices, edges, local_degrees, global2local
        


def sparseAdjacencyTensor(subgraph, n=None, add_diag=False):

    subgraph = torch.tensor(subgraph, dtype=torch.int64)

    if add_diag:
        diag1 = torch.tensor([[i, i] for i in range(n)], dtype=torch.float32)
        subgraph = torch.cat((subgraph, diag1), 0)

    sparse_tensor = torch.sparse_coo_tensor(torch.transpose(subgraph, 0, 1), torch.ones(subgraph.shape[0], dtype=torch.float32), (n, n), dtype=torch.float32, device=graph.device)
    return sparse_tensor


# Node Features
def eigenvectorCentrality(g, k=1000, embeddingDim=32):
    m = sparse.csr_matrix((np.ones(len(g)), (g[:, 0], g[:, 1])), shape=(graph.nVertices, graph.nVertices))
    vals, vecs = eigs(m, k=k, which='LM')
    #vals = list(map(lambda x: x.real, vals))
    vecs = list(map(lambda x: list(map(lambda y: y.real, x)), vecs))
    #vals = np.array(vals)
    # i = np.array(sorted(list(range(k)), key= lambda i: vals[i])[:embeddingDim])
    # vals = np.array(vals, dtype=np.float32)[i]
    vecs = np.array(vecs, dtype=np.float32) # np.expand_dims((1/vals), 0)*np.array(vecs, dtype=np.float32)#[:, i]
    # return vecs/np.std(vecs)
    return vecs # (vecs - np.min(vecs, 0, keepdims=True)) / (np.max(vecs, 0, keepdims=True) - np.min(vecs, 0, keepdims=True))

def degrees2_neighsConnections(g: Graph):
    from tqdm import tqdm
    degrees2 = dict([[i, 0] for i in range(g.nVertices)])
    neighsConnections = dict([[i, 0] for i in range(g.nVertices)])
    for u in tqdm(range(g.nVertices)):
        neighs1 = g.adj_sets[u]
        ku = len(neighs1)
        n = 0
        neighs2 = set()
        for v in neighs1:
            neighs2.update(g.adj_sets[v].difference(neighs1))
            n += len(g.adj_sets[v].intersection(neighs1))
        degrees2[u] = len(neighs2)
        m = ku*(ku-1) if ku>1 else 1
        neighsConnections[u] = n / m
    return degrees2, neighsConnections


# Edge features
def resource_allocation_index(g: Graph, u, v):
    s = 0
    for w in g.adj_sets[u].intersection(g.adj_sets[v]):
        s += 1/len(g.adj_sets[w])
    return s

def jaccard_coefficient(g: Graph, u, v):
    return len(g.adj_sets[u].intersection(g.adj_sets[v]))/max(1, len(g.adj_sets[u].union(g.adj_sets[v])))

def adamic_adar_index(g: Graph, u, v):
    s = 0
    for w in g.adj_sets[u].intersection(g.adj_sets[v]):
        s += 1/np.math.log(len(g.adj_sets[w]))
    return s

def preferential_attachment(g: Graph, u, v):
    return len(g.adj_sets[u])*len(g.adj_sets[v])

def dispersion(g: Graph, u, v):
    return nx.dispersion(g.nx_graph, u, v)


# Save features
def saveFeatures(method, method_name, years):
    from time import time
    for y in years:
        g = Graph(y)
        # d1, d2 = method(g)
        top = time()
        d = method(g)
        print(time() - top)
        if isinstance(d, dict):
            file = open(f"./Features/{y}/{method_name}.pkl", "wb")
            pickle.dump(d, file)
            file.close()
        elif isinstance(d, np.ndarray):
            np.save(f"./Features/{y}/{method_name}.npy")

if __name__ == '__main__':
    pass
    # saveFeatures(lambda g: nx.degree_centrality(g.nx_graph), "degree", range(2000, 2018))
    # saveFeatures(lambda g: nx.eigenvector_centrality(g.nx_graph, max_iter=1000), "eigenvector1", range(2000, 2018))
    # saveFeatures(degrees2_neighsConnections, ["degrees2", "neighsConnection"], range(2000, 2018))
    # saveFeatures(lambda g: nx.pagerank(g.nx_graph), "pagerank", range(2000, 2018))
 