from scipy.sparse import data
import torch
import numpy as np
import random

import graph
import graph.data_utils
from graph.graph_utils import Graph, adamic_adar_index, dispersion, eigenvectorCentrality, jaccard_coefficient, light_Graph, preferential_attachment, resource_allocation_index, saveFeatures
from graph.graph_utils import sparseAdjacencyTensor, graph_ultimate

from datetime import date
from tqdm import tqdm


def create_dataset_from_to(starting_year, finish_year, p, minimal_degs, max_size_links=1, shuffle=True):
    edges_list_start = graph.data_utils.create_graph(starting_year)
    edges_list_end = graph.data_utils.create_graph(finish_year)

    day_delta_start = date(starting_year, 12, 31) - date(1990, 1, 1)
    day_delta_end = date(finish_year, 12, 31) - date(1990, 1, 1)

    new_links = edges_list_end[day_delta_start.days < edges_list_end[:, 2]]
    new_links = new_links[new_links[:, 2] < day_delta_end.days]
    unconnected_vertex_pairs = []
    unconnected_vertex_pairs_solution = []

    # Set max_size of new links
    if max_size_links <= 1:
        max_size_links = int(max_size_links*new_links.shape[0])

    indexes = list(range(new_links.shape[0]))
    if shuffle:
        random.shuffle(indexes)
    indexes = np.array(indexes[:max_size_links])
    for link in new_links[indexes]:
        unconnected_vertex_pairs.append([link[0], link[1]])
        unconnected_vertex_pairs_solution.append(True)


    degs = np.zeros((graph.nVertices,), dtype=int)
    for edge in edges_list_end:
        if edge[0] != edge[1]:
            degs[edge[0]] += 1
            degs[edge[1]] += 1

    all_vertices = np.array(range(graph.nVertices))
    vertex_large_degs = list(all_vertices[degs >= minimal_degs])

    edges_set_start = set(((e[0], e[1]) for e in edges_list_start))
    edges_set_end = set(((e[0], e[1]) for e in edges_list_end))
    while(len(unconnected_vertex_pairs) < p * max_size_links):

        # iv1, iv2 = random.sample(range(len(vertex_large_degs)), k=2)
        # v1, v2 = vertex_large_degs[iv1], vertex_large_degs[iv2]
        v1, v2 = random.sample(vertex_large_degs, 2)


        # Remove v1 != V2 ?
        if v1 != v2 and (v1, v2) not in edges_set_start and (v2, v1) not in edges_set_start:
            unconnected_vertex_pairs.append([v1, v2])
            solution = ((v1, v2) in edges_set_end) | ((v2, v1) in edges_set_end)
            unconnected_vertex_pairs_solution.append(solution)

    print('Ratio links: ', sum(unconnected_vertex_pairs_solution)/len(unconnected_vertex_pairs_solution))

    return edges_list_start, unconnected_vertex_pairs, unconnected_vertex_pairs_solution


def shuffle_dataset(pairs, labels):
    pairs, labels = np.array(pairs), np.array(labels)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    indices = np.array(indices)
    pairs = pairs[indices]
    labels = labels[indices]
    return pairs, labels


def save_dataset(years, starting_year, finish_year, p, minimal_degs, edges_features=False):
    graphs = [Graph(y) for y in years]
    edges, pairs, labels = create_dataset_from_to(starting_year, finish_year, p, minimal_degs)
    if edges_features:
        pairs_features = []
        for [u, v] in tqdm(pairs):
            f = []
            for g in graphs:
                f.extend([
                    resource_allocation_index(g, u, v),
                    jaccard_coefficient(g, u, v),
                    adamic_adar_index(g, u, v),
                    preferential_attachment(g, u, v)
                ])
            pairs_features.append(f)
        pairs_features = np.array(pairs_features, dtype=np.float)
        np.save(f'./Datasets/{starting_year}_{finish_year}_pairs_features.npy', pairs_features)

    pairs = np.array(pairs, dtype=np.int)
    labels = np.array(labels, dtype=np.int)

    np.save(f'./Datasets/{starting_year}_{finish_year}_pairs.npy', pairs)
    np.save(f'./Datasets/{starting_year}_{finish_year}_labels.npy', labels)


def load_dataset(datapath, normalize=True, edge_features=False):
    pairs = np.load(datapath+'_pairs.npy')
    labels = np.load(datapath+'_labels.npy')
    if edge_features:
        pairs_features = np.load(datapath+'_pairs_features.npy')
        if normalize:
            # pairs_features -= np.mean(pairs_features, 0, keepdims=True)
            # pairs_features /= 1e-4 + np.std(pairs_features, 0, keepdims=True)
            pairs_features = (pairs_features - np.min(pairs_features, 0, keepdims=True)) / (np.max(pairs_features, 0, keepdims=True)-np.min(pairs_features, 0, keepdims=True))
    else:
        pairs_features = None
    return pairs, labels, pairs_features


class DataLoader_multiple_years():

    def __init__(self, years, ponderate=False, normalize=True):
        self.years = years
        self.graphs = [Graph(y, ponderated=ponderate) for y in years]

    def initNodesFeatures(self, method=None, embeddingDim=32):
        if method == 'random':
            return torch.rand((64719, embeddingDim), dtype=torch.float32)

        if method == 'eigen':
            f = [torch.tensor(eigenvectorCentrality(g, embeddingDim, embeddingDim), dtype=torch.float32) for g in self.graphs]
            f = torch.cat(f, -1)
            return f

        else:
            f = [g.load_centralities() for g in self.graphs]
            f = torch.cat(f, -1)
            f = f.numpy()
            f = (f -np.min(f, 0, keepdims=True)) / (np.max(f, 0, keepdims=True) - np.min(f, 0, keepdims=True))
            f = torch.tensor(f, dtype=torch.float32)
            # f -= torch.mean(f, 0, keepdim=True)
            # f /= 1e-4 + torch.std(f, 0, keepdim=True)
            # f -= torch.min(f, 0, keepdim=True)
            # f /= torch.max(f, 0, keepdim=True) - torch.min(f, 0, keepdim=True)
            return f

    def Generator(self, datapath, batch_size=16, shuffle=True, k=2, add_diag=True, edges_features=False):
        pairs, labels, pairs_features = load_dataset(datapath)

        if shuffle:
            indices = list(range(len(pairs)))
            random.shuffle(indices)
            indices = np.array(indices, dtype=np.int)
            pairs = pairs[indices]
            labels = labels[indices]
            if edges_features:
                pairs_features = pairs_features[indices]

        for l in range(0, pairs.shape[0]-batch_size, batch_size):
            batch_pairs = pairs[l: l+batch_size]
            batch_labels = labels[l: l+batch_size]
            if edges_features:
                batch_pairs_features = pairs_features[l: l+batch_size]

            if k != 0:
                vertices = set()
                for p in batch_pairs:
                    vertices.update(p)
                vertices = list(vertices)

                subgraph_edges, n, n1, vertices, vertices_indexes = self.graphs[-1].KHop(vertices, k, 10_000)
                n1 = len(vertices) # n0
                batch_pairs = [[vertices_indexes[u], vertices_indexes[v]] for [u, v] in batch_pairs]
                vertices = np.array(vertices)
                degs12 = 1 / (np.sqrt(self.graphs[-1].deg_list[vertices])+ 1e-10)
                degs12 = np.clip(degs12, 0, 2)
                sparse_adj_tensor = self.graphs[-1].sparseAdjacencyTensor(subgraph_edges, n, add_diag=add_diag)
                degs12 = torch.sparse_coo_tensor(np.tile(np.arange(0, n, 1), (2, 1)), degs12, (n, n), dtype=torch.float32, device=graph.device)
                vertices = torch.tensor(vertices, dtype=torch.int64)

            else:
                sparse_adj_tensor = None
                vertices = None
                degs12 = None

            batch_pairs = torch.tensor(batch_pairs, dtype=torch.int64)
            batch_labels = torch.tensor(batch_labels, dtype=torch.int64)
            if edges_features:
                batch_pairs_features = torch.tensor(batch_pairs_features, dtype=torch.float32)
            else:
                batch_pairs_features = None
            yield sparse_adj_tensor, degs12, vertices, n1, batch_pairs, batch_labels, batch_pairs_features


class unsupervised_dataLoader():

    def __init__(self, years, min_degs):
        self.years = years
        self.min_degs = min_degs
        self.years_depth = len(years)
        self.graphs = [light_Graph(y, min_degs) for y in years]
        self.nVertices = 64719


    def init_nodes_embedding(self, method, initial_embedding_dim, normalize=False):
        if method == 'random':
            f = torch.rand((self.years_depth , self.nVertices, initial_embedding_dim), dtype=torch.float32)

        if method == 'eigen':
            f = [torch.tensor(eigenvectorCentrality(g, initial_embedding_dim, initial_embedding_dim), dtype=torch.float32) for g in self.graphs]
            f = torch.stack(f, 0)

        if method == 'eigen2017':
            f = torch.tensor(eigenvectorCentrality(light_Graph(2017, self.min_degs), initial_embedding_dim, initial_embedding_dim), dtype=torch.float32)
            f = torch.stack(tuple(f for _ in range(self.years_depth)), 0)

        else:
            NotImplementedError(f"Following method not implemented : {method}")

        if normalize:
            f = f.numpy()
            f = (f - np.min(f, 1, keepdims=True)) / (np.max(f, 1, keepdims=True) - np.min(f, 1, keepdims=True))
            f = torch.tensor(f, dtype=torch.float32)

        return f

    def Generator(self, iter_number, max_neighbors, k_init, k_max, starting_size=16, add_diag=True):

        while(iter_number):
            vertices = random.sample(self.graphs[iter_number % self.years_depth].initial_nodes, k=starting_size)
            (vertices, edges, degs, _), n_init = self.graphs[iter_number % self.years_depth].subgraph_khop(vertices, k_init=k_init, k_max=k_max, max_neighbors=max_neighbors)
            del _

            degs = np.array(degs)
            degs12 = 1 / (np.sqrt(degs) + 1e-10)
            degs12 = np.clip(degs12, 0, 2)
            degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)

            sparse_adj_tensor = sparseAdjacencyTensor(edges, len(vertices), add_diag=add_diag)

            values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]

            sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values)

            iter_number -= 1

            yield sparse_adj_tensor, vertices, n_init, (iter_number % self.years_depth)



class supervised_dataLoader():

    def __init__(self, years, finish_year, max_neighbors, k, batch_size=16, add_diag=True, p=4, min_degs=10, graphs=None, create_dataset=True):
        self.years = years
        self.finish_year = finish_year
        self.years_depth = len(years)
        if graphs == None:
            self.graphs = [light_Graph(y) for y in years]
        else:
            self.graphs = graphs
        self.nVertices = 64719
        self.max_neighbors = max_neighbors
        self.k = k
        self.batch_size = batch_size
        self.add_diag = add_diag
        self.min_degs_dataset = min_degs

        if create_dataset:
            _, self.pairs, self.labels = create_dataset_from_to(self.years[-1], finish_year, p, self.min_degs_dataset)
            del _


    def Generator(self):
        self.pairs, self.labels = shuffle_dataset(self.pairs, self.labels)

        for k in range(0, len(self.pairs)-self.batch_size, self.batch_size):
            batch_pairs = self.pairs[k: k+self.batch_size].copy()
            batch_labels = self.labels[k: k+self.batch_size].copy()
            vertices = list(np.concatenate(batch_pairs, 0))
            yield self.sub_Generator(vertices, batch_pairs), torch.tensor(batch_labels, dtype=torch.int64, device=graph.device)

    def sub_Generator(self, init_vertices, pairs):

        for i in range(self.years_depth):
            (vertices, edges, degs, global2local), _ = self.graphs[i].subgraph_khop(init_vertices.copy(), k_init=0, k_max=self.k, max_neighbors=self.max_neighbors)
            del _
            vertices = torch.tensor(vertices, dtype=torch.int64, device=graph.device)

            degs = np.array(degs)
            degs12 = 1 / (np.sqrt(degs) + 1e-10)
            degs12 = np.clip(degs12, 0, 2)
            degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)

            sparse_adj_tensor = sparseAdjacencyTensor(edges, len(vertices), add_diag=self.add_diag)

            values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]

            sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values)

            pairs_tensor = torch.tensor([[global2local[u], global2local[v]] for [u, v] in pairs], dtype=torch.int64, device=graph.device)

            yield sparse_adj_tensor, vertices, i, pairs_tensor


class supervised_data_loader_with_test():

    def __init__(self, delta_years=3, years_min=2008, years_max=2013, minimal_degs=10, p=4, max_neighbors=1000,
                 k=2, pairs_batch_size=16, add_diag=True, normalize_adj_m=True, test_size=1024) -> None:
        self.years = list(range(years_min, years_max+1))
        self.years_min = years_min
        self.years_max = years_max
        self.delta_years = delta_years

        self.k = k
        self.max_neighbors = max_neighbors
        self.pairs_batch_size = pairs_batch_size
        self.add_diag = add_diag
        self.normalize_adj_m = normalize_adj_m

        self.graphs = [light_Graph(y, 0) for y in self.years]
        self.datasets = [create_dataset_from_to(starting_year=y, finish_year=y+3, p=p, minimal_degs=minimal_degs,  max_size_links=1)[1:] for y in range(years_min+delta_years-1, years_max+1)]

        self.test_size = test_size
        self.graphs_test = [light_Graph(y, 0) for y in range(2015-delta_years, 2015)] # Predictions from 2014 for 2017
        self.dataset_test = create_dataset_from_to(2014, 2017, p, minimal_degs, test_size//p, shuffle=False)[1:]


    def Generator(self, iter_number, test_rate):
        c = 0
        # Shuffle each dataset
        for i in range(len(self.datasets)):
            self.datasets[i] = shuffle_dataset(*self.datasets[i])

        # Shuffle indexes of which dataset we pick the batch
        indexes_datasets = []
        for i in range(len(self.datasets)):
            indexes_datasets.extend([i]*(self.datasets[i][0].shape[0]//self.pairs_batch_size))
        random.shuffle(indexes_datasets)
        indexes_datasets = indexes_datasets[:min(len(indexes_datasets), iter_number)]

        # indexes representing where we are in each dataset
        indexes_progress = [0] * len(self.datasets)

        for i in indexes_datasets:
            batch_pairs = self.datasets[i][0][indexes_progress[i]:indexes_progress[i]+self.pairs_batch_size]
            batch_labels = self.datasets[i][1][indexes_progress[i]:indexes_progress[i]+self.pairs_batch_size]
            indexes_progress[i] += self.pairs_batch_size

            batch_graphs = [self.graphs[i+c] for c in range(self.delta_years)]
            batch_year_start = self.years[i+self.delta_years-1]
            yield self.subGenerator(batch_pairs, batch_labels, batch_graphs), False, batch_year_start
            c += 1

            if c == test_rate:
                c = 0
                yield self.subGenerator(self.dataset_test[0], self.dataset_test[1], self.graphs_test), True, 2014

    def subGenerator(self, pairs, labels, graphs):
        init_vertices = list(np.concatenate(pairs, 0))
        labels = torch.tensor(labels, dtype=torch.int64, device=graph.device)
        for g in graphs:
            (vertices, edges, degs, global2local), _ = g.subgraph_khop(init_vertices.copy(), k_init=0, k_max=self.k, max_neighbors=self.max_neighbors)
            del _
            vertices = torch.tensor(vertices, dtype=torch.int64, device=graph.device)

            sparse_adj_tensor = sparseAdjacencyTensor(edges, n=vertices.size(0), add_diag=self.add_diag)

            if self.normalize_adj_m:
                degs = np.array(degs)
                degs12 = 1 / (np.sqrt(degs) + 1e-10)
                degs12 = np.clip(degs12, 0, 2)
                degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)
                values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]
                sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values)

            pairs_indices = torch.tensor([[global2local[u], global2local[v]] for [u, v] in pairs], dtype=torch.int64, device=graph.device)

            yield sparse_adj_tensor, vertices, pairs_indices, labels


def make_matrix(g, add_diag, normalize_matrix):
    edges = []
    for u in range(graph.nVertices):
        for v in g.adj_list[u]:
            edges.append([u, v])
    edges = np.array(edges)
    sparse_adj_tensor = sparseAdjacencyTensor(edges, n=graph.nVertices, add_diag=add_diag)

    if normalize_matrix:
        degs = [len(g.adj_list[u]) for u in range(graph.nVertices)]
        degs = np.array(degs)
        degs12 = 1 / (np.sqrt(degs) + 1e-10)
        degs12 = np.clip(degs12, 0, 2)
        degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)
        values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]
        sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values, size=(graph.nVertices, graph.nVertices), device=graph.device)
    
    return sparse_adj_tensor


class supervised_data_loader_with_test_all_vertices():

    def __init__(self, p=4, pairs_batch_size=16, add_diag=False, normalize_adj_m=True, test_size=250_000) -> None:
        self.delta_years = 3
        self.pairs_batch_size = pairs_batch_size
        self.add_diag = add_diag
        self.normalize_adj_m = normalize_adj_m
        self.years = [2014-3, 2013-3, 2012-3, 2011-3, 2010-3]

        self.matrices = dict()
        self.datasets = dict()
        for y in self.years:
            self.matrices[y] = [
                make_matrix(light_Graph(y-2, 0), add_diag, normalize_adj_m).to("cpu"),
                make_matrix(light_Graph(y-1, 0), add_diag, normalize_adj_m).to("cpu"),
                make_matrix(light_Graph(y, 0), add_diag, normalize_adj_m).to("cpu")
            ]
            self.datasets[y] = create_dataset_from_to(y, y+3, p=p, minimal_degs=5, max_size_links=0.2)[1:]
            self.datasets[y] = shuffle_dataset(*self.datasets[y])

        self.test_size = test_size
        self.matrices_test = [
            make_matrix(light_Graph(2012, 0), add_diag, normalize_adj_m).to("cpu"),
            make_matrix(light_Graph(2013, 0), add_diag, normalize_adj_m).to("cpu"),
            make_matrix(light_Graph(2014, 0), add_diag, normalize_adj_m).to("cpu")
        ]
        self.dataset_test = create_dataset_from_to(2014, 2017, 1000, 5, test_size//1000, shuffle=False)[1:]


    def Generator(self, iter_number, test_rate):
        c = 0
        # Shuffle indexes of which dataset we pick the batch
        indexes_datasets = []
        for y in self.years:
            indexes_datasets.extend([y]*(self.datasets[y][0].shape[0]//self.pairs_batch_size))
        random.shuffle(indexes_datasets)
        indexes_datasets = indexes_datasets[:min(len(indexes_datasets), iter_number)]
        indexes_progress = dict([(y, 0) for y in self.years])

        for y in indexes_datasets:
            batch_pairs = self.datasets[y][0][indexes_progress[y]:indexes_progress[y]+self.pairs_batch_size]
            batch_labels = self.datasets[y][1][indexes_progress[y]:indexes_progress[y]+self.pairs_batch_size]

            indexes_progress[y] += self.pairs_batch_size

            yield self.subGenerator(batch_pairs, batch_labels, self.matrices[y]), False, y
            c += 1

            if c == test_rate:
                c = 0
                yield self.subGenerator(self.dataset_test[0], self.dataset_test[1], self.matrices_test), True, 2014

    def subGenerator(self, pairs, labels, matrices):
        pairs = torch.tensor(pairs, dtype=torch.int64, device=graph.device)
        labels = torch.tensor(labels, dtype=torch.int64, device=graph.device)
        for m in matrices:
            yield m.to(graph.device), pairs, labels

def create_dataset_using(start_data, end_data, p, minimal_degs, max_size_links=1):

    edges_list_start, starting_year = start_data
    edges_list_end, finish_year = end_data

    day_delta_start = date(starting_year, 12, 31) - date(1990, 1, 1)
    day_delta_end = date(finish_year, 12, 31) - date(1990, 1, 1)

    new_links = edges_list_end[day_delta_start.days < edges_list_end[:, 2]]
    
    unconnected_vertex_pairs = []
    unconnected_vertex_pairs_solution = []

    # Set max_size of new links
    if max_size_links <= 1:
        max_size_links = int(max_size_links*new_links.shape[0])
    max_size_links = min(max_size_links, new_links.shape[0])

    indexes = list(range(max_size_links))


    for link in new_links[indexes]:
        unconnected_vertex_pairs.append([link[0], link[1]])
        unconnected_vertex_pairs_solution.append(True)


    degs = np.zeros((graph.nVertices,), dtype=int)
    for edge in edges_list_end:
        if edge[0] != edge[1]:
            degs[edge[0]] += 1
            degs[edge[1]] += 1

    all_vertices = np.array(range(graph.nVertices))
    vertex_large_degs = list(all_vertices[degs >= minimal_degs])

    edges_set_start = set(((e[0], e[1]) for e in edges_list_start))
    edges_set_end = set(((e[0], e[1]) for e in edges_list_end))
    
    new_links = 0
    
    while(new_links < p * max_size_links):

        v1, v2 = random.sample(vertex_large_degs, 2)

        if v1 != v2 and (v1, v2) not in edges_set_start and (v2, v1) not in edges_set_start:
            unconnected_vertex_pairs.append([v1, v2])
            solution = ((v1, v2) in edges_set_end) | ((v2, v1) in edges_set_end)
            unconnected_vertex_pairs_solution.append(solution)
            
        new_links += 1

    print('Ratio links: ', sum(unconnected_vertex_pairs_solution)/len(unconnected_vertex_pairs_solution))

    return [np.array(unconnected_vertex_pairs, dtype=np.int64), np.array(unconnected_vertex_pairs_solution, dtype=np.int64)]

def clipped_datasets(datasets):

    # -- shuffle datasets
    indices = [np.arange(dataset[1].shape[0]) for dataset in datasets]
    for i in range(len(indices)):
        np.random.shuffle(indices[i])
    for i in range(len(datasets)):
        datasets[i][0] = datasets[i][0][indices[i]]
        datasets[i][1] = datasets[i][1][indices[i]]

    # -- compute minimal size
    minimal_size = min((dataset[1].shape[0] for dataset in datasets))
    print(f"Minimal size of dataset: {minimal_size}")

    # -- create clipped datasets
    clipped_datasets = []
    for i in range(len(datasets)):
        print(f"size of dataset before clipping: {datasets[i][1].shape[0]}")
        clipped_datasets.append([datasets[i][0][:minimal_size], datasets[i][1][:minimal_size]])

    return clipped_datasets, minimal_size

def load_test_dataset():
    pairs = np.load("./test_dataset/pairs.npy")
    labels = np.load("./test_dataset/labels.npy")
    return [pairs, labels]




class ultimate_dataloader:
    """
    Generator : represent one epoch
    return subgenerator, is_test, starting_years, 
    


    subgenerator : represent a batch
    return sparse_adj_tensor, pairs, labels
    from starting_years to starting_years + delta_years - 1
    pairs = (batch_size * 2)
    labels = (batch_size)
    sparse_adj_tensor    
    
    """

    def __init__(self, min_year, proportion_dataset, minimal_degs_unsupervised, minimal_degs_supervised, add_diag, normalize_adj_matrix, max_size):

        # --load dataset
        graph_sparse, _, _, _ = graph.data_utils.load_data('CompetitionSet2017_3.pkl')

        # -- extract every graph
        graphs_years = [(graph.data_utils.extract_graph(graph_sparse, year), year) for year in range(min_year, 2017 + 1)]
        del graph_sparse
        del _

        # -- create graph until 2014
        self.graphs = [graph_ultimate(g[0], minimal_degs_unsupervised) for g in graphs_years[0:-3]]

        # -- compute adj matrix
        self.matrices = [make_matrix(g, add_diag, normalize_adj_matrix) for g in self.graphs]

        # -- create datasets
        self.full_datasets = [create_dataset_using(graphs_years[i], graphs_years[i+3], proportion_dataset, minimal_degs_supervised, max_size) for i in range(0, len(graphs_years)-3)]

        # -- shuffle datasets
        for i in range(len(self.full_datasets)):
            rng_state = np.random.get_state()
            self.full_datasets[i][0] = np.random.permutation(self.full_datasets[i][0])
            np.random.set_state(rng_state)
            self.full_datasets[i][1] = np.random.permutation(self.full_datasets[i][1])
        self.datasets = self.full_datasets[:-3]
        del graphs_years


        # -- create test datasets
        self.test_dataset = load_test_dataset()
        self.test_size = self.test_dataset[0].shape[0]

        self.min_year = min_year

    def supervised_generator3(self, batch_size, delta_years, iter_number):

        # -- initialize counter
        c = 0

        # training_datasets, dataset_size = clipped_datasets(self.datasets[delta_years-1:]) 
        training_datasets = [[self.full_datasets[-1][0].copy(), self.full_datasets[-1][1].copy()]]
        print(training_datasets[0][1].shape)
        # -- shuffle datasets
        for i in range(len(training_datasets)):
            rng_state = np.random.get_state()
            training_datasets[i][0] = np.random.permutation(training_datasets[i][0])[:iter_number]
            np.random.set_state(rng_state)
            training_datasets[i][1] = np.random.permutation(training_datasets[i][1])[:iter_number]
            print(training_datasets[0][1].shape)

        # for i in range(0, dataset_size - batch_size, batch_size):       
        #     for l, dataset in enumerate(training_datasets):
        
        # Shuffle indexes of which dataset we pick the batch
        indexes_datasets = []
        for y in range(len(training_datasets)):
            indexes_datasets.extend([y]*(training_datasets[y][0].shape[0]//batch_size))
        random.shuffle(indexes_datasets)
        # indexes_datasets = indexes_datasets[:min(len(indexes_datasets), iter_number)]
        indexes_progress = [0] * len(training_datasets)

        for y in indexes_datasets:
            pairs = training_datasets[y][0][indexes_progress[y]:indexes_progress[y]+batch_size]
            labels = training_datasets[y][1][indexes_progress[y]:indexes_progress[y]+batch_size]

            indexes_progress[y] += batch_size

            # pairs =     dataset[0][i : i+batch_size]
            # labels =    dataset[1][i : i+batch_size]
            matrices = self.matrices[-delta_years:]
            yield self.subgenerator(pairs, labels, matrices), False, 2014

            c += 1


    def supervised_generator2(self, batch_size, delta_years, test_rate):

        # -- initialize counter
        c = 0

        # training_datasets, dataset_size = clipped_datasets(self.datasets[delta_years-1:]) 
        training_datasets = [self.datasets[-1]]

        # -- shuffle datasets
        for i in range(len(training_datasets)):
            rng_state = np.random.get_state()
            training_datasets[i][0] = np.random.permutation(training_datasets[i][0])
            np.random.set_state(rng_state)
            training_datasets[i][1] = np.random.permutation(training_datasets[i][1])

        # for i in range(0, dataset_size - batch_size, batch_size):       
        #     for l, dataset in enumerate(training_datasets):
        
        # Shuffle indexes of which dataset we pick the batch
        indexes_datasets = []
        for y in range(len(training_datasets)):
            indexes_datasets.extend([y]*(training_datasets[y][0].shape[0]//batch_size))
        random.shuffle(indexes_datasets)
        # indexes_datasets = indexes_datasets[:min(len(indexes_datasets), iter_number)]
        indexes_progress = [0] * len(training_datasets)

        for y in indexes_datasets:
            pairs = training_datasets[y][0][indexes_progress[y]:indexes_progress[y]+batch_size]
            labels = training_datasets[y][1][indexes_progress[y]:indexes_progress[y]+batch_size]

            indexes_progress[y] += batch_size

            # pairs =     dataset[0][i : i+batch_size]
            # labels =    dataset[1][i : i+batch_size]
            matrices = self.matrices[-3-delta_years:-3]
            yield self.subgenerator(pairs, labels, matrices), False, 2011

            c += 1

            if c % test_rate == 0:
                matrices = self.matrices[-delta_years:]
                yield self.subgenerator(self.test_dataset[0], self.test_dataset[1], matrices), True, 2014

    def supervised_generator(self, batch_size, delta_years, test_rate):

        # -- initialize counter
        c = 0

        # training_datasets, dataset_size = clipped_datasets(self.datasets[delta_years-1:]) 
        training_datasets = self.datasets[delta_years-1:]

        # -- shuffle datasets
        for i in range(len(training_datasets)):
            rng_state = np.random.get_state()
            training_datasets[i][0] = np.random.permutation(training_datasets[i][0])
            np.random.set_state(rng_state)
            training_datasets[i][1] = np.random.permutation(training_datasets[i][1])

        # for i in range(0, dataset_size - batch_size, batch_size):       
        #     for l, dataset in enumerate(training_datasets):

        # Shuffle indexes of which dataset we pick the batch
        indexes_datasets = []
        for y in range(len(training_datasets)):
            indexes_datasets.extend([y]*(training_datasets[y][0].shape[0]//batch_size))
        random.shuffle(indexes_datasets)
        # indexes_datasets = indexes_datasets[:min(len(indexes_datasets), iter_number)]
        indexes_progress = [0] * len(training_datasets)

        for y in indexes_datasets:
            pairs = training_datasets[y][0][indexes_progress[y]:indexes_progress[y]+batch_size]
            labels = training_datasets[y][1][indexes_progress[y]:indexes_progress[y]+batch_size]

            indexes_progress[y] += batch_size

            # pairs =     dataset[0][i : i+batch_size]
            # labels =    dataset[1][i : i+batch_size]
            matrices = [self.matrices[k] for k in range(y, delta_years + y)]
            yield self.subgenerator(pairs, labels, matrices), False, self.min_year+y+delta_years-1

            c += 1

            if c % test_rate == 0:
                matrices = self.matrices[-delta_years:]
                yield self.subgenerator(self.test_dataset[0], self.test_dataset[1], matrices), True, 2014
                
    def subgenerator(self, pairs, labels, matrices):
        pairs = torch.tensor(pairs, dtype=torch.int64, device=graph.device)
        labels = torch.tensor(labels, dtype=torch.int64, device=graph.device)
        for m in matrices:
            yield m, pairs, labels

    def unsupervised_generator(self, iter_number, max_neighbors, k_init, k_max, starting_size=16, add_diag=True, years=None):

        if years == None:
            training_graphs = self.graphs
        else:
            years = np.array(years, dtype=np.int32)
            years = years - self.min_year
            training_graphs = [self.graphs[y] for y in years]
                    
            years_depth = len(years)

            while(iter_number):
                vertices = random.sample(training_graphs[iter_number % years_depth].initial_nodes, k=starting_size)
                (vertices, edges, degs, _), n_init = training_graphs[iter_number % years_depth].subgraph_khop(vertices, k_init=k_init, k_max=k_max, max_neighbors=max_neighbors)
            del _

            degs = np.array(degs)
            degs12 = 1 / (np.sqrt(degs) + 1e-10)
            degs12 = np.clip(degs12, 0, 2)
            degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)

            sparse_adj_tensor = sparseAdjacencyTensor(edges, len(vertices), add_diag=add_diag)

            values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]

            sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values)

            iter_number -= 1

            yield sparse_adj_tensor, vertices, n_init, (iter_number % years_depth)

    def unsupervised_generator2(self, iter_number, starting_size=16, nb_neighbors=1, nb_non_neighbors=4, years=None):
        all_vertices = set(list(range(graph.nVertices)))

        if years is None:
            training_graphs = self.graphs
            training_matrices = self.matrices
        else:
            years = np.array(years, dtype=np.int32)
            years = years - self.min_year
            training_graphs = [self.graphs[y] for y in years]
            training_matrices = [self.matrices[y] for y in years]

        years_depth = len(years)
        while(iter_number):

            vertices = random.sample(training_graphs[iter_number % len(training_graphs)].initial_nodes, k=starting_size)
            vertices_neighbors = [random.sample(training_graphs[iter_number % len(training_graphs)].adj_list[u], nb_neighbors) for u in vertices]
            vertices_not_neighbors = [random.sample(all_vertices.difference(training_graphs[iter_number % len(training_graphs)].adj_list[u]), nb_non_neighbors) for u in vertices]

            vertices = torch.tensor(vertices, dtype=torch.int64, device=graph.device)
            vertices_neighbors = torch.tensor(vertices_neighbors, dtype=torch.int64, device=graph.device)
            vertices_not_neighbors = torch.tensor(vertices_not_neighbors, dtype=torch.int64, device=graph.device)

            degs = training_graphs[iter_number % len(training_graphs)].degs
            degs12 = 1 / (np.sqrt(degs) + 1e-10)
            degs12 = np.clip(degs12, 0, 2)
            degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)

            matrix = training_matrices[iter_number % len(training_graphs)]

            iter_number -= 1

            yield matrix, vertices, vertices_neighbors, vertices_not_neighbors, (iter_number % years_depth)

    def unsupervised_generator3(self, iter_number, max_neighbors, k_init, k_max, starting_size=16, add_diag=True, years=None):

        if years == None:
            training_graphs = self.graphs
        else:
            years = np.array(years, dtype=np.int32)
            years = years - self.min_year
            training_graphs = [self.graphs[y] for y in years]
            corresponding_datasets = [self.full_datasets[y] for y in years]
        years_depth = len(years)
        while(iter_number):
            vertices = random.sample(list(corresponding_datasets[iter_number % years_depth][0].flatten()), k=starting_size)
            (vertices, edges, degs, _), n_init = training_graphs[iter_number % years_depth].subgraph_khop(vertices, k_init=k_init, k_max=k_max, max_neighbors=max_neighbors)
            del _

            degs = np.array(degs)
            degs12 = 1 / (np.sqrt(degs) + 1e-10)
            degs12 = np.clip(degs12, 0, 2)
            degs12 = torch.tensor(degs12, device=graph.device, dtype=torch.float32)

            sparse_adj_tensor = sparseAdjacencyTensor(edges, len(vertices), add_diag=add_diag)

            values = degs12[sparse_adj_tensor._indices()[0]] * degs12[sparse_adj_tensor._indices()[1]]

            sparse_adj_tensor = torch.sparse_coo_tensor(indices=sparse_adj_tensor._indices(), values=values)

            iter_number -= 1

            yield sparse_adj_tensor, vertices, n_init, (iter_number % years_depth)


def init_nodes_embedding(method, initial_embedding_dim, graphs, min_degs=0, normalize=False):

    years_depth = len(graphs)

    if method == "random":
        f = torch.rand((years_depth , graph.nVertices, initial_embedding_dim), dtype=torch.float32)

    if method == "eigen":
        f = [torch.tensor(eigenvectorCentrality(g, initial_embedding_dim, initial_embedding_dim), dtype=torch.float32) for g in graphs]
        f = torch.stack(f, 0)

    if method == "eigen2017":
        f = torch.tensor(eigenvectorCentrality(light_Graph(2017, min_degs).graph_sparse, initial_embedding_dim, initial_embedding_dim), dtype=torch.float32)
        f = torch.stack(tuple(f for _ in range(years_depth)), 0)
        
    if method == "eigen2014":
        f = torch.tensor(eigenvectorCentrality(light_Graph(2014, min_degs).graph_sparse, initial_embedding_dim, initial_embedding_dim), dtype=torch.float32)
        f = torch.stack(tuple(f for _ in range(years_depth)), 0)

    if method == "eigenMelted":
        f = [torch.tensor(eigenvectorCentrality(g, initial_embedding_dim//len(graphs), initial_embedding_dim), dtype=torch.float32) for g in graphs]
        f.append(
            torch.tensor(eigenvectorCentrality(light_Graph(2017, min_degs).graph_sparse, initial_embedding_dim-len(f)*(initial_embedding_dim//len(graphs)), initial_embedding_dim), dtype=torch.float32)
        )
        f = torch.cat(f, -1)
        f = torch.stack(tuple(f for _ in range(years_depth)), 0)

    else:
        NotImplementedError(f"Following method not implemented : {method}")

    if normalize:
        f = f.numpy()
        f = (f - np.min(f, 1, keepdims=True)) / (np.max(f, 1, keepdims=True) - np.min(f, 1, keepdims=True))
        f = torch.tensor(f, dtype=torch.float32)

    return f


if __name__ == '__main__':
    from time import time
    save_dataset([2012, 2013, 2014], 2014, 2017, 2, 10)
    #load_dataset('Datasets/2014_2017')
