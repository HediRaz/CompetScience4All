import pickle
from datetime import date

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse

import graph


def load_data(data_source='TrainSet2014_3.pkl'):
    graph_sparse, unconnected_vertex, year_start, year_delta = pickle.load(open(data_source, "rb"))

    return graph_sparse, unconnected_vertex, year_start, year_delta

def random_vertices(graph_sparse, n=500, min_deg=0):
    degs = np.zeros((graph.nVertices,), dtype=int)
    for edge in graph_sparse:
        degs[edge[0]] += 1
        degs[edge[1]] += 1

    return np.random.choice(np.where(degs > min_deg)[0], n)

def computeSubgraph(graph_sparse, vertices):
    full_idx0 = []
    full_idx1 = []
    for ii in range(len(vertices)):  # here we will find all indices of edges where the concept is either the first or the second vertex
        full_idx0 = np.concatenate((full_idx0, np.where(graph_sparse[:, 0] == vertices[ii])[0]))
        full_idx1 = np.concatenate((full_idx1, np.where(graph_sparse[:, 1] == vertices[ii])[0]))

    all_idx = np.array(list(set(full_idx0) & set(full_idx1)), dtype=int)
    subgraph_sparse = graph_sparse[all_idx]

    for ii in range(len(vertices)):
        subgraph_sparse[:, 0] = [ii if x == vertices[ii] else x for x in subgraph_sparse[:, 0]]
        subgraph_sparse[:, 1] = [ii if x == vertices[ii] else x for x in subgraph_sparse[:, 1]]

    return subgraph_sparse, len(vertices)

def create_graph(year):
    graph_sparse, unconnected_vertex, year_start, year_delta = load_data('CompetitionSet2017_3.pkl')
    day_delta = date(year, 12, 31) - date(1990, 1, 1)
    subgraph_sparse = graph_sparse[graph_sparse[:, 2] < day_delta.days]

    print(f'Graph for year {year} has {len(subgraph_sparse)} edges')


    return subgraph_sparse

def extract_graph(graph_sparse, year):
    day_delta = date(year, 12, 31) - date(1990, 1, 1)
    subgraph_sparse = graph_sparse[graph_sparse[:, 2] < day_delta.days]

    print(f'Graph for year {year} has {len(subgraph_sparse)} edges')


    return subgraph_sparse


def plot_graph(graph_sparse, nb_vertices):
    graph = sparse.csr_matrix((np.ones(len(graph_sparse)), (graph_sparse[:, 0], graph_sparse[:, 1])), shape=(nb_vertices, nb_vertices))
    graph = nx.from_scipy_sparse_matrix(graph, parallel_edges=False, create_using=None, edge_attribute='weight')
    nx.draw(graph, pos=nx.spring_layout(graph), node_size=20)
    plt.show()
