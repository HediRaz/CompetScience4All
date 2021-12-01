import numpy as np
import graph
import torch
import json
import os
from graph.data_utils import load_data
from graph.graph_utils import light_Graph, sparseAdjacencyTensor
from graph.load_data import supervised_data_loader_with_test, make_matrix, supervised_data_loader_with_test_all_vertices
from graph.training import pass_forward, pass_forward_all
from tqdm import tqdm


def save_models(h0, gnn, classifier, name=None):
    if name is None:
        name = len(os.listdir("Models"))
    os.mkdir(f"Models/{name}")
    torch.save(h0, f"Models/{name}/h0.pt")
    torch.save(gnn.state_dict(), f"Models/{name}/gnn.pth")
    torch.save(classifier.state_dict(), f"Models/{name}/classifier.pth")


def compute_save_submission(m, h0, gnn, classifier, filename="Models/model_all_idx2017_3.json"):
    # Load dataset
    _, unconnected_pairs, _, _ = load_data("CompetitionSet2017_3.pkl")
    # unconnected_pairs = np.array(unconnected_pairs, dtype=int)
    unconnected_pairs = torch.tensor(unconnected_pairs, dtype=torch.int64)

    # gnn.eval()
    classifier.eval()

    res = gnn.forward(m, h0[0])
    res = res[unconnected_pairs]
    res = classifier.forward(res)
    res = torch.softmax(res, -1)
    res = res[:, 1]

    torch.save(res, filename)


def test_compute_save_submission(h0, gnn, classifier, delta_years, k, filename, test_dataset, add_diag=False, normailize_adj_m=True):
    gnn.train()
    classifier.eval()

    # Data loader
    loader = supervised_data_loader_with_test(
        delta_years=delta_years,
        years_min=2001,
        years_max=2000,
        test_size=8,
        add_diag=add_diag,
        normalize_adj_m=normailize_adj_m,
        max_neighbors=65000
    )

    loader.graphs_test = [light_Graph(y) for y in range(2014-delta_years+1, 2014+1)]
    # unconnected_pairs_split = np.split(unconnected_pairs, 10)
    # unconnected_pairs_preds = []
    # for pairs in tqdm(unconnected_pairs_split):
    sub_generator = loader.subGenerator(test_dataset, [], loader.graphs_test)

    with torch.no_grad():
        preds, _ = pass_forward(
            pairs_batch_size=test_dataset.shape[0],
            sub_generator=sub_generator,
            h0=h0,
            gnn=gnn,
            classifier=classifier,
            split=True
            )
        preds = torch.softmax(preds, -1)[:, 1].to("cpu")
    torch.save(preds, filename)


def compute_save_submission_all(loader:supervised_data_loader_with_test_all_vertices, h0, gnn, classifier, filename="Models/model_all_idx2017_3.json"):

    # gnn.eval()
    gnn.train()
    classifier.eval()

    # Load dataset
    _, unconnected_pairs, _, _ = load_data("CompetitionSet2017_3.pkl")
    # unconnected_pairs = np.array(unconnected_pairs, dtype=int)
    unconnected_pairs = np.array(unconnected_pairs, dtype=int)

    matrices_submit = [
            make_matrix(light_Graph(2015, 0), add_diag=False, normalize_matrix=True).to("cpu"),
            make_matrix(light_Graph(2016, 0), add_diag=False, normalize_matrix=True).to("cpu"),
            make_matrix(light_Graph(2017, 0), add_diag=False, normalize_matrix=True).to("cpu")
        ]
    sub_generator = loader.subGenerator(unconnected_pairs, [], matrices_submit)

    with torch.no_grad():
        preds, _ = pass_forward_all(
            pairs_batch_size=unconnected_pairs.shape[0],
            sub_generator=sub_generator,
            h0=h0,
            gnn=gnn,
            classifier=classifier,
            split=False
            )
        preds = torch.softmax(preds, -1)[:, 1].to("cpu")
    torch.save(preds, filename)