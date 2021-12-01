import graph
import torch
from graph import load_data
from graph.submission import compute_save_submission, test_compute_save_submission
from graph.load_data import create_dataset_from_to
from sklearn import metrics
import json
import numpy as np

h0 = torch.load("./Models/0/h0.pt")
gnn = torch.load("./Models/0/gnn")
classifier = torch.load("./Models/0/classifier")


def test_submission(h0, gnn, classifier):

    _, unconnected_vertex_pairs, unconnected_vertex_pairs_solution = create_dataset_from_to(starting_year=2014, finish_year=2017, p=4, minimal_degs=10)
    del _

    unconnected_vertex_pairs = np.array(unconnected_vertex_pairs, dtype=np.int64)
    unconnected_vertex_pairs_solution = np.array(unconnected_vertex_pairs_solution, dtype=np.int64)

    shuffle = np.arange(unconnected_vertex_pairs_solution.shape[0])
    np.random.shuffle(shuffle)

    unconnected_vertex_pairs_solution = unconnected_vertex_pairs_solution[shuffle]
    unconnected_vertex_pairs = unconnected_vertex_pairs[shuffle]

    unconnected_vertex_pairs = unconnected_vertex_pairs[:250_000]
    unconnected_vertex_pairs_solution = unconnected_vertex_pairs_solution[:250_000]

    # -- compute prediction
    test_compute_save_submission(h0, gnn, classifier, delta_years=3, k=2, filename="./testing/preds_nosort.pt", test_dataset=unconnected_vertex_pairs)

    # --load prediction

    preds = torch.load("./testing/preds_nosort.pt").to('cpu').detach().numpy()

    auc = metrics.roc_auc_score(unconnected_vertex_pairs_solution, preds)

    print(f"AUC of prediction before sort: {auc}")
