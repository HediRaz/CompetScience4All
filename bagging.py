import torch
import os

import graph
from graph.submission import save_models, compute_save_submission
from graph.training import ultimate_unsupervised_training, ultimate_supervised_training, ultimate_unsupervised_training2, ultimate_supervised_training3
from graph.gcn import GCN, GAT, GIN, DenseBlock, GIN_shared_weights, GIN_GRU
from graph.load_data import make_matrix, ultimate_dataloader, init_nodes_embedding 
from graph.classifier import Classifier_Dense, Classifier_RNN, style_Dense
from graph.config import Config
from graph.auc_loss import ROC_LOSS, ROC_STAR_LOSS
from graph.graph_utils import graph_ultimate



config = {

    # Structure
    "gnn_type" : GIN,
    "classifier_type" : Classifier_Dense,
    "depth_dense_block": 1,

    # Embedding dims and GCN
    "init_embedding_method" : 'eigen2017',
    "gnn_hidden_dims" : [24,32,32],

    # Classifier
    "classifier_hidden_dims" : [None, 64, 32, 2],
    "classifier_activation" : "Relu",

    "unsupervised_training_years" : [y for y in range(2011-5+1, 2011+1)],

    # Config Unsupervised Training loader
    "minimal_degs_unsupervised" : 5,

    # Generator unsupervised
    "iter_number_unsupervised" : 250,
    "max_neighbors_unsupervised" : 5,
    "k_init" : 1,
    "starting_size" : 32,

    # Unsupervised training
    "epochs_supervised": 1,
    "batch_size_unsupervised" : 1,

    # Unsupervised optimizer
    "optimizer_algo_unsupervised" : "Adam",
    "lr_unsupervised": 5e-4,

    # Config Supervised Training loader
    "delta_years": 1,
    "min_year": 2014,
    "batch_size_supervised" : 1,
    "pairs_batch_size": 2**15,
    "proportion_dataset": 2,
    "minimal_degs_supervised": 5,
    "test_rate": 1,
    "max_size_dataset": 1_000_000_000,

    # Supervised training
    "add_diag": False,
    "normalize_adj_matrix": True,
    "loss_fn" : "CrossEntropy",
    "iter_number_supervised": 300,

    # Supervised optimizer
    "optimizer_algo_supervised_gnn" : "Adam",
    "optimizer_algo_supervised_classifier" : "Adam",
    "lr_supervised_gnn": 1e-3,
    "lr_supervised_classifier": 1e-3,
    
    "drop": 0.3,
    "batch_norm": False,


    "early_stop":False

}


config = Config(config)

depth = len(config.gnn_hidden_dims) - 1


loader = ultimate_dataloader(   min_year=config.min_year,
                                proportion_dataset=config.proportion_dataset,
                                minimal_degs_unsupervised=config.minimal_degs_unsupervised,
                                minimal_degs_supervised=config.minimal_degs_supervised,
                                add_diag=config.add_diag,
                                normalize_adj_matrix=config.normalize_adj_matrix,
                                max_size=config.max_size_dataset
)


graph_sparse, _, _, _ = graph.data_utils.load_data('CompetitionSet2017_3.pkl')
# -- extract every graph
graphs_years = [graph.data_utils.extract_graph(graph_sparse, year) for year in range(config.min_year, 2014 + 1)]

initial_embedding = init_nodes_embedding(config.init_embedding_method, config.gnn_hidden_dims[0], graphs_years, config.minimal_degs_unsupervised, normalize=True).to(graph.device)

del graphs_years
del _


g_2017 = graph_ultimate(graph_sparse, config.minimal_degs_unsupervised)
matrice_2017 = make_matrix(g_2017, config.add_diag, config.normalize_adj_matrix)

del graph_sparse
while True:
    gnn = config.gnn_type(config.gnn_hidden_dims, config.depth_dense_block).to(graph.device)

    # Optimizer GNN
    unsupervised_optimizer = torch.optim.Adam(gnn.parameters(), lr=config.lr_unsupervised)


    input_dim_classifier = config.gnn_hidden_dims[-1] * config.delta_years * 2
    config.classifier_hidden_dims[0] = input_dim_classifier

    if config.classifier_activation == "Relu":
        activation_type = torch.nn.ReLU
    elif config.classifier_activation == "PRelu":
        activation_type = torch.nn.PReLU
    elif config.classifier_activation == "Gelu":
        activation_type = torch.nn.GELU
    else:
        print(f"The following activation is not implemented: {config.classifier_activation}")
        exit()

    if config.classifier_type == Classifier_Dense:

        classifier = config.classifier_type(config.classifier_hidden_dims, activation_type).to(graph.device)

    if config.classifier_type == Classifier_RNN:
        classifier = config.classifier_type(config.classifier_hidden_dims, activation_type, config.gnn_hidden_dims[-1]).to(graph.device)

    if config.classifier_type == style_Dense:
        classifier = config.classifier_type(config.classifier_hidden_dims, activation_type, config.drop, config.batch_norm).to(graph.device)

    # Loss function
    if config.loss_fn == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif config.loss_fn == "ROC_loss":
        loss_fn = ROC_LOSS(2048, 2048)
    elif config.loss_fn == "ROC_star_loss":
        loss_fn = ROC_STAR_LOSS(2048, 2048, 0.4)

    # Optimizer classification
    optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=config.lr_supervised_gnn)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=config.lr_supervised_classifier)


    ultimate_supervised_training3(  epochs=config.epochs_supervised,
                                    loader=loader,
                                    initial_embedding=initial_embedding,
                                    gnn=gnn,
                                    classifier=classifier,
                                    iter_number=2_000_000,
                                    optimizer_gnn=optimizer_gnn,
                                    optimizer_classifier=optimizer_classifier,
                                    loss_fn=loss_fn,
                                    pairs_batch_size=config.pairs_batch_size,
                                    batch_size=config.batch_size_supervised,
                                    delta_years=config.delta_years,
                                    test_rate=config.test_rate,
                                    early_stop=config.early_stop)



    c = len(os.listdir("Models"))

    save_models(initial_embedding, gnn, classifier, name=c)

    compute_save_submission(matrice_2017, initial_embedding, gnn, classifier, filename=f"Models/{c}/preds.pt")
