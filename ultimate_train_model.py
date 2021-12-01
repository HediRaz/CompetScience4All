import torch
import numpy as np
import wandb
import os

import graph
from graph.submission import save_models, compute_save_submission
from graph.training import ultimate_unsupervised_training, ultimate_supervised_training, ultimate_unsupervised_training2 
from graph.gcn import GCN, GAT, GIN
from graph.load_data import ultimate_dataloader, init_nodes_embedding 
from graph.classifier import Classifier_Dense, Classifier_RNN
from graph.config import Config
from graph.auc_loss import ROC_LOSS, ROC_STAR_LOSS


config = {

    # Structure
    "gnn_type" : GIN,
    "classifier_type" : Classifier_Dense,
    "depth_dense_block": 3,

    # Embedding dims and GCN
    "init_embedding_method" : 'eigen2017',
    "gnn_hidden_dims" : [16, 32, 16],

    # Classifier
    "classifier_hidden_dims" : [None, 256, 128, 64, 32, 2],
    "classifier_activation" : "Relu",

    "unsupervised_training_years" : [y for y in range(2010, 2015)],

    # Config Unsupervised Training loader
    "minimal_degs_unsupervised" : 5,

    # Generator unsupervised
    "iter_number_unsupervised" : 100,
    "max_neighbors_unsupervised" : 5,
    "k_init" : 1,
    "starting_size" : 8,

    # Unsupervised training
    "epochs_supervised": 1,
    "batch_size_unsupervised" : 5,

    # Unsupervised optimizer
    "optimizer_algo_unsupervised" : "Adam",
    "lr_unsupervised": 1e-4,

    # Config Supervised Training loader
    "delta_years": 1,
    "min_year": 2011,
    "batch_size_supervised" : 2,
    "pairs_batch_size": 4096,
    "proportion_dataset": 4,
    "minimal_degs_supervised": 10,
    "test_rate": 5,
    "max_size_dataset":500_000,

    # Supervised training
    "add_diag": False,
    "normalize_adj_matrix": True,
    "loss_fn" : "CrossEntropy",
    "iter_number_supervised": 500,

    # Supervised optimizer
    "optimizer_algo_supervised_gnn" : "Adam",
    "optimizer_algo_supervised_classifier" : "Adam",
    "lr_supervised_gnn": 5e-4,
    "lr_supervised_classifier": 1e-3,


    "early_stop":False

}

# Wandb
log = True
# Interactive display
display= False
# Save models and predictions
save = False
compute_predictions = False

if log:
    wandb.init(project="HyperparametersTesting", entity="argocs", config=config)
    wandb.define_metric("unsupervised step")
    wandb.define_metric("supervised step")
    wandb.define_metric("loss GNN", step_metric="unsupervised step")
    wandb.define_metric("train AUC", step_metric="supervised step")
    wandb.define_metric("test AUC", step_metric="supervised step")
    wandb.define_metric("train loss classifier", step_metric="supervised step")
    wandb.define_metric("test loss classifier", step_metric="supervised step")

config = Config(config)

depth = len(config.gnn_hidden_dims) - 1

## Models
gnn = config.gnn_type(config.gnn_hidden_dims, config.depth_dense_block).to(graph.device)

# Optimizer GNN
unsupervised_optimizer = torch.optim.Adam(gnn.parameters(), lr=config.lr_unsupervised)


input_dim_classifier = config.gnn_hidden_dims[-1] * config.delta_years * 2
config.classifier_hidden_dims[0] = input_dim_classifier

if config.classifier_activation == "Relu":
    activation_type = torch.nn.ReLU
else:
    print(f"The following activation is not implemented: {config.classifier_activation}")
    exit()

if config.classifier_type == Classifier_Dense:

    classifier = config.classifier_type(config.classifier_hidden_dims, activation_type).to(graph.device)
    
if config.classifier_type == Classifier_RNN:
    classifier = config.classifier_type(config.classifier_hidden_dims, activation_type, config.gnn_hidden_dims[-1]).to(graph.device)

# Loss function
if config.loss_fn == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
elif config.loss_fn == "ROC_loss":
    loss_fn = ROC_LOSS(4096, 4096)
elif config.loss_fn == "ROC_star_loss":
    loss_fn = ROC_STAR_LOSS(4096, 4096, 0.4)

# Optimizer classification
optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=config.lr_supervised_gnn)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=config.lr_supervised_classifier)

## Loaders

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

del graph_sparse
del graphs_years
del _

# ultimate_unsupervised_training( epochs=1,
#                                 loader=loader,
#                                 initial_embedding=initial_embedding,
#                                 model=gnn,
#                                 optimizer=optimizer_gnn,
#                                 batch_size=config.batch_size_unsupervised,
#                                 iter_number=config.iter_number_unsupervised,
#                                 max_neighbors=config.max_neighbors_unsupervised,
#                                 k_init=config.k_init,
#                                 k_max=depth,
#                                 starting_size=config.starting_size,
#                                 display=display,
#                                 log=log,
#                                 years=config.unsupervised_training_years)

ultimate_unsupervised_training2(
    epochs=1,
    loader=loader,
    initial_embedding=initial_embedding,
    model=gnn,
    optimizer=unsupervised_optimizer,
    batch_size=1,
    iter_number=300,
    starting_size=256,
    nb_neighbors=1,
    nb_non_neighbors=5,
    years=np.array([2011]),
    log=True
)

ultimate_supervised_training(   epochs=config.epochs_supervised,
                                loader=loader,
                                initial_embedding=initial_embedding,
                                gnn=gnn,
                                classifier=classifier,
                                optimizer_gnn=optimizer_gnn,
                                optimizer_classifier=optimizer_classifier,
                                loss_fn=loss_fn,
                                pairs_batch_size=config.pairs_batch_size,
                                batch_size=config.batch_size_supervised,
                                delta_years=config.delta_years,
                                test_rate=config.test_rate,
                                log=log,
                                early_stop=config.early_stop)


c = len(os.listdir("Models"))

if save:
    save_models(initial_embedding, gnn, classifier, name=c)

if compute_predictions:
    compute_save_submission(initial_embedding, gnn, classifier, config.delta_years, 2, filename=f"Models/{c}/preds.pt")
