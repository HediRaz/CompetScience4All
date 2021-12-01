import graph
from graph.training import unsupervised_training_multiple, supervised_training_multiple_with_test
from graph.gcn import GCN, GIN
from graph.load_data import unsupervised_dataLoader, supervised_data_loader_with_test
from graph.classifier import Classifier_Dense, Classifier_RNN
from graph.config import Config
from graph.auc_loss import ROC_LOSS, ROC_STAR_LOSS

import torch
import wandb

import argparse

# Structure
config = {

    # Structure
    "gnn_type" : GIN,
    "classifier_type" : Classifier_RNN,

    # Embedding dims and GCN
    "init_embedding_method" : 'eigen2017',
    "gnn_hidden_dims" : [32, 64, 64],

    # Classifier
    "classifier_hidden_dims" : [None, 64, 512, 256, 128, 64, 32, 2],
    "classifier_activation" : "Relu",
    

    # Unsupervised training
    "years" : [y for y in range(2008, 2015)],
    "min_deg_unsupervised": 5,
    "iter_number_unsupervised" : 200,
    "max_neighbors_unsupervised" : 5,
    "k_init" : 1,
    "starting_size" : 8,
    "batch_size_unsupervised" : 5,

    "optimizer_algo_unsupervised" : "Adam",
    "lr_unsupervised": 5e-4,

    # Config Supervised Training loader
    # "delta_years": 3,
    # "years_min": 2000,
    "years_max": 2011,
    # "max_neighbors_supervised" : 10000,
    # "pairs_batch_size" : 8192,
    "proportion_unconnected": 4,
    "min_degs_dataset": 10,
    "test_size": 200_000,
    "test_rate": 5,

    # Supervised training
    "add_diag": False,
    # "normalize_adjacency_matrix": True,
    # "batch_size_supervised" : 2,
    "loss_fn" : "CrossEntropy",
    "iter_number_supervised": 1500,

    # Supervised optimizer
    "optimizer_algo_supervised_gnn" : "Adam",
    "optimizer_algo_supervised_classifier" : "Adam"
    # "lr_supervised_gnn": 1e-3,
    # "lr_supervised_classifier": 1e-3,
}


# Hyperparameters
sweeps_config = {

    "sweep_name": None,

    # Config Supervised Training loader
    "delta_years": None,
    "years_min": None,
    "max_neighbors_supervised" : None,
    "normalize_adjacency_matrix": None,

    "pairs_batch_size" : None,
    "batch_size_supervised": None,

    # Supervised optimizer
    "lr_supervised_gnn": None,
    "lr_supervised_classifier": None

}

sweeps_config_type = {
    
    "sweep_name": str,

    # Config Supervised Training loader
    "delta_years": int,
    "years_min": int,
    "max_neighbors_supervised" : int,
    "normalize_adjacency_matrix": lambda x: bool(int(x)),

    "pairs_batch_size" : int,
    "batch_size_supervised": int,

    # Supervised optimizer
    "lr_supervised_gnn": float,
    "lr_supervised_classifier": float

}


parser = argparse.ArgumentParser()

for key in sweeps_config.keys():
    parser.add_argument("--" + key)

args = parser.parse_args()

sweeps_config = vars(args)

for (key, value) in sweeps_config.items():
    sweeps_config[key] = sweeps_config_type[key](value)

config.update(sweeps_config)

# Wandb
log = True
# Interactive display
display= False

if log:
    wandb.init(project=config["sweep_name"], entity="argocs", config=config)
    wandb.define_metric("unsupervised step")
    wandb.define_metric("supervised step")
    wandb.define_metric("loss GNN", step_metric="unsupervised step")
    wandb.define_metric("train AUC", step_metric="supervised step")
    wandb.define_metric("test AUC", step_metric="supervised step")
    wandb.define_metric("train loss classifier", step_metric="supervised step")
    wandb.define_metric("test loss classifier", step_metric="supervised step")

config = Config(config)

depth = len(config.gnn_hidden_dims) -1

## Models
gnn = config.gnn_type(config.gnn_hidden_dims).to(graph.device)

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
loader_unsupervised = unsupervised_dataLoader(
    config.years,
    config.min_deg_unsupervised
    )

initial_embedding = loader_unsupervised.init_nodes_embedding(config.init_embedding_method, config.gnn_hidden_dims[0]).to(graph.device)

loader_supervised = supervised_data_loader_with_test(
    delta_years=config.delta_years,
    years_min=config.years_min,
    years_max=config.years_max,
    minimal_degs=config.min_degs_dataset,
    p=config.proportion_unconnected,
    max_neighbors=config.max_neighbors_supervised,
    k=2,
    pairs_batch_size=config.pairs_batch_size,
    add_diag=config.add_diag,
    normalize_adj_m=config.normalize_adjacency_matrix,
    test_size=config.test_size
)


unsupervised_training_multiple(
    epochs=1,
    loader=loader_unsupervised,
    initial_embedding=initial_embedding,
    model=gnn,
    optimizer=unsupervised_optimizer,
    batch_size=config.batch_size_unsupervised,
    iter_number=config.iter_number_unsupervised,
    max_neighbors=config.max_neighbors_unsupervised,
    k_init=config.k_init,
    k_max=depth,
    starting_size=config.starting_size,
    display=display,
    log=log
    )

initial_embedding = initial_embedding[0]
supervised_training_multiple_with_test(
    epochs=1,
    iter_number=config.iter_number_supervised,
    loader=loader_supervised,
    initial_embedding=initial_embedding,
    gnn=gnn,
    classifier=classifier,
    optimizer_gnn=optimizer_gnn,
    optimizer_classifier=optimizer_classifier,
    loss_fn=loss_fn,
    batch_size=config.batch_size_supervised,
    test_rate=config.test_rate,
    log=log,
    early_stop=True
)
