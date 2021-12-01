import torch
import wandb
import os

import graph
from graph.submission import save_models, compute_save_submission
from graph.training import supervised_training_multiple_with_test, unsupervised_train, supervised_train, unsupervised_training_multiple, supervised_training_multiple
from graph.gcn import GCN, GAT, GIN
from graph.load_data import supervised_data_loader_with_test, unsupervised_dataLoader, supervised_dataLoader
from graph.classifier import Classifier_Dense, Classifier_RNN
from graph.config import Config
from graph.auc_loss import ROC_LOSS, ROC_STAR_LOSS


config = {

    # Structure
    "gnn_type" : GIN,
    "classifier_type" : Classifier_Dense,

    # Embedding dims and GCN
    "init_embedding_method" : 'eigen2017',
    "gnn_hidden_dims" : [32, 64, 128],

    # Classifier
    "classifier_hidden_dims" : [None, 1024, 512, 256, 128, 64, 32, 2],
    "classifier_activation" : "Relu",

    "years" : [y for y in range(2008, 2015)],

    # Config Unsupervised Training loader
    "min_deg_unsupervised" : 5,

    # Generator unsupervised
    "iter_number_unsupervised" : 10,
    "max_neighbors_unsupervised" : 5,
    "k_init" : 1,
    "starting_size" : 8,

    # Unsupervised training
    "batch_size_unsupervised" : 5,

    # Unsupervised optimizer
    "optimizer_algo_unsupervised" : "Adam",
    "lr_unsupervised": 5e-4,

    # Config Supervised Training loader
    "delta_years": 3,
    "years_min": 2007,
    "years_max": 2014,
    "max_neighbors_supervised" : 65000,
    "pairs_batch_size" : 2**6,
    "proportion_unconnected": 4,
    "min_degs_dataset": 10,
    "test_size": 32,
    "test_rate": 5,

    # Supervised training
    "add_diag": False,
    "normalize_adjacency_matrix": True,
    "batch_size_supervised" : 5,
    "loss_fn" : "CrossEntropy",
    "iter_number_supervised": 30,

    # Supervised optimizer
    "optimizer_algo_supervised_gnn" : "Adam",
    "optimizer_algo_supervised_classifier" : "Adam",
    "lr_supervised_gnn": 5e-4,
    "lr_supervised_classifier": 1e-3

}

# Wandb
log = True
# Interactive display
display= False
# Save models and predictions
save = True
compute_predictions = True

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


# loader_supervised = supervised_dataLoader(
#     config.years,
#     finish_year=2017,
#     max_neighbors=config.max_neighbors_supervised,
#     k=depth, 
#     batch_size=config.pairs_batch_size,
#     min_degs=config.min_degs_dataset,
#     graphs=loader_unsupervised.graphs
#     )
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


# supervised_training_multiple(
#     epochs=1,
#     iter_number=config.iter_number_supervised,
#     loader=loader_supervised,
#     initial_embedding=initial_embedding,
#     gnn=gnn,
#     classifier=classifier,
#     optimizer_classifier=optimizer_classifier,
#     optimizer_gnn=optimizer_gnn,
#     batch_size=config.batch_size_supervised,
#     loss_fn=loss_fn,
#     display=display,
#     log=log
# )
initial_embedding = initial_embedding[0]
supervised_training_multiple_with_test(
    epochs=2,
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
    log=log
)

# # Optimizer classification
# optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=config.lr_supervised_gnn/5)
# optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=config.lr_supervised_classifier/5)

# supervised_training_multiple_with_test(
#     epochs=1,
#     iter_number=config.iter_number_supervised,
#     loader=loader_supervised,
#     initial_embedding=initial_embedding,
#     gnn=gnn,
#     classifier=classifier,
#     optimizer_gnn=optimizer_gnn,
#     optimizer_classifier=optimizer_classifier,
#     loss_fn=loss_fn,
#     batch_size=config.batch_size_supervised,
#     test_rate=config.test_rate,
#     log=log
# )

# # Optimizer classification
# optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=config.lr_supervised_gnn/20)
# optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=config.lr_supervised_classifier/20)

# supervised_training_multiple_with_test(
#     epochs=1,
#     iter_number=config.iter_number_supervised,
#     loader=loader_supervised,
#     initial_embedding=initial_embedding,
#     gnn=gnn,
#     classifier=classifier,
#     optimizer_gnn=optimizer_gnn,
#     optimizer_classifier=optimizer_classifier,
#     loss_fn=loss_fn,
#     batch_size=config.batch_size_supervised,
#     test_rate=config.test_rate,
#     log=log
# )

c = len(os.listdir("Models"))

if save:
    save_models(initial_embedding, gnn, classifier, name=c)

if compute_predictions:
    compute_save_submission(initial_embedding, gnn, classifier, config.delta_years, 2, filename=f"Models/{c}/preds.pt")
