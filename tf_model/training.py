from load_data import DataLoader
from gcn import GCN, GIN
from model import EmbeddingTrainer, buildSimpleLayer

import tensorflow as tf


if __name__ == '__main__':
    embeddingDim = 32

    loader = DataLoader(2013, 2014, 4, k=3, minimal_degs=4, max_neighbors=8, p=2)

    layer11 = buildSimpleLayer(embeddingDim)
    layer12 = buildSimpleLayer(embeddingDim)
    layer13 = buildSimpleLayer(2*embeddingDim, embeddingDim)
    layer21 = buildSimpleLayer(embeddingDim)
    layer22 = buildSimpleLayer(embeddingDim)
    layer23 = buildSimpleLayer(2*embeddingDim, embeddingDim)
    layers = [(layer11, layer12, layer13), (layer21, layer22, layer23)]
    gnn = GIN(64719, embeddingDim, layers, tf.keras.activations.sigmoid)

    # layer1 = buildSimpleLayer(embeddingDim)
    # layer2 = buildSimpleLayer(embeddingDim)
    # layers = [layer1, layer2]
    # gnn = GCN(64719, embeddingDim, layers, tf.keras.activations.sigmoid)

    classifier = EmbeddingTrainer(64719, embeddingDim, gnn, loader.graph_sparse)
    # classifier.load_embeddings(trainable=True)
    # classifier.gnn.load()

    print("######## Train ########")
    # classifier.load_embeddings('Models/embeddings2.npy', True)
    classifier.UnsupervisedTrain2(1, loader, batch_size=1, verbose=1)
