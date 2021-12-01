import tensorflow as tf
import os


class GCN():

    def __init__(self, nVertices, embeddingDim, layers, activation):

        self.nVertices = nVertices
        self.embeddingDim = embeddingDim

        self.activation = activation
        self.layers = layers

    # @tf.function
    def forward(self, embeddingMatrix, adjacencyMatrix, degs):
        d12 = tf.math.rsqrt(degs+1)
        adjacencyMatrix += tf.eye(adjacencyMatrix.shape[0])
        adjacencyMatrix = adjacencyMatrix * d12
        adjacencyMatrix = adjacencyMatrix * tf.expand_dims(d12, -1)
        for layer in self.layers:
            embeddingMatrix = self.activation(layer(tf.matmul(adjacencyMatrix, embeddingMatrix)))
            embeddingMatrix = tf.math.l2_normalize(embeddingMatrix, 1)
        return embeddingMatrix

    def save(self):
        for i in range(len(self.layers)):
            self.layers[i].save(f"Models/GCN/layer{i}")


class GIN():

    def __init__(self, nVertices, embeddingDim, layers, activation):

        self.nVertices = nVertices
        self.embeddingDim = embeddingDim

        self.activation = activation
        self.layers = layers

    def forward(self, embeddingMatrix, adjacencyMatrix, degs):
        d12 = tf.math.rsqrt(degs+1)
        adjacencyMatrix = adjacencyMatrix * d12
        adjacencyMatrix = adjacencyMatrix * tf.expand_dims(d12, -1)
        for layer1, layer2, layer3 in self.layers:
            neigh_messages = self.activation(tf.matmul(adjacencyMatrix, layer1(embeddingMatrix)))
            self_messages = self.activation(layer2(embeddingMatrix))
            agg = tf.keras.backend.concatenate([neigh_messages, self_messages], axis=-1)
            embeddingMatrix = self.activation(layer3(agg))
            embeddingMatrix = tf.math.l2_normalize(embeddingMatrix, 1)
        return embeddingMatrix

    def save(self):
        for i in range(len(self.layers)):
            layer1, layer2, layer3 = self.layers[i]
            layer1.save(f"Models/GIN/layer{i}1")
            layer2.save(f"Models/GIN/layer{i}2")
            layer3.save(f"Models/GIN/layer{i}3")

    def load(self):
        nb_layers = len(os.listdir("Models/GIN")) // 3
        layers = []
        for i in range(nb_layers):
            layer1 = tf.keras.models.load_model(f"Models/GIN/layer{i}1")
            layer2 = tf.keras.models.load_model(f"Models/GIN/layer{i}2")
            layer3 = tf.keras.models.load_model(f"Models/GIN/layer{i}3")
            layers.append((layer1, layer2, layer3))
        self.layers = layers
