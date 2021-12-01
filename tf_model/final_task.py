import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from load_data import DataLoader
from gcn import GCN, GIN
from model import EmbeddingTrainer

import matplotlib.pyplot as plt

class FF_network(tf.keras.Model):

    def __init__(self) -> None:
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(128)
        self.dense3 = tf.keras.layers.Dense(64)
        self.dense4 = tf.keras.layers.Dense(1)

    
    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dense2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dense3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dense4(x)
        
        return tf.keras.activations.sigmoid(x)

def training_step(ff_network, gcn, batch_size, names, pairs, adj_matrix, degs, label, loss_function, optimizer, gradient_corrections, sensibility=0.5):
    """
    batch = [[0,1], ...] = [v1,v2,v3,v4] (names)
    names, pairs
    
    
    """

    embedding = tf.gather(gcn.embeddings, names)
    embedding = tf.gather(gcn.gnn.forward(embedding, adj_matrix, degs), pairs)

    with tf.GradientTape() as tape:

        prediction = ff_network.call(embedding)
        loss = tf.reduce_sum(loss_function(label, prediction)) / batch_size

    grads = tape.gradient(loss, ff_network.trainable_variables)

    grads = [gradient_corrections(grad) for grad in grads]

    optimizer.apply_gradients(zip(grads, ff_network.trainable_variables))

    prediction = tf.map_fn(lambda x: 1 if x > 0.5 else 0, prediction)

    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, label), tf.float32))
    nb_p = tf.reduce_sum(tf.cast(tf.equal(label, 1), tf.float32))
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction, 1), tf.equal(label, 1)), tf.float32)) / nb_p
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(prediction, 0), tf.equal(label, 1)), tf.float32)) / nb_p

    return loss, acc, tp, fn


def training(epochs, dataloader, ff_network, gcn, loss_function, optimizer, gradient_corrections):

    losses = []
    batch_num = []
    moy = []

    plt.figure('Dessin')
    plt.ion()

    for e in range(epochs):
        for adj_matrix, degs, names, pairs, solutions in dataloader.yieldBatchSupervised():
            l, acc, tp, fn = training_step(ff_network, gcn, dataloader.batch_size, names, pairs, adj_matrix, degs, solutions, loss_function, optimizer, gradient_corrections)
            print(f"Epochs : {e}, loss on batch : {l}, acc : {acc}, tp : {tp}, fn : {fn}")
            batch_num.append(0 if len(batch_num) == 0 else batch_num[-1] +1)
            losses.append(l)
            moy.append(l if len(losses) < 20 else sum(losses[-20:])/20)

            plt.clf()
            plt.plot(batch_num, losses, color="blue")
            plt.plot(batch_num, moy, color="red")
            plt.show()
            plt.pause(1e-3)
            


    return

if __name__ == "__main__":

    nVertices = 64719

    epochs = 10
    embedding_size = 32
    dataloader = DataLoader(2013, 2014, 8, ponderate=False, p=4, minimal_degs=4, k=2, max_neighbors=10)
    ff_network = FF_network()

    # layers = [tf.keras.models.load_model("./Models/GCN/layer0"), tf.keras.models.load_model("./Models/GCN/layer1")]
    # gnn = GCN(nVertices, embedding_size, layers, tf.keras.activations.sigmoid)
    layers = None
    gnn = GIN(nVertices, embedding_size, layers, tf.keras.activations.sigmoid)
    gnn.load()

    gcn = EmbeddingTrainer(64719, embedding_size, gnn, None)
    gcn.load_embeddings('Models/embeddings2.npy')

    loss_function = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.optimizers.Adam()

    gradient_corrections = lambda x:x


    training(epochs, dataloader, ff_network, gcn, loss_function, optimizer, gradient_corrections)

