from tensorflow.python.ops.numpy_ops.np_math_ops import positive
from load_data import DataLoader
import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import matplotlib.pyplot as plt

# from gem.embedding.lle import LocallyLinearEmbedding
# from scipy import sparse
# import networkx as nx
from time import time


def buildSimpleLayer(embeddingDim, out_dim=None):
    if out_dim is None:
        out_dim = embeddingDim
    x_in = kl.Input(shape=(embeddingDim,))
    x = kl.Dense(out_dim)(x_in)
    x = kl.BatchNormalization()(x)
    # x = kl.Dropout(0.5)(x)
    x_out = x
    return tf.keras.Model(inputs=x_in, outputs=x_out)


def buildSimpleClassifier(embeddingDim):
    x_in = kl.Input(shape=(2, embeddingDim))
    x = kl.Flatten()(x_in)
    x = kl.Dense(128)(x)
    x = kl.ReLU()(x)
    x = kl.Dense(128)(x)
    x = kl.ReLU()(x)
    x_out = kl.Dense(1)(x)
    # x_out = tf.keras.activations.sigmoid(x)
    return tf.keras.Model(inputs=x_in, outputs=x_out)


class EmbeddingTrainer():

    def __init__(self, nVertices, embeddingDim, gnn, g):
        self.nVertices = nVertices
        self.embeddingDim = embeddingDim
        self.gnn = gnn

        self._initEmbeddings(g)
        self.predictor = buildSimpleClassifier(embeddingDim)

        # self.build()

        self.optimizer = tf.keras.optimizers.Adam(0.001)

    # A faire bien
    def _initEmbeddings(self, g):
        from graph_utils import eigenvectorCentrality, Graph
        self.embeddings = tf.constant(
            eigenvectorCentrality(Graph(2013), 200, self.embeddingDim),
            dtype=tf.float32
        )

    def build(self):
        for layer in self.gnn.layers:
            layer.build((None, self.embeddingDim))

    def forward(self, hk, pairs):
        return self.predictor(tf.gather(hk, pairs))

    def supervisedTrain(self, loader):
        losses = []
        accs = []
        iterations = []
        c = 0
        for i in range(10):
            print(f"\n######## Epoch {i} ########\n")
            top = time()
            for adjMatrix, degs, names, pairs, solutions in loader.yieldBatchSupervised():
                with tf.GradientTape() as tape:
                    # h0 = tf.gather(self.embeddings, names)
                    h0 = self.embeddings(names)
                    # trainable_variables_gnn = []
                    # for layers in self.gnn.layers:
                    #     for layer in layers:
                    #         trainable_variables_gnn.extend(layer.trainable_variables)
                    #     # trainable_variables_gnn.extend(layers.trainable_variables)
                    # tape2.watch(trainable_variables_gnn)
                    # tape3.watch(self.predictor.trainable_variables)
                    hk = self.gnn.forward(h0, adjMatrix, degs)
                    # hk = self.gnn.layers[0](tf.matmul(adjMatrix, h0))
                    y = self.forward(hk, pairs)
                    loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(solutions, y))

                # self.embeddings.scatter_add(tf.IndexedSlices(1e-3*tape1.gradient(loss, h0), names))
                # print(tape2.gradient(loss, trainable_variables_gnn))

                self.optimizer1.apply_gradients(zip(tape.gradient(loss, tape.watched_variables()), tape.watched_variables()))
                # self.optimizer1.apply_gradients(zip(tape1.gradient(loss, self.embeddings), self.embeddings))
                # self.optimizer2.apply_gradients(zip(tape2.gradient(loss, trainable_variables_gnn), trainable_variables_gnn))
                # self.optimizer3.apply_gradients(zip(tape3.gradient(loss, self.predictor.trainable_variables), self.predictor.trainable_variables))

                y_true = solutions.numpy()
                y_pred = np.squeeze(y.numpy())
                y_pred[y_pred > 0.5] = 1
                y_pred[y_pred <= 0.5] = 0
                acc = np.sum(y_true == y_pred)/y_true.shape[0]
                r = tf.reduce_sum(solutions)/solutions.shape[0]
                mean_loss = sum(losses[-100:])/100 if c > 110 else 0
                k = loader.nVertices-loader.batch_size//loader.batch_size
                print(f'{int(time()-top)}s  {c}/{k}   Loss: {loss.numpy():.2f}   Mean loss: {mean_loss:.2f}   Acc: {acc:.2f}   ratio links: {r:.2f}   shape matrix:   {names.shape[0]}', end='\r')

                losses.append(loss)
                accs.append(acc)
                iterations.append(c)
                c += 1
                # if c == 1000:
                # break
        plt.plot(iterations, losses)
        plt.plot(iterations, accs)
        plt.show()

    def UnsupervisedTrain(self, loader: DataLoader):
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        losses = []
        mean_losses = []
        iterations = []
        c = 0
        plt.figure("Loss")
        plt.ion()
        for i in range(1):
            print(f"\n######## Epoch {i} ########\n")
            top = time()
            for adjMatrix, degs, names, n1 in loader.yieldBatchUnsupervised():
                with tf.GradientTape() as tape:
                    h0 = tf.gather(self.embeddings, names)
                    # h0 = tf.math.l2_normalize(self.embeddings(names), 1)
                    hk = self.gnn.forward(h0, adjMatrix, degs)

                    adjMatrix = tf.slice(adjMatrix, [0, 0], [n1, n1])
                    hk = tf.slice(hk, [0, 0], [n1, self.embeddingDim])

                    reconstructedMatrix = tf.math.sigmoid(tf.matmul(hk, tf.transpose(hk)))

                    loss1 = - adjMatrix * tf.math.log(reconstructedMatrix+1e-2) / (tf.reduce_sum(adjMatrix)+1e-2)
                    loss2 = - (1 - adjMatrix - tf.eye(n1)) * tf.math.log(1 - reconstructedMatrix + 1e-2) / (tf.reduce_sum(1 - adjMatrix - tf.eye(n1)) + 1e-2)
                    loss = tf.reduce_sum(loss1 + loss2)

                self.optimizer.apply_gradients(zip(tape.gradient(loss, tape.watched_variables()), tape.watched_variables()))
                self.embeddings.assign(tf.math.l2_normalize(self.embeddings, 1))

                # sigma = tf.reduce_mean(tf.math.reduce_std(self.embeddings, axis=0))
                mean_loss = sum(losses[-100:])/100 if c > 110 else 0
                k = loader.nVertices-loader.batch_size//loader.batch_size
                print(f'{int(time()-top)}s  {c}/{k}  ',
                      f'nb edges: {tf.reduce_sum(adjMatrix)}  ',
                      f'Loss1: {tf.reduce_sum(loss1.numpy()):.5e}  ',
                      f'Loss2: {tf.reduce_sum(loss2.numpy()):.5e}  ',
                      f'Mean loss: {mean_loss:.5e}  ',
                      # f'Sigma: {sigma}  '
                      f'shape matrix: {names.shape[0]:5}\t', end='\r')

                losses.append(loss)
                mean_losses.append(mean_loss)
                iterations.append(c)
                c += 1
                # if c % 1000 == 0:
                #     plt.plot(iterations, losses)
                #     plt.plot(iterations[111:], mean_losses[111:])
                #     plt.show()
                if c == 15000:
                    break
                plt.plot(iterations, losses)
                plt.plot(iterations[111:], mean_losses[111:])
                plt.show()
                plt.pause(1e-4)

        self.gnn.save()
        np.save("Models/embeddings2.npy", self.embeddings.read_value().numpy())

    def UnsupervisedTrain2(self, epochs, loader: DataLoader, batch_size=16, verbose=0):
        losses = []
        mean_losses = []
        iterations = []
        c = 0
        if verbose == 1:
            plt.figure("Loss")
            plt.ion()
        for i in range(epochs):
            print(f"\n######## Epoch {i} ########\n")
            top = time()
            # generator = loader.yieldBatchUnsupervised()
            generator = loader.yieldBatchUnsupervised()
            for _ in range(64000):
                with tf.GradientTape() as tape:
                    loss = 0
                    loss1 = 0
                    loss2 = 0
                    for _ in range(batch_size):
                        adjMatrix, degs, names, n1 = next(generator)
                        h0 = tf.gather(self.embeddings, names)
                        hk = self.gnn.forward(h0, adjMatrix, degs)

                        adjMatrix = tf.slice(adjMatrix, [0, 0], [n1, n1])
                        hk = tf.slice(hk, [0, 0], [n1, self.embeddingDim])

                        reconstructedMatrix = tf.matmul(hk, tf.transpose(hk))

                        # l1 = - adjMatrix * tf.math.log(reconstructedMatrix+1e-3) / (tf.reduce_sum(adjMatrix)+1e-3)
                        # loss1 += tf.reduce_sum(l1)
                        # l2 = - (tf.ones(n1) - adjMatrix - tf.eye(n1)) * tf.math.log(tf.ones(n1) - reconstructedMatrix + 1e-2) / (tf.reduce_sum(tf.ones(n1) - adjMatrix - tf.eye(n1)) + 1e-2)
                        # loss2 += tf.reduce_sum(l2)
                        l1 = - adjMatrix * tf.math.log(reconstructedMatrix+1e-3)
                        l2 = - (tf.ones(n1) - adjMatrix - tf.eye(n1)) * tf.math.log(tf.ones(n1) - reconstructedMatrix + 1e-2)
                        loss1 += tf.reduce_sum(l1) / (tf.reduce_sum(adjMatrix)+1e-3)
                        loss2 += tf.reduce_sum(l2) / (tf.reduce_sum(tf.ones(n1) - adjMatrix - tf.eye(n1)) + 1e-2)
                        loss += tf.reduce_sum(l1 + l2) / (n1*n1)

                    # loss = loss1 + loss2
                    loss /= batch_size

                self.optimizer.apply_gradients(zip(tape.gradient(loss, tape.watched_variables()), tape.watched_variables()))

                sigma = tf.math.reduce_std(self.embeddings[:, 0])
                mean_loss = sum(losses[-100:])/100 if c > 110 else 0
                k = loader.nVertices-loader.batch_size//loader.batch_size
                print(f'{int(time()-top)}s  {c}/{k}  ',
                      f'nb edges: {tf.reduce_sum(adjMatrix)}  ',
                      f'Loss1: {tf.reduce_sum(loss1.numpy()):.5e}  ',
                      f'Loss2: {tf.reduce_sum(loss2.numpy()):.5e}  ',
                      f'Mean loss: {mean_loss:.5e}  ',
                      f'Sigma: {sigma:4e}  ',
                      f'n1: {n1}  ',
                      f'shape matrix: {names.shape[0]:5}\t', end='\r')

                losses.append(loss)
                mean_losses.append(mean_loss)
                iterations.append(c)
                c += 1
                # if c % 1000 == 0:
                #     plt.plot(iterations, losses)
                #     plt.plot(iterations[111:], mean_losses[111:])
                #     plt.show()
                if c == 2000:
                    break
                if verbose == 1:
                    plt.plot(iterations[10:], losses[10:])
                    plt.plot(iterations[111:], mean_losses[111:])
                    plt.show()
                    plt.pause(1e-4)

        self.gnn.save()
        np.save("Models/embeddings2.npy", self.embeddings.numpy())
        
        plt.figure("Loss")
        plt.plot(iterations, losses)
        plt.plot(iterations[111:], mean_losses[111:])
        plt.savefig("Models/loss.png")

    def UnsupervisedTrain3(self, epochs, loader: DataLoader, batch_size=16, verbose=0, Q=100):
        losses = []
        mean_losses = []
        iterations = []
        c = 0
        if verbose == 1:
            plt.figure("Loss")
            plt.ion()
        for i in range(epochs):
            print(f"\n######## Epoch {i} ########\n")
            top = time()
            generator = loader.yieldBatchUnsupervised2()
            for _ in range(64000):
                with tf.GradientTape() as tape:
                    loss = 0
                    loss1 = 0
                    loss2 = 0
                    for _ in range(batch_size):
                        adjMatrix, degs, names, n = next(generator)
                        h0 = tf.gather(self.embeddings, names)
                        hk = self.gnn.forward(h0, adjMatrix, degs)

                        u = tf.slice(hk, [0, 0], [loader.batch_size, self.embeddingDim])
                        nu = tf.slice(hk, [loader.batch_size, 0], [loader.batch_size, self.embeddingDim])
                        v = tf.slice(hk, [2*loader.batch_size, 0], [loader.batch_size, self.embeddingDim])

                        pos = tf.sigmoid(tf.reduce_sum(u * nu, -1))
                        neg = tf.sigmoid(-tf.reduce_sum(u * v, -1))
                       
                        loss1 += -tf.reduce_sum(tf.math.log(pos))
                        loss2 += -Q*tf.reduce_sum(tf.math.log(neg))

                    loss = loss1 + loss2
                    loss /= batch_size

                self.optimizer.apply_gradients(zip(tape.gradient(loss, tape.watched_variables()), tape.watched_variables()))
                self.embeddings.assign(tf.math.l2_normalize(self.embeddings, 1))

                sigma = tf.math.reduce_std(self.embeddings[:, 0])
                mean_loss = sum(losses[-100:])/100 if c > 110 else 0
                k = loader.nVertices-loader.batch_size//loader.batch_size
                print(f'{int(time()-top)}s  {c}/{k}  ',
                      f'nb edges: {tf.reduce_sum(adjMatrix)}  ',
                      f'Loss1: {tf.reduce_sum(loss1.numpy()):.5e}  ',
                      f'Loss2: {tf.reduce_sum(loss2.numpy()):.5e}  ',
                      f'Mean loss: {mean_loss:.5e}  ',
                      f'Sigma: {sigma:4e}  '
                      f'shape matrix: {names.shape[0]:5}\t', end='\r')

                losses.append(loss)
                mean_losses.append(mean_loss)
                iterations.append(c)
                c += 1
                # if c % 1000 == 0:
                #     plt.plot(iterations, losses)
                #     plt.plot(iterations[111:], mean_losses[111:])
                #     plt.show()
                if c == 2000:
                    break
                if verbose == 1:
                    plt.plot(iterations[10:], losses[10:])
                    plt.plot(iterations[111:], mean_losses[111:])
                    plt.show()
                    plt.pause(1e-4)

        self.gnn.save()
        np.save("Models/embeddings2.npy", self.embeddings.read_value().numpy())
        
        plt.figure("Loss")
        plt.plot(iterations, losses)
        plt.plot(iterations[111:], mean_losses[111:])
        plt.savefig("Models/loss.png")

    def load_embeddings(self, file='Models/embeddings.npy'):
        embeddings = np.load(file)
        self.embeddings = tf.constant(embeddings)
