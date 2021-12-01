import graph
import torch
import graph.load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class DirectClassifier(torch.nn.Module):

    def __init__(self):
        super(DirectClassifier, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(42, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(-1)
        )

    def forward(self, x1, x2):
        x = torch.cat((self.flatten(x1), x2), 1)
        logits = self.linear_relu_stack(x)
        return logits

model = DirectClassifier().to(graph.device)
print(model)


def train(datagenerator, model, node_features):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    aucs = []
    iterations = []
    # plt.figure("loss")
    # plt.ion()
    c = 0
    for _, _, _, batch_pairs, batch_labels, batch_pairs_features in datagenerator:
        c += 1
        batch_pairs = batch_pairs.to(graph.device)
        batch_labels = batch_labels.to(graph.device)
        batch_pairs_features = batch_pairs_features.to(graph.device)

        X1 = node_features[batch_pairs]
        X2 = batch_pairs_features
        y = batch_labels

        # Compute prediction error
        pred = model(X1, X2)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        iterations.append(c)

        if c % 50 == 0:
            # plt.plot(iterations, losses, color='black')
            # plt.show()
            # plt.pause(1e-4)
            loss, current = loss.item(), c
            pred = torch.squeeze(pred[:, 1])
            auc = metrics.roc_auc_score(y.to('cpu').numpy(), pred.to('cpu').detach().numpy())
            pred[pred>0.5] = 1.0
            pred[pred<0.5] = 0.0
            acc = torch.sum(pred*y + (1-pred)*(1-y))/y.shape[0]
            tp = torch.sum(pred * y)/torch.sum(y)
            fp = torch.sum(pred * (1-y))/torch.sum(1-y)
            aucs.append([c, auc])
            print(f"loss: {loss:>7f}  auc: {auc}  acc: {acc}  tp: {tp}  fp: {fp} [{current:>5d}] {torch.sum(y)}")

    aucs = np.array(aucs)
    plt.figure('Loss')
    plt.plot(iterations, losses, color='black')
    plt.figure('AUC')
    plt.plot(list(aucs[:, 0]), list(aucs[:, 1]), color='red')
    plt.show()


if __name__ == '__main__':
    loader = graph.load_data.DataLoader([2012, 2013, 2014])
    node_features = loader.initNodesFeatures()
    node_features = node_features.to(graph.device)
    print(node_features.shape)
    print("Start")
    train(loader.supervisedGenerator('./Datasets/2014_2017', k=0, batch_size=4096), model, node_features)

