import matplotlib.pyplot as plt
from time import time
from sklearn import metrics
import numpy as np


import graph
import graph.gcn
import torch
import graph.load_data

import wandb

def unsupervised_train(epochs, loader, initial_embedding, model, batch_size, loader_batch_size=16):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = [0]
    mean_losses = [0]
    iterations = [-1]
    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        top = time()
        generator = loader.Generator("Datasets/2014_2017", batch_size=loader_batch_size)
        loss = 0
        c = 0
        for sparse_adj_tensor, degs12, vertices, n1, _, _, _ in generator:
            # Get initial embedding from 
            hk = initial_embedding[vertices]

            print(vertices.shape)

            # Scale adjacency matrix
            sparse_adj_tensor = torch.sparse.mm(sparse_adj_tensor, degs12)
            sparse_adj_tensor = torch.sparse.mm(degs12, sparse_adj_tensor)

            hk = model.forward(sparse_adj_tensor, hk)

            i, v = sparse_adj_tensor._indices(), sparse_adj_tensor._values()
            mask = i < n1
            mask = torch.logical_and(mask[0], mask[1])
            sparse_adj_tensor = torch.sparse_coo_tensor(i[:, mask], v[mask], (n1, n1)).to_dense()
            del mask
            del i
            del v
            hk = hk[:n1]
            reconstructedMatrix = torch.mm(hk, torch.transpose(hk, 0, 1))

            # loss = loss_fn(reconstructedMatrix, sparse_adj_tensor)
            l1 = - sparse_adj_tensor * torch.log(torch.sigmoid(reconstructedMatrix)+1e-5)
            l1 = torch.sum(l1) / (torch.sum(sparse_adj_tensor)+1e-5)
            l2 = - (1 - sparse_adj_tensor - torch.diag(torch.diagonal(sparse_adj_tensor, 0))) * torch.log(torch.sigmoid(-reconstructedMatrix)+1e-5)
            l2 = torch.sum(l2) / (torch.sum(1 - sparse_adj_tensor - torch.diag(torch.diagonal(sparse_adj_tensor, 0)))+1e-5)
            loss = l1 + l2

            if c == batch_size:
                loss /= batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                iterations.append(iterations[-1] + 1)
                m_loss = losses[-1] if len(losses)<=10 else sum(losses[-10:])/10
                mean_losses.append(m_loss)
                print(f"[{iterations[-1]}]  Loss: {losses[-1]:.3f}   l1: {l1:.3f}   l2: {l2:.3f}   mean_loss: {m_loss}")
                # print(hk[0])
                # print(hk[1])
                loss = 0
                c = 0

            c += 1


def supervised_train(epochs, loader, initial_embedding, gnn, classifier, batch_size, loader_batch_size=16):
    optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = [0]
    mean_losses = [0]
    aucs = [0]
    mean_auc = [0]
    iterations = [-1]
    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        top = time()
        generator = loader.supervisedGenerator("Datasets/2014_2017", batch_size=loader_batch_size)
        loss = 0
        c = 0
        for sparse_adj_tensor, degs12, vertices, n1, batch_pairs, batch_labels, batch_pairs_features in generator:
            batch_pairs = batch_pairs.to(graph.device)
            batch_labels = batch_labels.to(graph.device)           
            
            # Get initial embedding from 
            h0 = initial_embedding[vertices]

            # Scale adjacency matrix
            sparse_adj_tensor = torch.sparse.mm(sparse_adj_tensor, degs12)
            sparse_adj_tensor = torch.sparse.mm(degs12, sparse_adj_tensor)

            hk = gnn.forward(sparse_adj_tensor, h0)
            
            pairs_hk = hk[batch_pairs]
            pairs_preds = classifier(pairs_hk)

            loss = loss_fn(pairs_preds, batch_labels)


            if c == batch_size:
                loss /= batch_size
                optimizer_gnn.zero_grad()
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_gnn.step()
                optimizer_classifier.step()

                losses.append(loss.item())
                iterations.append(iterations[-1] + 1)

                pairs_preds = torch.squeeze(pairs_preds[:, 1])
                auc = metrics.roc_auc_score(batch_labels.to('cpu').numpy(), pairs_preds.to('cpu').detach().numpy())

                m_loss = losses[-1] if len(losses)<=10 else sum(losses[-10:])/10
                mean_losses.append(m_loss)
                
                aucs.append(auc)
                m_auc = 0 if len(aucs)<=10 else sum(aucs[-10:])/10
                mean_auc.append(m_auc)

                print(f"[{iterations[-1]}]   Loss: {losses[-1]}  AUC: {auc}  Mean Loss: {m_loss}  Mean AUC: {m_auc}")
                # print(hk[0])
                # print(hk[1])
                loss = 0
                c = 0

            c += 1


def unsupervised_training_multiple(epochs, loader, initial_embedding, model, optimizer, batch_size, iter_number, max_neighbors, k_init, k_max, starting_size=16, display=False, log=False):

    iterations = []

    if display:
        plt.figure()
        plt.ion()
        losses = []
        mean_losses = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        top = time()
        generator = loader.Generator(iter_number=iter_number, max_neighbors=max_neighbors, starting_size=starting_size, k_init=k_init, k_max=k_max)
        c = 0
        loss_batch = 0
        for sparse_adj_tensor, vertices, n1, embedding_index in generator:

            # Get initial embedding from
            hk = initial_embedding[embedding_index][vertices]

            # Scale adjacency matrix
            # sparse_adj_tensor = torch.sparse.mm(sparse_adj_tensor, degs12)
            # sparse_adj_tensor = torch.sparse.mm(degs12, sparse_adj_tensor)

            hk = model.forward(sparse_adj_tensor, hk)

            i, v = sparse_adj_tensor._indices(), sparse_adj_tensor._values()
            mask = i < n1
            mask = torch.logical_and(mask[0], mask[1])
            sparse_adj_tensor = torch.sparse_coo_tensor(i[:, mask], v[mask], (n1, n1)).to_dense()
            sparse_adj_tensor = sparse_adj_tensor - torch.diag(torch.diagonal(sparse_adj_tensor, 0))
            del mask
            del i
            del v
            hk = hk[:n1]
            reconstructedMatrix = torch.mm(hk, torch.transpose(hk, 0, 1))

            # loss = loss_fn(reconstructedMatrix, sparse_adj_tensor)
            l1_mat = - sparse_adj_tensor * torch.log(torch.sigmoid(reconstructedMatrix)+1e-5)
            l1 = torch.sum(l1_mat) / (torch.sum(sparse_adj_tensor)+1e-5)
            l2_mat = - (1 - sparse_adj_tensor - torch.eye(n1, n1, dtype=torch.float32, device=graph.device)) * torch.log(torch.sigmoid(-reconstructedMatrix)+1e-5)
            l2 = torch.sum(l2_mat) / (torch.sum(1 - sparse_adj_tensor - torch.eye(n1, n1, dtype=torch.float32, device=graph.device))+1e-5)
            loss = (l1 + l2) / batch_size
            loss.backward()
            loss_batch += loss.item() / batch_size

            iterations.append(iterations[-1] + 1 if iterations != [] else 0)

            if c == batch_size:
                optimizer.step()
                optimizer.zero_grad()


                if display:

                    losses.append(loss_batch)
                    k = min(len(losses), 100)
                    m_loss = sum(losses[-k:])/k
                    mean_losses.append(m_loss)

                    plt.clf()
                    plt.plot(iterations, losses, label='Loss')
                    plt.plot(iterations, mean_losses, label='mean Loss')
                    plt.legend()
                    plt.show()
                    plt.pause(1e-3)
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {losses[-1]:.5f}   l1: {l1:.5f}   l2: {l2:.5f}   mean_loss: {m_loss}   n1: {n1}")
                else:
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {loss_batch:.5f}   l1: {l1:.5f}   l2: {l2:.5f}   n1: {n1}")

                if log:
                    wandb.log({"loss GNN": loss_batch, "unsupervised step" : iterations[-1]})

                c = 0
                loss_batch = 0

            c += 1

    if display:
        plt.close()


def supervised_training_multiple(epochs, iter_number, loader, initial_embedding, gnn, classifier, optimizer_gnn, optimizer_classifier, batch_size, loss_fn, display=False, log=False):
    iterations = []

    if display:
        losses = []
        mean_losses = []
        aucs = []
        mean_auc = []
        plt.figure()
        plt.ion()

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        top = time()
        generator = loader.Generator()
        c = 0
        loss_batch = 0
        for sub_generator, labels in generator:

            pairs_hk = torch.zeros(size=(loader.batch_size, 2, 0), dtype=torch.float32, device=graph.device)

            for sparse_adj_tensor, vertices, i, pairs in sub_generator:

                # batch_pairs = pairs.to(device)
                
                # Get initial embedding from 
                hk = initial_embedding[i][vertices]
                del vertices

                # Scale adjacency matrix
                #sparse_adj_tensor = torch.sparse.mm(sparse_adj_tensor, degs12)
                #sparse_adj_tensor = torch.sparse.mm(degs12, sparse_adj_tensor)

                # Compute embedding for years i
                hk = gnn.forward(sparse_adj_tensor, hk)
                
                # Concatenate embedding
                pairs_hk = torch.cat((pairs_hk, hk[pairs]), -1)
                del hk
                del pairs


            # Forward classifier
            pairs_preds = classifier(pairs_hk)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size
            loss_batch += loss.item() / batch_size
            loss.backward()

            iterations.append(iterations[-1] + 1 if iterations != [] else 0)

            if c == batch_size:
                optimizer_gnn.step()
                optimizer_classifier.step()
                optimizer_gnn.zero_grad()
                optimizer_classifier.zero_grad()

                pairs_preds = torch.squeeze(pairs_preds[:, 1])
                auc = metrics.roc_auc_score(labels.to('cpu').numpy(), pairs_preds.to('cpu').detach().numpy())


                if display:

                    losses.append(loss_batch)

                    k = min(len(losses), 15)
                    m_loss = sum(losses[-k:])/k
                    mean_losses.append(m_loss)
                    aucs.append(auc)
                    m_auc = sum(aucs[-k:])/k
                    mean_auc.append(m_auc)

                    plt.clf()
                    plt.plot(iterations, losses, label='Loss')
                    plt.plot(iterations, mean_losses, label='mean Loss')
                    plt.legend()
                    plt.show()
                    plt.pause(1e-3)
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {losses[-1]:.5f}   AUC: {auc:.5f}   Mean Loss: {m_loss:.5f}   Mean AUC: {m_auc:.5f}")
                else:
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {loss_batch:.5f}   AUC: {auc:.5f}")

                if log:
                    wandb.log({"loss classifier": loss_batch, "AUC classifier": auc, "supervised step" : iterations[-1]})

                if iterations[-1] == iter_number:
                    break
                # print(hk[0])
                # print(hk[1])
                c = 0
                loss_batch = 0

            c += 1


def supervised_training_multiple_with_test(epochs, iter_number, loader, initial_embedding, gnn, classifier,
                                           optimizer_gnn, optimizer_classifier, loss_fn, batch_size,
                                           test_rate=5, log=False, early_stop=False):
    iterations = []
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        generator = loader.Generator(iter_number, test_rate*batch_size)
        c = 0
        for sub_generator, is_test, year_start in generator:

            if is_test:
                # gnn.eval()
                classifier.eval()
                with torch.no_grad():
                    pairs_preds, labels = pass_forward(pairs_batch_size=loader.test_size, sub_generator=sub_generator, h0=initial_embedding, gnn=gnn, classifier=classifier, split=False)
            else:
                iterations.append(iterations[-1] + 1 if iterations != [] else 1)
                gnn.train()
                classifier.train()
                pairs_preds, labels = pass_forward(pairs_batch_size=loader.pairs_batch_size, sub_generator=sub_generator, h0=initial_embedding, gnn=gnn, classifier=classifier)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size

            if not is_test:
                loss.backward()
                c += 1
                if c == batch_size:
                    optimizer_gnn.step()
                    optimizer_classifier.step()
                    optimizer_gnn.zero_grad()
                    optimizer_classifier.zero_grad()
                    c = 0

                compute_scores(loss.item()*batch_size, train_losses, pairs_preds, labels, train_aucs)
                print(f"    [{iterations[-1]:^4}/{iter_number:^4}]  Loss: {train_losses[-1]:.8f}   AUC: {train_aucs[-1]:.8f}   from {year_start} to {year_start+3}")

                if log:
                    wandb.log({"train loss classifier": train_losses[-1], "train AUC": train_aucs[-1], "supervised step" : iterations[-1]})

            else:
                compute_scores(loss.item()*batch_size, test_losses, pairs_preds, labels, test_aucs)
                print(f"Loss: {test_losses[-1]:.8f}   AUC: {test_aucs[-1]:.8f}   from {year_start} to {year_start+3}")
                
                if log:
                    wandb.log({"test loss classifier": test_losses[-1], "test AUC": test_aucs[-1]})
                
            if early_stop and len(test_losses) > 10:
                if test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3] and test_losses[-3] > test_losses[-4] and test_losses[-4] > test_losses[-5]:
                    break


def supervised_training_multiple_with_test_all_vertices(epochs, iter_number, loader, h0, gnn, classifier,
                                           optimizer_gnn, optimizer_classifier, loss_fn, batch_size,
                                           test_rate=5, log=False, early_stop=False):
    iterations = []
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        generator = loader.Generator(iter_number, test_rate*batch_size)
        c = 0
        for sub_generator, is_test, year_start in generator:

            if is_test:
                # gnn.eval()
                classifier.eval()

                with torch.no_grad():
                    pairs_preds, labels = pass_forward_all(pairs_batch_size=loader.test_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier, split=False)
            else:
                iterations.append(iterations[-1] + 1 if iterations != [] else 1)
                gnn.train()
                classifier.train()

                # -- get correct embedding

                pairs_preds, labels = pass_forward_all(pairs_batch_size=loader.pairs_batch_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size

            if not is_test:
                loss.backward()
                c += 1
                if c == batch_size:
                    optimizer_gnn.step()
                    optimizer_classifier.step()
                    optimizer_gnn.zero_grad()
                    optimizer_classifier.zero_grad()
                    c = 0

                compute_scores(loss.item()*batch_size, train_losses, pairs_preds, labels, train_aucs)
                print(f"    [{iterations[-1]:^4}/{iter_number:^4}]  Loss: {train_losses[-1]:.8f}   AUC: {train_aucs[-1]:.8f}   from {year_start} to {year_start+3}")

                if log:
                    wandb.log({"train loss classifier": train_losses[-1], "train AUC": train_aucs[-1], "supervised step" : iterations[-1]})

            else:
                compute_scores(loss.item()*batch_size, test_losses, pairs_preds, labels, test_aucs)
                print(f"Loss: {test_losses[-1]:.8f}   AUC: {test_aucs[-1]:.8f}   from {year_start} to {year_start+3}")
                
                if log:
                    wandb.log({"test loss classifier": test_losses[-1], "test AUC": test_aucs[-1]})
                
            if early_stop and len(test_losses) > 10:
                if test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3] and test_losses[-3] > test_losses[-4] and test_losses[-4] > test_losses[-5]:
                    break

def ultimate_supervised_training(epochs, loader, initial_embedding, gnn, classifier,
                                           optimizer_gnn, optimizer_classifier, loss_fn, pairs_batch_size, batch_size, delta_years,
                                           test_rate=5, log=False, early_stop=False):
    iterations = []
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        # batch_size, delta_years, test_rate
        generator = loader.supervised_generator(pairs_batch_size, delta_years, test_rate)
        c = 0
        for sub_generator, is_test, year_start in generator:

            years = np.array([y for y in range(year_start-delta_years+1, year_start+1)], dtype=np.int32) - loader.min_year
            h0 = initial_embedding[years]
            if is_test:
                # gnn.eval()
                classifier.eval()
                with torch.no_grad():
                    pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=loader.test_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier, split=True)
            else:
                iterations.append(iterations[-1] + 1 if iterations != [] else 1)
                gnn.train()
                classifier.train()
                pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=pairs_batch_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size

            if not is_test:
                loss.backward()
                c += 1
                if c == batch_size:
                    optimizer_gnn.step()
                    optimizer_classifier.step()
                    optimizer_gnn.zero_grad()
                    optimizer_classifier.zero_grad()
                    c = 0

                compute_scores(loss.item()*batch_size, train_losses, pairs_preds, labels, train_aucs)
                print(f"    [{iterations[-1]:^4}]  Loss: {train_losses[-1]:.8f}   AUC: {train_aucs[-1]:.8f}   from {year_start} to {year_start+3}")

                if log:
                    wandb.log({"train loss classifier": train_losses[-1], "train AUC": train_aucs[-1], "supervised step" : iterations[-1]})

            else:
                compute_scores(loss.item()*batch_size, test_losses, pairs_preds, labels, test_aucs)
                print(f"Loss: {test_losses[-1]:.8f}   AUC: {test_aucs[-1]:.8f}   from {year_start} to {year_start+3}")
                
                if log:
                    wandb.log({"test loss classifier": test_losses[-1], "test AUC": test_aucs[-1]})
                
            if early_stop and len(test_losses) > 10:
                if test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3] and test_losses[-3] > test_losses[-4] and test_losses[-4] > test_losses[-5]:
                    break


def ultimate_supervised_training2(epochs, loader, initial_embedding, gnn, classifier,
                                           optimizer_gnn, optimizer_classifier, loss_fn, pairs_batch_size, batch_size, delta_years,
                                           test_rate=5, log=False, early_stop=False):
    iterations = []
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        # batch_size, delta_years, test_rate
        generator = loader.supervised_generator2(pairs_batch_size, delta_years, test_rate)
        c = 0
        for sub_generator, is_test, year_start in generator:

            years = np.array([y for y in range(year_start-delta_years+1, year_start+1)], dtype=np.int32) - loader.min_year
            h0 = initial_embedding[years]
            if is_test:
                # gnn.eval()
                classifier.eval()
                with torch.no_grad():
                    pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=loader.test_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier, split=True)
            else:
                iterations.append(iterations[-1] + 1 if iterations != [] else 1)
                gnn.train()
                classifier.train()
                pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=pairs_batch_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size

            if not is_test:
                loss.backward()
                c += 1
                if c == batch_size:
                    optimizer_gnn.step()
                    optimizer_classifier.step()
                    optimizer_gnn.zero_grad()
                    optimizer_classifier.zero_grad()
                    c = 0

                compute_scores(loss.item()*batch_size, train_losses, pairs_preds, labels, train_aucs)
                print(f"    [{iterations[-1]:^4}]  Loss: {train_losses[-1]:.8f}   AUC: {train_aucs[-1]:.8f}   from {year_start} to {year_start+3}")

                if log:
                    wandb.log({"train loss classifier": train_losses[-1], "train AUC": train_aucs[-1], "supervised step" : iterations[-1]})

            else:
                compute_scores(loss.item()*batch_size, test_losses, pairs_preds, labels, test_aucs)
                print(f"Loss: {test_losses[-1]:.8f}   AUC: {test_aucs[-1]:.8f}   from {year_start} to {year_start+3}")
                
                if log:
                    wandb.log({"test loss classifier": test_losses[-1], "test AUC": test_aucs[-1]})
                
            if early_stop and len(test_losses) > 10:
                if test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3] and test_losses[-3] > test_losses[-4] and test_losses[-4] > test_losses[-5]:
                    break

def ultimate_supervised_training3(epochs, loader, initial_embedding, gnn, classifier, iter_number,
                                           optimizer_gnn, optimizer_classifier, loss_fn, pairs_batch_size, batch_size, delta_years,
                                           test_rate=5, log=False, early_stop=False):
    iterations = []
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")
        # batch_size, delta_years, test_rate
        generator = loader.supervised_generator3(pairs_batch_size, delta_years, iter_number)
        c = 0
        for sub_generator, is_test, year_start in generator:

            years = np.array([y for y in range(year_start-delta_years+1, year_start+1)], dtype=np.int32) - loader.min_year
            h0 = initial_embedding[years]
            if is_test:
                # gnn.eval()
                classifier.eval()
                with torch.no_grad():
                    pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=loader.test_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier, split=True)
            else:
                iterations.append(iterations[-1] + 1 if iterations != [] else 1)
                gnn.train()
                classifier.train()
                pairs_preds, labels = ultimate_pass_forward(pairs_batch_size=pairs_batch_size, sub_generator=sub_generator, h0=h0, gnn=gnn, classifier=classifier)

            # Compute loss
            loss = loss_fn(pairs_preds, labels) / batch_size

            if not is_test:
                loss.backward()
                c += 1
                if c == batch_size:
                    optimizer_gnn.step()
                    optimizer_classifier.step()
                    optimizer_gnn.zero_grad()
                    optimizer_classifier.zero_grad()
                    c = 0

                compute_scores(loss.item()*batch_size, train_losses, pairs_preds, labels, train_aucs)
                print(f"    [{iterations[-1]:^4}]  Loss: {train_losses[-1]:.8f}   AUC: {train_aucs[-1]:.8f}   from {year_start} to {year_start+3}")

                if log:
                    wandb.log({"train loss classifier": train_losses[-1], "train AUC": train_aucs[-1], "supervised step" : iterations[-1]})

            else:
                compute_scores(loss.item()*batch_size, test_losses, pairs_preds, labels, test_aucs)
                print(f"Loss: {test_losses[-1]:.8f}   AUC: {test_aucs[-1]:.8f}   from {year_start} to {year_start+3}")
                
                if log:
                    wandb.log({"test loss classifier": test_losses[-1], "test AUC": test_aucs[-1]})
                
            if early_stop and len(test_losses) > 10:
                if test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3] and test_losses[-3] > test_losses[-4] and test_losses[-4] > test_losses[-5]:
                    break


def ultimate_unsupervised_training(epochs, loader, initial_embedding, model, optimizer, batch_size, iter_number, max_neighbors, k_init, k_max, starting_size=16, display=False, log=False, years=None):

    iterations = []

    if display:
        plt.figure()
        plt.ion()
        losses = []
        mean_losses = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")

        # -- create generator
        generator = loader.unsupervised_generator3(iter_number=iter_number, max_neighbors=max_neighbors, starting_size=starting_size, k_init=k_init, k_max=k_max, years=years)

        # -- get corresponding initial embedding
        years = np.array(years, dtype=np.int32)
        h0 = initial_embedding[years - loader.min_year]

        c = 0
        loss_batch = 0
        for sparse_adj_tensor_scale, vertices, n1, embedding_index in generator:

            # Get initial embedding from
            hk = h0[embedding_index][vertices]

            # Scale adjacency matrix
            # sparse_adj_tensor = torch.sparse.mm(sparse_adj_tensor, degs12)
            # sparse_adj_tensor = torch.sparse.mm(degs12, sparse_adj_tensor)

            hk = model.forward(sparse_adj_tensor_scale, hk)

            i, v = sparse_adj_tensor_scale._indices(), sparse_adj_tensor_scale._values()
            v = torch.ones_like(v, device=graph.device)
            mask = i < n1
            mask = torch.logical_and(mask[0], mask[1])
            sparse_adj_tensor = torch.sparse_coo_tensor(i[:, mask], v[mask], (n1, n1)).to_dense()
            sparse_adj_tensor = sparse_adj_tensor - torch.diag(torch.diagonal(sparse_adj_tensor, 0))
            del mask
            del i
            del v
            hk = hk[:n1]
            reconstructedMatrix = torch.mm(hk, torch.transpose(hk, 0, 1))

            # loss = loss_fn(reconstructedMatrix, sparse_adj_tensor)
            l1_mat = - sparse_adj_tensor * torch.log(torch.sigmoid(reconstructedMatrix)+1e-5)
            l1 = torch.sum(l1_mat) / (torch.sum(sparse_adj_tensor)+1e-5)
            l2_mat = - (1 - sparse_adj_tensor - torch.eye(n1, n1, dtype=torch.float32, device=graph.device)) * torch.log(torch.sigmoid(-reconstructedMatrix)+1e-5)
            l2 = torch.sum(l2_mat) / (torch.sum(1 - sparse_adj_tensor - torch.eye(n1, n1, dtype=torch.float32, device=graph.device))+1e-5)
            loss = (l1 + l2) / batch_size
            loss.backward()
            loss_batch += loss.item() / batch_size

            iterations.append(iterations[-1] + 1 if iterations != [] else 0)

            if c == batch_size:
                optimizer.step()
                optimizer.zero_grad()

                if display:

                    losses.append(loss_batch)
                    k = min(len(losses), 100)
                    m_loss = sum(losses[-k:])/k
                    mean_losses.append(m_loss)

                    plt.clf()
                    plt.plot(iterations, losses, label='Loss')
                    plt.plot(iterations, mean_losses, label='mean Loss')
                    plt.legend()
                    plt.show()
                    plt.pause(1e-3)
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {losses[-1]:.5f}   l1: {l1:.5f}   l2: {l2:.5f}   mean_loss: {m_loss}   n1: {n1}")
                else:
                    print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {loss_batch:.5f}   l1: {l1:.5f}   l2: {l2:.5f}   n1: {n1}")

                if log:
                    wandb.log({"loss GNN": loss_batch, "unsupervised step" : iterations[-1]})

                c = 0
                loss_batch = 0

            c += 1

    if display:
        plt.close()


def ultimate_unsupervised_training2(epochs, loader, initial_embedding, model, optimizer, batch_size, iter_number,
                                    starting_size=16, nb_neighbors=1, nb_non_neighbors=4, years=None, display=False, log=False):

    iterations = []

    for epoch in range(epochs):
        print(f"\n######## Epoch {epoch} ########\n")

        # -- create generator
        generator = loader.unsupervised_generator2(
            iter_number=iter_number,
            starting_size=starting_size,
            nb_neighbors=nb_neighbors,
            nb_non_neighbors=nb_non_neighbors,
            years=years
        )

        # -- get corresponding initial embedding
        years = np.array(years, dtype=np.int32)
        h0 = initial_embedding[years - loader.min_year]

        c = 0
        loss_batch = 0
        for m, vertices, vertices_neighbors, vertices_non_neighbors, embedding_index in generator:
            # Get initial embedding from
            hk = h0[embedding_index]
            hk = model.forward(m, hk)

            # print("vertices", hk[vertices])
            # print("ein l1", torch.einsum("ijk, ik -> ij", hk[vertices_neighbors], hk[vertices]))
            # print("ein l2", torch.einsum("ijk, ik -> ij", hk[vertices_non_neighbors], hk[vertices]))
            # print("sigmoid l1", torch.sigmoid(torch.einsum("ijk, ik -> ij", hk[vertices_neighbors], hk[vertices])))
            # print("sigmoid l2", torch.sigmoid(torch.einsum("ijk, ik -> ij", hk[vertices_non_neighbors], hk[vertices])))
            l1 = torch.mean(-torch.log(torch.sigmoid(torch.einsum("ijk, ik -> ij", hk[vertices_neighbors], hk[vertices]))))
            l2 = torch.mean(-torch.log(torch.sigmoid(-torch.einsum("ijk, ik -> ij", hk[vertices_non_neighbors], hk[vertices]))))
            loss = (l1 + l2) / batch_size
            loss.backward()
            loss_batch += loss.item() / batch_size

            iterations.append(iterations[-1] + 1 if iterations != [] else 0)

            if c == batch_size:
                optimizer.step()
                optimizer.zero_grad()

                if log:
                    wandb.log({"loss GNN": loss_batch, "unsupervised step" : iterations[-1]})

                print(f"[{iterations[-1]:^4}/{iter_number:^4}]   Loss: {loss_batch:.5f}   l1: {l1:.5f}   l2: {l2:.5f}")

                c = 0
                loss_batch = 0
            c += 1


def pass_forward(pairs_batch_size, sub_generator, h0, gnn, classifier, split=False):
    pairs_hk = torch.zeros(size=(pairs_batch_size, 2, 0), dtype=torch.float32, device=graph.device)
    
    for sparse_adj_tensor, vertices, pairs, labels in sub_generator:
        
        # Get initial embeddings
        hk = h0[vertices]

        # Compute embeddings with gnn
        hk = gnn.forward(sparse_adj_tensor, hk)
        
        # Concatenate embeddings
        pairs_hk = torch.cat((pairs_hk, hk[pairs]), -1)

    # Forward classifier
    if not split:
        return classifier(pairs_hk), labels

    pairs_hk = torch.split(pairs_hk, 100000)
    pairs_preds = []
    for p in pairs_hk:
        pairs_preds.append(classifier(p))
    pairs_preds = torch.cat(pairs_preds, 0)
    return pairs_preds, labels


def pass_forward_all(pairs_batch_size, sub_generator, h0, gnn, classifier, split=False):
    pairs_hk = torch.zeros(size=(pairs_batch_size, 2, 0), dtype=torch.float32, device=graph.device)
    
    for sparse_adj_tensor, pairs, labels in sub_generator:
        hk = gnn.forward(sparse_adj_tensor, h0)
        # Concatenate embeddings
        pairs_hk = torch.cat((pairs_hk, hk[pairs]), -1)

    # Forward classifier
    if not split:
        return classifier(pairs_hk), labels

    pairs_hk = torch.split(pairs_hk, 100000)
    pairs_preds = []
    for p in pairs_hk:
        pairs_preds.append(classifier(p))
    pairs_preds = torch.cat(pairs_preds, 0)
    return pairs_preds, labels

def ultimate_pass_forward(pairs_batch_size, sub_generator, h0, gnn, classifier, split=False):
    pairs_hk = torch.zeros(size=(pairs_batch_size, 2, 0), dtype=torch.float32, device=graph.device)
    
    for l, (sparse_adj_tensor, pairs, labels) in enumerate(sub_generator):
        hk = h0[l]
        hk = gnn.forward(sparse_adj_tensor, hk)
        # Concatenate embeddings
        pairs_hk = torch.cat((pairs_hk, hk[pairs]), -1)

    # Forward classifier
    if not split:
        return classifier(pairs_hk), labels

    pairs_hk = torch.split(pairs_hk, 10000)
    pairs_preds = []
    for p in pairs_hk:
        pairs_preds.append(classifier(p))
    pairs_preds = torch.cat(pairs_preds, 0)
    return pairs_preds, labels



def compute_scores(loss, losses, preds, labels, aucs):
    losses.append(loss)

    preds = torch.softmax(preds, -1)
    preds = torch.squeeze(preds[:, 1]).to('cpu').detach().numpy()
    labels = labels.to('cpu').numpy()
    auc = metrics.roc_auc_score(labels, preds)
    aucs.append(auc)

