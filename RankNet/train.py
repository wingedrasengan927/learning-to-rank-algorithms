import torch
import torch.nn.functional as F
import torch.nn as nn
import random

def pairwise_loss(o_ij, P_ij):
    '''
    Parameters
    ----------
    o_ij: f(x_i) - f(x_j) where f is the ranking function and x is the feature vector
    P_ij: actual probability that document i is ranked higher than document j
        P_ij can take values (1, 0.5, 0)

    Returns
    -------
    L: The Ranknet pairwise loss function 
        L = -P_ij * o_ij + log(1 + e^o_ij)
    '''
    return -P_ij * o_ij + torch.log1p(torch.exp(o_ij))

def create_dataset(n_train_qids, n_val_qids, n_docs, n_dim):
    '''
    create an artificial dataset with the specified dimensions and
    assign a relevance to each of them by passing through a network
    '''
    n_relevance = 6
    bin_net = nn.Linear(n_dim, n_relevance)

    X_train = torch.randn(n_train_qids, n_docs, n_dim)
    X_val = torch.randn(n_val_qids, n_docs, n_dim)

    # pass the feature vectors through the network
    # and categorize the output into one of six bins based on its score
    Y_train = torch.ones(n_train_qids, n_docs)
    Y_val = torch.ones(n_val_qids, n_docs)
    with torch.no_grad():
        # get train relevances
        for i in range(n_train_qids):
            out = bin_net(X_train[i])
            probs = F.softmax(out, dim=-1)
            relevances = torch.argmax(probs, dim=-1)
            Y_train[i, :] = relevances

        # get val relevances
        for i in range(n_val_qids):
            out = bin_net(X_val[i])
            probs = F.softmax(out, dim=-1)
            relevances = torch.argmax(probs, dim=-1)
            Y_val[i, :] = relevances
    
    return X_train, Y_train, X_val, Y_val

def compute_dcg(pos, rels):
    '''
    computes dcg given by: sum from i=1 to i=k (2^(l_i) - 1/log(1+i))
    '''
    num = torch.pow(2, rels) - 1
    den = torch.log1p(pos)
    return torch.sum(num / den)

def compute_ndcg(scores, rels):
    '''
    computes ndcg given by dcg / ideal_dcg
    '''
    scores = scores.view(-1)
    sorted_pos = torch.argsort(scores, descending=True) + 1
    # compute dcg
    dcg = compute_dcg(sorted_pos, rels)
    # compute ideal dcg
    sorted_rels, _ = torch.sort(rels, descending=True)
    ideal_pos = torch.arange(1, scores.size(0) + 1)
    ideal_dcg = compute_dcg(ideal_pos, sorted_rels)

    return dcg / ideal_dcg 

def swapped_pairs(scores_pred, relevances):
    '''
    No. of swaps required to get actual sequence from predicted sequence
    '''
    N = scores_pred.size(0)
    n_swaps = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            if relevances[i] < relevances[j]:
                if scores_pred[i] > scores_pred[j]:
                    n_swaps += 1
            elif relevances[i] > relevances[j]:
                if scores_pred[i] < scores_pred[j]:
                    n_swaps += 1
    return n_swaps

def train_one_epoch(X_train, Y_train, X_val, Y_val, model, optimizer, n_sampling_combinations):

    n_train_qids = X_train.size(0)
    n_val_qids = X_val.size(0)

    total_loss = 0
    for qid in range(n_train_qids):
        X = X_train[qid]
        Y = Y_train[qid]

        # shuffle the query
        shuffled_indices = torch.randperm(X.size(0))
        X = X[shuffled_indices]
        Y = Y[shuffled_indices]

        qid_loss = torch.zeros(1)
        if X.size(0) > 0:
            out = model(X)

            seen_pairs = set()
            for _ in range(n_sampling_combinations):
                i, j = random.sample(range(X.size(0)), 2)

                if (i, j) in seen_pairs or (j, i) in seen_pairs: # cost is swap invariant
                    continue
                seen_pairs.add((i, j))

                o_ij = out[i] - out[j]

                if Y[i] > Y[j]:
                    P_ij = 1
                elif Y[i] < Y[j]:
                    P_ij = 0
                else:
                    P_ij = 0.5

                loss = pairwise_loss(o_ij, P_ij)
                qid_loss += loss

            optimizer.zero_grad()
            qid_loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += qid_loss.item()

    total_ndcg = 0
    with torch.no_grad():
        total_pair_swaps = 0
        for qid in range(n_val_qids):
            n_val = X_val[qid].size(0)
            val_out = model(X_val[qid])
            n_swaps = swapped_pairs(val_out, Y_val[qid])
            pair_swaps = n_swaps / ( n_val * (n_val - 1) / 2 )
            total_pair_swaps += pair_swaps

            ndcg = compute_ndcg(val_out, Y_val[qid])
            total_ndcg += ndcg



    return total_loss / n_train_qids, total_pair_swaps / n_val_qids, total_ndcg / n_val_qids


                

