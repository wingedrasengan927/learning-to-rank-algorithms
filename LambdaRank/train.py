import torch
import torch.nn.functional as F
import torch.nn as nn

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

def train_one_epoch(X_train, Y_train, X_val, Y_val, model, optimizer):

    n_train_qids = X_train.size(0)
    n_val_qids = X_val.size(0)

    for qid in range(n_train_qids):

        X = X_train[qid]
        Y = Y_train[qid]

        n_docs = X.size(0)

        # shuffle the query
        shuffled_indices = torch.randperm(n_docs)
        X = X[shuffled_indices]
        Y = Y[shuffled_indices]

        # compute reciprocal of ideal dcg
        sorted_rels, _ = torch.sort(Y, descending=True) # remember, more relevance means it comes at the top
        ideal_pos = torch.arange(1, n_docs + 1)
        N = 1 / compute_dcg(ideal_pos, sorted_rels)

        Y_pred = model(X)

        with torch.no_grad():

            exp_score_diffs = torch.exp(Y_pred - Y_pred.T)

            # sort docs based on scores
            sorted_pos = torch.argsort(Y_pred, descending=True, dim=0) + 1
            
            rels = Y.view(-1, 1)
            rels_diff = rels - rels.T
            pos_pairs = rels_diff > 0 # l_i > l_j
            neg_pairs = rels_diff < 0 # l_i < l_j    

            # compute S_ij     
            # S_ij = 1 if l_i > l_j
            # S_ij = -1 if l_i < l_j
            # S_ij = 0 otherwise
            S = torch.zeros(n_docs, n_docs)
            S.masked_fill_(pos_pairs, 1)
            S.masked_fill_(neg_pairs, -1)

            bce_grad = 0.5 * (1 - S) - 1 / (1 + exp_score_diffs) # gradient of binary cross entropy cost

            # compute ndcg gain
            gain_diff = torch.pow(2, rels) - torch.pow(2, rels.T)
            decay_diff = 1 / torch.log1p(sorted_pos) - 1 / torch.log1p(sorted_pos.T)
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)

            # compute lambdas
            lambda_update = bce_grad * delta_ndcg
            # perform summation along the rows
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            assert Y_pred.size() == lambda_update.size()

        # backbrop wrt lambdas
        optimizer.zero_grad()
        Y_pred.backward(lambda_update)
        optimizer.step()

    with torch.no_grad():
        total_ndcg = 0
        for qid in range(n_val_qids):
            n_val = X_val[qid].size(0)
            val_out = model(X_val[qid])
            ndcg = compute_ndcg(val_out, Y_val[qid])
            total_ndcg += ndcg

    return total_ndcg / n_val_qids
