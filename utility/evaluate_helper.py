import numpy as np
import torch
def ctr_evaluate(model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    while start + batch_size <= data.shape[0]:
        user_indices = torch.tensor(data[start:start + batch_size, 0]).long().to(device)
        item_indices = torch.tensor(data[start:start + batch_size, 1]).long().to(device)
        labels = data[start:start + batch_size, 2]
        auc, f1 = model.evaluate(user_indices, item_indices, labels)
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def topk_settings(train_data, test_data, n_item):
    user_num = 100
    k_list = [1, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list

def topk_evaluate(model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    model.eval()
    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        while start + batch_size <= len(test_item_list):
            user_indices = torch.tensor([user] * batch_size).long().to(device)
            item_indices = torch.tensor(test_item_list[start:start + batch_size]).long().to(device)
            scores = model.get_scores(user_indices, item_indices)
            for item, score in zip(item_indices, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            user_indices = torch.tensor([user] * batch_size).long().to(device)
            item_indices = torch.tensor(test_item_list[start:] + [test_item_list[-1]] * (
                                     batch_size - len(test_item_list) + start)).long().to(device)
            scores = model.get_scores(user_indices, item_indices)
            for item, score in zip(item_indices, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        # item_sorted = item_sorted[:100]  # at most top@100
        hits = np.zeros(len(item_sorted))
        index = [i for i, x in enumerate(item_sorted) if int(x) in test_record[user]]
        hits[index] = 1

        for k in k_list:
            hit_k = hits[:k]
            hit_num = np.sum(hit_k)
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            dcg = np.sum((2 ** hit_k - 1) / np.log2(np.arange(2, k + 2)))
            sorted_hits_k = np.flip(np.sort(hits))[:k]
            idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))
            # idcg[idcg == 0] = np.inf
            ndcg_list[k].append(dcg / idcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg

def get_total_parameters(model):
    total_parameters = 0
    for para in model.parameters():
        total_parameters += 1
    return total_parameters


