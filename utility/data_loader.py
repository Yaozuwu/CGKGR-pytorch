import os
import numpy as np


def data_split(args):
    data_dir = os.path.join(args.data_dir, args.data_name)
    train_file = os.path.join(data_dir, 'ratings_final.txt')
    data, _, _ = load_cf(train_file)

    print('splitting data ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = data.shape[0]

    for i in range(5):
        eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        train_data = data[train_indices]
        eval_data = data[eval_indices]
        test_data = data[test_indices]

        train_file = os.path.join(data_dir, 'train_' + str(i + 1) + '.txt')
        eval_file = os.path.join(data_dir, 'eval_' + str(i + 1) + '.txt')
        test_file = os.path.join(data_dir, 'test_' + str(i + 1) + '.txt')
        np.savetxt(train_file, train_data, fmt='%u')
        np.savetxt(eval_file, eval_data, fmt='%u')
        np.savetxt(test_file, test_data, fmt='%u')


def load_cf(train_file):
    print('loading CF file: ' + train_file + ' ...')
    data = np.loadtxt(train_file, dtype=np.int32)

    return data


def get_user_item_dict(data):
    user_item_dict = dict()
    item_user_dict = dict()
    for l in data:
        user, item, rate = l
        if user not in user_item_dict:
            user_item_dict[user] = set()
        user_item_dict[user].add(item)
        if item not in item_user_dict:
            item_user_dict[item] = set()
        item_user_dict[item].add(user)

    for k, v in user_item_dict.items():
        user_item_dict[k] = np.array(list(v))
    for k, v in item_user_dict.items():
        item_user_dict[k] = np.array(list(v))

    return user_item_dict, item_user_dict


def load_data(args, train_file, eval_file, test_file):
    data_dir = os.path.join(args.data_dir, args.data_name)

    train_file = os.path.join(data_dir, train_file)
    eval_file = os.path.join(data_dir, eval_file)
    test_file = os.path.join(data_dir, test_file)

    train_data = load_cf(train_file)
    eval_data = load_cf(eval_file)
    test_data = load_cf(test_file)

    return train_data, eval_data, test_data


def load_kg(kg_file):
    print('loading KG file: ' + kg_file + '...')
    kg_data = np.loadtxt(kg_file, dtype=np.int32)
    kg_dict = construct_kg(kg_data)

    n_relation = max(kg_data[:, 1]) + 1
    n_entity = max(max(kg_data[:, 0]), max(kg_data[:, 2])) + 1
    n_triplet = len(kg_data)
    for i in range(n_entity):
        if i not in kg_dict:
            print('error!', i)
    return kg_dict, n_relation, n_entity, n_triplet


def construct_kg(kg_data):
    print('constructing KG ...')
    kg_dict = dict()
    for triple in kg_data:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg_dict:
            kg_dict[head] = []
        kg_dict[head].append((tail, relation))
        if tail not in kg_dict:
            kg_dict[tail] = []
        kg_dict[tail].append((head, relation))

    for k, v in kg_dict.items():
        kg_dict[k] = np.array(v)

    return kg_dict


def construct_adj(args, n_user, n_item, n_entity, kg_dict, user_item_dict, item_user_dict):
    print('constructing adjacency matrix...')
    # each line of adj_u2i stores the sampled item neighbors for a given user
    # each line of adj_i2u stores the sampled user neighbors for a given item
    adj_u2i = np.zeros([n_user, args.sample_size], dtype=np.int32)
    adj_i2u = np.zeros([n_item, args.sample_size], dtype=np.int32)

    for user in range(n_user):
        if user not in user_item_dict:
            adj_u2i[user] = [0]
        else:
            items = user_item_dict[user]
            n_items = len(items)
            if n_items >= args.sample_size:
                sampled_indices = np.random.choice(list(range(n_items)), size=args.sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_items)), size=args.sample_size, replace=True)
            adj_u2i[user] = items[sampled_indices]

    for item in range(n_item):
        if item not in item_user_dict:
            adj_i2u[item] = [0]
        else:
            users = item_user_dict[item]
            n_users = len(users)
            if n_users >= args.sample_size:
                sampled_indices = np.random.choice(list(range(n_users)), size=args.sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_users)), size=args.sample_size, replace=True)
            adj_i2u[item] = users[sampled_indices]

    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_e2e = np.zeros([n_entity, args.sample_size], dtype=np.int32)
    adj_relation = np.zeros([n_entity, args.sample_size], dtype=np.int32)
    for entity in range(n_entity):
        neighbors = kg_dict[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.sample_size, replace=True)
        adj_e2e[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    ngh_sample_dict = dict()
    ngh_sample_dict['adj_u2i'] = adj_u2i
    ngh_sample_dict['adj_i2u'] = adj_i2u
    ngh_sample_dict['adj_e2e'] = adj_e2e
    ngh_sample_dict['adj_relation'] = adj_relation

    return ngh_sample_dict
