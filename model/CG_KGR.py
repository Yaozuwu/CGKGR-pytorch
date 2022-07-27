import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, _utils
from time import time
from utility.data_loader import construct_adj, load_kg, get_user_item_dict

from src.aggregators import EntityAggregator, UserAggregator
from data.dataset import RecoDataset

from utility.train_helper import l2_loss

from sklearn.metrics import f1_score, roc_auc_score

from utility.evaluate_helper import ctr_evaluate, topk_settings, topk_evaluate, get_total_parameters


class CGKGR(nn.Module):
    def __init__(self, args, train_data, eval_data, test_data):
        super(CGKGR, self).__init__()
        # Dataset
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.n_users = max(max(self.train_data[:, 0]), max(self.eval_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.eval_data[:, 1]), max(self.test_data[:, 1])) + 1

        # Configuration
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # General hyper-parameters
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        self.dropout = args.dropout
        self.node_dim = args.node_dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

        # User-Item Interaction
        self.user_item_dict, self.item_user_dict = get_user_item_dict(train_data)

        # Knowledge graph
        data_dir = os.path.join(args.data_dir, args.data_name)
        kg_file = os.path.join(data_dir, args.kg_file)
        kg_dict, n_relation, n_entity, n_triplet = load_kg(kg_file)

        # Sample Ngh
        self.ngh_sample_dict = construct_adj(args, self.n_users, self.n_items, n_entity, kg_dict, self.user_item_dict,
                                             self.item_user_dict)

        # User-Entity Aggregator
        # currently we have n_layers for entity aggregator
        Entity_aggregator = EntityAggregator
        # current we only have 1 layer for user aggregator
        User_aggregator = UserAggregator

        # Define User_aggregator
        self.user_aggregator = User_aggregator(self.args, act_f=torch.tanh, name=None).to(self.device)

        # Define Entity_aggregator
        self.entity_aggregators_list = nn.ModuleList()  # store all entity_aggregators
        for i in range(self.n_layer):
            if i == self.n_layer - 1:
                entity_aggregator = Entity_aggregator(self.args, act_f=torch.tanh, name=None).to(self.device)
            else:
                entity_aggregator = Entity_aggregator(self.args, act_f=F.relu, name=None).to(self.device)
            self.entity_aggregators_list.append(entity_aggregator)

        # Build Inputs
        self._build_inputs(self.n_users, n_entity, n_relation)

    def _build_inputs(self, n_user, n_entity, n_relation):
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.user_emb_matrix = torch.nn.Embedding(self.n_user, self.node_dim)
        nn.init.xavier_uniform_(self.user_emb_matrix.weight)
        self.entity_emb_matrix = torch.nn.Embedding(self.n_entity, self.node_dim)
        nn.init.xavier_uniform_(self.entity_emb_matrix.weight)
        self.W_R = nn.Parameter(torch.Tensor(n_relation + 1, self.node_dim, self.node_dim))
        nn.init.xavier_uniform_(self.W_R)

        self.adj_u2i = torch.tensor(self.ngh_sample_dict['adj_u2i'])
        self.adj_i2u = torch.tensor(self.ngh_sample_dict['adj_i2u'])
        self.adj_e2e = torch.tensor(self.ngh_sample_dict['adj_e2e'])
        self.adj_relation = torch.tensor(self.ngh_sample_dict['adj_relation'])

    def _get_item_ngh(self, user_seeds):
        """
        :param user_seeds:      [batch_size]
        :return:                [batch_size, sample_size]
        """
        return torch.reshape(self.adj_u2i[user_seeds], [self.batch_size, -1]).long()

    def _get_user_ngh(self, item_seeds):
        """
        :param item_seeds:      [batch_size]
        :return:                [batch_size, sample_size]
        """
        return torch.reshape(self.adj_i2u[item_seeds], [self.batch_size, -1]).long()

    def _get_entity_ngh_multihop(self, item_seeds):
        """
        :param item_seeds:  [batch_size]
        :return:   entity: {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
                           relation: {[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        """
        #   [batch_size, 1]
        item_seeds = torch.unsqueeze(item_seeds, dim=1)
        entities = [item_seeds]
        relations = []
        for i in range(self.n_layer):
            neighbor_entities = torch.reshape(self.adj_e2e[entities[i]], [self.batch_size, -1]).long().to(self.device)
            neighbor_relations = torch.reshape(self.adj_relation[entities[i]],
                                               [self.batch_size, -1]).long().to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate_for_users(self, user_index):
        """
        :param args:
        :param user_index:              [batch_size]
        :return:                        [batch_size, dim]
        """
        # [batch_size, sample_size]
        item_ngh_index = self._get_item_ngh(user_index).to(self.device)
        # [batch_size, d]
        user_embedding = self.user_emb_matrix(user_index).to(self.device)
        # [batch_size, sample_size, d]
        item_ngh_embedding = self.entity_emb_matrix(item_ngh_index).to(self.device)
        # W_ui:   [dim, dim]
        W_ui = self.W_R[self.n_relation].to(self.device)

        output = self.user_aggregator(user_embedding, item_ngh_embedding, W_ui)

        return output

    def aggregate_for_items(self, item_index, new_user_embeddings):
        """
        :param args:
        :param item_index:              [batch_size]
        :param new_user_embeddings:     [batch_size, dim]
        :param W_R                      [n_relation, dim, dim]
        :return:
        """
        # [batch_size, sample_size]
        ngh_user_index = self._get_user_ngh(item_index).to(self.device)
        # [batch_size, dim]
        item_embeddings = self.entity_emb_matrix(item_index).to(self.device)

        # [batch_size, sample_size, dim]
        ngh_user_embeddings = self.user_emb_matrix(ngh_user_index).to(self.device)

        # part 2: aggregate from entity side in KG
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # entity: {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        # relation: {[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        entities, relations = self._get_entity_ngh_multihop(item_index)

        entity_embeddings = [self.entity_emb_matrix(i) for i in entities]
        is_item_layer = False
        for i in range(self.n_layer):
            entity_aggregator = self.entity_aggregators_list[i]
            if i == self.n_layer - 1:
                is_item_layer = True

            entity_vectors_next_iter = []
            for hop in range(self.n_layer - i):
                # [batch_size, -1, sampled_size, dim]
                shape = [self.batch_size, -1, self.sample_size, self.node_dim]
                # [batch_size, -1, sampled_size, dim, dim]
                tmp_index = torch.reshape(relations[hop], [self.batch_size, -1, self.sample_size])

                W_r = self.W_R[tmp_index]
                W_ui = self.W_R[self.n_relation]
                para = [W_r, W_ui]

                embeddings = entity_aggregator(self_embeddings=entity_embeddings[hop],
                                               ngh_user_embeddings=ngh_user_embeddings,
                                               ngh_entity_embeddings=torch.reshape(entity_embeddings[hop + 1], shape),
                                               user_embeddings=new_user_embeddings,
                                               item_embeddings=item_embeddings,
                                               parameters=para,
                                               is_item_layer=is_item_layer)

                entity_vectors_next_iter.append(embeddings)
            entity_embeddings = entity_vectors_next_iter

        res = torch.reshape(entity_embeddings[0], [self.batch_size, self.node_dim])

        return res

    def forward(self, user_indices, item_indices):
        # [batch_size, dim]
        new_user_embeddings = self.aggregate_for_users(user_indices)
        new_item_embeddings = self.aggregate_for_items(item_indices, new_user_embeddings)

        scores = torch.sum(new_user_embeddings * new_item_embeddings, dim=-1)
        scores_normalized = torch.sigmoid(scores)

        return scores, scores_normalized

    def evaluate(self, user_indices, item_indices, labels):
        _, scores_normalized = self.forward(user_indices, item_indices)
        scores_normalized = scores_normalized.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores_normalized)
        scores_normalized[scores_normalized >= 0.5] = 1
        scores_normalized[scores_normalized < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores_normalized)
        return auc, f1


    def get_scores(self, user_indices, item_indices):
        _, scores_normalized = self.forward(user_indices, item_indices)
        return scores_normalized

    def train_model(self):
        # Define Eval Value
        max_eval_auc = 0.0
        best_test_auc = 0.0
        best_test_f1 = 0.0
        best_epoch1 = 0
        eval_precision_list = []
        eval_recall_list = []
        eval_ndcg_list = []

        test_precision_list = []
        test_recall_list = []
        test_ndcg_list = []

        best_eval_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        best_epoch2 = [0, 0, 0, 0, 0, 0]


        # Define Dataset
        train_dataset = RecoDataset(self.train_data)
        eval_dataset = RecoDataset(self.eval_data)
        test_dataset = RecoDataset(self.test_data)

        # Define Dataloader
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Define Optimizer
        # for name, param in self.named_parameters():
        #     print(name, param.size())
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, self.args.n_epoch + 1):
            time0 = time()
            total_loss, total_base_loss, total_reg_loss = 0.0, 0.0, 0.0
            iter = 0


            # Training
            for user_indices, item_indices, labels in train_dataloader:
                user_indices = user_indices.long().to(self.device)
                item_indices = item_indices.long().to(self.device)
                labels = labels.float().to(self.device)
                scores, scores_normalized = self.forward(user_indices, item_indices)


                # Base Loss
                self.base_loss = F.binary_cross_entropy(scores_normalized, labels)

                # L2 Loss
                self.l2_loss = l2_loss(
                    self.user_emb_matrix(user_indices),
                    self.entity_emb_matrix(item_indices),
                    self.W_R,
                )

                W = self.user_aggregator.get_weight()
                b = self.user_aggregator.get_bias()
                self.l2_loss = self.l2_loss + l2_loss(
                    W,
                    b,
                )
                for aggregator in self.entity_aggregators_list:
                    W1, W2 = aggregator.get_weight()
                    b1, b2 = aggregator.get_bias()
                    self.l2_loss = self.l2_loss + l2_loss(
                        W1,
                        b1,
                    )
                    if W2 is not None:
                        self.l2_loss = self.l2_loss + l2_loss(W2)
                    if b2 is not None:
                        self.l2_loss = self.l2_loss + l2_loss(b2)

                loss = self.base_loss + self.l2_weight * self.l2_loss
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter += 1

            time_train = time() - time0

            # Evaluate
            if self.args.task == 'ctr':
            # ctr evaluation
                time0 = time()
                train_auc, train_f1 = ctr_evaluate(self, self.train_data, self.args.batch_size)
                eval_auc, eval_f1 = ctr_evaluate(self, self.eval_data, self.args.batch_size)
                test_auc, test_f1 = ctr_evaluate(self, self.test_data, self.args.batch_size)
                if eval_auc > max_eval_auc:
                    max_eval_auc = eval_auc
                    best_test_auc = test_auc
                    best_test_f1 = test_f1
                    best_epoch1 = epoch
                time1 = time() - time0
                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                            % (epoch, total_loss / iter, time_train, time1))
                logging.info(
                    'Train auc: %.4f  f1: %.4f      Eval auc: %.4f  f1: %.4f      Test auc: %.4f  f1: %.4f'
                    % (train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            elif self.args.task == 'topk':
            # top-K evaluation
                time0 = time()
                user_list, train_record, eval_record, item_set, k_list = topk_settings(self.train_data, self.eval_data, self.n_items)
                eval_precision, eval_recall, eval_ndcg = topk_evaluate(self, user_list, train_record, eval_record,
                                                                    item_set, k_list, self.args.batch_size)

                user_list, train_record, test_record, item_set, k_list = topk_settings(self.train_data, self.test_data, self.n_items)
                test_precision, test_recall, test_ndcg = topk_evaluate(self, user_list, train_record, test_record,
                                                                   item_set, k_list, self.args.batch_size)
                time2 = time() - time0

                eval_precision_list.append(eval_precision)
                eval_recall_list.append(eval_recall)
                eval_ndcg_list.append(eval_ndcg)

                test_precision_list.append(test_precision)
                test_recall_list.append(test_recall)
                test_ndcg_list.append(test_ndcg)

                for i, _ in enumerate(k_list):
                    if eval_recall[i] > best_eval_recall[i]:
                        best_eval_recall[i] = eval_recall[i]
                        best_epoch2[i] = epoch  # record the epoch number

                line1 = 'Eval P:'
                for i in eval_precision:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'R:'
                for i in eval_recall:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'NDCG:'
                for i in eval_ndcg:
                    line1 = line1 + '%.4f\t' % i

                line2 = 'Test P:'
                for i in test_precision:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'R:'
                for i in test_recall:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'NDCG:'
                for i in test_ndcg:
                    line2 = line2 + '%.4f\t' % i

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs   Top@k-eval-Time: %.3fs '
                            % (epoch, total_loss / iter, time_train, time2))
                logging.info(line1)
                logging.info(line2)

            elif self.args.task == 'ALL':
                time0 = time()
                eval_auc, eval_f1 = ctr_evaluate(self, self.eval_data, self.args.batch_size)
                test_auc, test_f1 = ctr_evaluate(self, self.test_data, self.args.batch_size)

                if eval_auc > max_eval_auc:
                    max_eval_auc = eval_auc
                    best_test_auc = test_auc
                    best_test_f1 = test_f1
                    best_epoch1 = epoch
                time1 = time() - time0

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))
                logging.info('Eval auc: %.4f  f1: %.4f      Test auc: %.4f  f1: %.4f'
                             % (eval_auc, eval_f1, test_auc, test_f1))

                time0 = time()

                user_list, train_record, eval_record, item_set, k_list = topk_settings(self.train_data, self.eval_data, self.n_items)
                eval_precision, eval_recall, eval_ndcg = topk_evaluate(self, user_list, train_record,
                                                                       eval_record,
                                                                       item_set, k_list, self.args.batch_size)

                user_list, train_record, test_record, item_set, k_list = topk_settings(self.train_data, self.test_data, self.n_items)
                test_precision, test_recall, test_ndcg = topk_evaluate(self, user_list, train_record,
                                                                       test_record,
                                                                       item_set, k_list, self.args.batch_size)
                time1 = time() - time0

                eval_precision_list.append(eval_precision)
                eval_recall_list.append(eval_recall)
                eval_ndcg_list.append(eval_ndcg)

                test_precision_list.append(test_precision)
                test_recall_list.append(test_recall)
                test_ndcg_list.append(test_ndcg)

                for i, _ in enumerate(k_list):
                    if eval_recall[i] > best_eval_recall[i]:
                        best_eval_recall[i] = eval_recall[i]
                        best_epoch2[i] = epoch  # record the epoch number

                line1 = 'Eval P:'
                for i in eval_precision:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'R:'
                for i in eval_recall:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'NDCG:'
                for i in eval_ndcg:
                    line1 = line1 + '%.4f\t' % i

                line2 = 'Test P:'
                for i in test_precision:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'R:'
                for i in test_recall:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'NDCG:'
                for i in test_ndcg:
                    line2 = line2 + '%.4f\t' % i

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))

                logging.info(line1)
                logging.info(line2)

                total_num = get_total_parameters(self)
                logging.info('Total: %d ' % total_num)

            else:
                raise NotImplementedError

        k_list = [1, 5, 10, 20, 50, 100]
        return best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
            eval_precision_list, eval_recall_list, eval_ndcg_list, \
            test_precision_list, test_recall_list, test_ndcg_list