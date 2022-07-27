import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class UserAggregator(nn.Module):
    def __init__(self, args, act_f, name):
        super(UserAggregator,self).__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = args.dropout
        self.act_f = act_f
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.dim = args.node_dim
        self.agg_type = args.agg_type
        self.n_head = args.n_head

        if self.agg_type in ['sum', 'ngh']:
            self.linear = nn.Linear(self.dim, self.dim)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        elif self.agg_type == 'concat':
            self.linear = nn.Linear(self.dim * 2, self.dim)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=1 - self.dropout)

    def _compute_item_ego_embedding(self, self_embeddings, item_ngh_embeddings, W):
        """
        :param self_embeddings:                 [batch_size, dim]
        :param item_ngh_embeddings:             [batch_size, sample_size, dim]
        :param parameters:                      [dim, dim]
        :return:                                [batch_size, dim]
        """
        # [batch_size, 1, dim]
        self_embeddings = torch.reshape(self_embeddings, [self.batch_size, 1, self.dim])
        # [1, dim, dim]
        W = torch.reshape(W, [1, self.dim, self.dim])
        # [batch_size, 1, dim]
        Wu = torch.reshape(torch.sum(W * self_embeddings, dim=-1), [self.batch_size, 1, self.dim])

        # [n_head*batch_size, 1, dim/n_head]
        Wu = torch.cat(torch.split(Wu, int(self.dim / self.n_head), dim=-1), dim=0)
        # [n_head*batch_size, sample_size, dim/n_head]
        item_ngh_embeddings = torch.cat(torch.split(item_ngh_embeddings, int(self.dim / self.n_head), dim=-1), dim=0)
        # [n_head*batch_size, sample_size]
        att = torch.sum(Wu * item_ngh_embeddings, dim=-1) / np.sqrt(float(self.dim) / float(self.n_head))

        # [n_head*batch_size, sample_size]
        att_norm = F.softmax(att, dim=1)
        # [n_head*batch_size, sample_size, 1]
        att_norm = torch.unsqueeze(att_norm, dim=-1)
        # [n_head*batch_size, dim/n_head]
        ego_embeddings = torch.sum(att_norm * item_ngh_embeddings, dim=1)
        # [batch_size, dim]
        ego_embeddings = torch.cat(torch.split(ego_embeddings, self.batch_size, dim=0), dim=-1)

        return ego_embeddings

    def forward(self, self_embeddings, item_ngh_embeddings, W):
        """
        :param self_embeddings:                 [batch_size, dim]
        :param item_ngh_embeddings:             [batch_size, sample_size, dim]
        :param W:                               [dim, dim]
        :return:                                [batch_size, dim]
        """
        # [batch_size, dim]
        ego_embeddings = self._compute_item_ego_embedding(self_embeddings, item_ngh_embeddings, W)

        if self.agg_type == 'sum':
            # [batch_size, dim]
            output = self_embeddings + ego_embeddings
        elif self.agg_type == 'concat':
            # [batch_size, 2 * dim]
            output = torch.cat([self_embeddings, ego_embeddings], dim=-1)
        elif self.agg_type == 'ngh':
            # [batch_size, dim]
            output = ego_embeddings
        else:
            raise NotImplementedError

        output = self.linear(output)

        output = self.dropout(output)

        # [batch_size, dim]
        return self.act_f(output)

    def get_weight(self):
        if self.agg_type in ['sum', 'ngh', 'concat']:
            return self.linear.weight
        else:
            raise NotImplementedError

    def get_bias(self):
        if self.agg_type in ['sum', 'ngh', 'concat']:
            return self.linear.bias
        else:
            raise NotImplementedError


class EntityAggregator(nn.Module):
    def __init__(self, args, act_f, name):
        super(EntityAggregator, self).__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))

        self.name = name
        self.dropout = args.dropout
        self.act_f = act_f
        self.batch_size = args.batch_size
        self.dim = args.node_dim
        self.sample_size = args.sample_size
        self.agg_type = args.agg_type
        self.repr_type = args.repr_type
        self.n_head = args.n_head
        self.a = args.a

        if self.agg_type in ['sum', 'ngh']:
            self.linears = nn.Linear(self.dim, self.dim)
            nn.init.xavier_uniform_(self.linears.weight)
            nn.init.zeros_(self.linears.bias)

            self.linear_UI = nn.Linear(self.dim, self.dim)
            nn.init.xavier_uniform_(self.linear_UI.weight)
            nn.init.zeros_(self.linear_UI.bias)
        elif self.agg_type == 'concat':
            self.linear1 = nn.Linear(self.dim * 2, self.dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.zeros_(self.linear1.bias)
            self.weight_1 = self.linear1.weight
            self.bias_1 = self.linear1.bias

            self.linear2 = nn.Linear(self.dim * 3, self.dim)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
            self.weight_2 = self.linear2.weight
            self.bias_2 = self.linear2.bias

            self.linear_UI = nn.Linear(self.dim * 2, self.dim)
            nn.init.xavier_uniform_(self.linear_UI.weight)
            nn.init.zeros_(self.linear_UI.bias)

        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=1 - self.dropout)

    def _get_representative(self, user_embeddings, item_embeddings):
        """
        :param user_embeddings:         [batch_size, dim]
        :param item_embeddings:         [batch_size, dim]
        :return:                        [batch_size, dim]
        """

        if self.repr_type == 'sum':
            ui_embeddings = (user_embeddings + item_embeddings)
        elif self.repr_type == 'mean':
            ui_embeddings = (user_embeddings + item_embeddings) / 2
        elif self.repr_type == 'max':
            ui_embeddings = torch.where(user_embeddings > item_embeddings, user_embeddings, item_embeddings)
        elif self.repr_type == 'combine':
            ui_embeddings = self.a * user_embeddings + (1.0 - self.a) * item_embeddings
        else:
            raise NotImplementedError

        return ui_embeddings

    def _compute_user_ego_embedding(self, item_embeddings, ngh_user_embeddings, W):
        """
        :param item_embeddings:             [batch_size, dim]
        :param ngh_user_embeddings:         [batch_size, sample_size, dim]
        :param W:                           [dim, dim]
        :return:                            [batch_size, 1, dim]
        """
        # [batch_size, 1 ,dim]
        item_embeddings = torch.reshape(item_embeddings, [self.batch_size, 1, self.dim])
        # [1, dim, dim]
        W = torch.reshape(W, [1, self.dim, self.dim])
        # [batch_size, 1, dim]
        Wi = torch.reshape(torch.sum(W * item_embeddings, dim=-1), [self.batch_size, 1, self.dim])

        # [n_head*batch_size, 1, dim/n_head]
        Wi = torch.cat(torch.split(Wi, int(self.dim / self.n_head), dim=-1), dim=0)
        # [n_head*batch_size, sample_size, dim/n_head]
        ngh_user_embeddings = torch.cat(torch.split(ngh_user_embeddings, int(self.dim / self.n_head), dim=-1), dim=0)

        # [n_head*batch_size, sample_size]
        att = torch.sum(Wi * ngh_user_embeddings, dim=-1) / np.sqrt(float(self.dim) / float(self.n_head))

        # [n_head*batch_size, sample_size]
        att_norm = F.softmax(att, dim=1)
        # [n_head*batch_size, sample_size, 1]
        att_norm = torch.unsqueeze(att_norm, dim=-1)
        # [n_head*batch_size, dim/n_head]
        user_side_ego_embeddings = torch.sum(att_norm * ngh_user_embeddings, dim=1)
        # [batch_size, dim]
        user_side_ego_embeddings = torch.cat(torch.split(user_side_ego_embeddings, self.batch_size, dim=0), dim=-1)
        # [batch_size, 1, dim]
        user_side_ego_embeddings = torch.unsqueeze(user_side_ego_embeddings, dim=1)

        #  for ease of collaborative encoding, we compute the UI side aggregation for items

        if self.agg_type == 'sum':
            # [-1, dim]
            output = torch.reshape(item_embeddings + user_side_ego_embeddings, [-1, self.dim])
            output = self.linear_UI(output)
            output = self.dropout(output)

        elif self.agg_type == 'concat':
            output = torch.cat([item_embeddings, user_side_ego_embeddings], dim=-1)
            # [-1, dim * 2]
            output = self.linear_UI(output)
            output = self.dropout(output)
            output = torch.reshape(output, [-1, self.dim * 2])

        elif self.agg_type == 'ngh':
            # [-1, dim]
            output = torch.reshape(user_side_ego_embeddings, [-1, self.dim])
            output = self.linear_UI(output)
            output = self.dropout(output)

        else:
            raise NotImplementedError
        output = torch.reshape(output, [self.batch_size, -1, self.dim])
        # [batch_size, -1, dim]
        output = self.act_f(output)
        # [batch_size, dim]
        output = torch.squeeze(output)
        return user_side_ego_embeddings, output

    def _compute_entity_ego_embedding(self, user_embeddings, item_embeddings, self_embeddings,
                                      ngh_entity_embeddings, W_r):
        """
        we compute the attention for the quadruplet <(u,i), h, r, t>
        :param user_embeddings:                 [batch_size, dim]
        :param item_embeddings:                 [batch_size, dim]
        :param self_embeddings:                 [batch_size, -1, dim]
        :param ngh_entity_embeddings:           [batch_size, -1, sample_size, dim]
        :param W_r:                             [batch_size, -1, sample_size, dim, dim]
        :return:                                [batch_size, -1, dim]
        """

        # [batch_size, dim]
        signal = self._get_representative(user_embeddings, item_embeddings)
        if self.repr_type == 'concat':
            # [batch_size, 1, 1, 1, dim]
            signal = torch.reshape(signal, [self.batch_size, 1, 1, 1, 2 * self.dim])
            # [batch_size, -1, sample_size, dim, 2 * dim]
            W_r = torch.cat([W_r, W_r], dim=-1)
            # [batch_size, -1, sample_size, dim, 2 * dim]
            W_rui = W_r * signal
            # [batch_size, -1, 1, 1, 2 * dim]
            self_embeddings = torch.cat([self_embeddings, self_embeddings], dim=-1)
            self_embeddings = torch.reshape(self_embeddings, [self.batch_size, -1, 1, 1, 2 * self.dim])
            # [batch_size, -1, sample_size, dim]
            # W_rui_mut_Vh = tf.reduce_sum(W_rui * self_embeddings, axis=-1)
            W_rui_mut_Vh = torch.mean(W_rui * self_embeddings, dim=-1)

        else:
            # [batch_size, 1, 1, 1, dim]
            signal = torch.reshape(signal, [self.batch_size, 1, 1, 1, self.dim])
            # [batch_size, -1, sample_size, dim, dim]
            W_rui = W_r * signal
            # [batch_size, -1, 1, 1, dim]
            self_embeddings = torch.reshape(self_embeddings, [self.batch_size, -1, 1, 1, self.dim])
            # [batch_size, -1, sample_size, dim]
            W_rui_mut_Vh = torch.sum(W_rui * self_embeddings, dim=-1)

        # [n_head*batch_size, -1, sample_size, dim/n_head]
        W_rui_mut_Vh = torch.cat(torch.split(W_rui_mut_Vh, int(self.dim / self.n_head), dim=-1), dim=0)
        ngh_entity_embeddings = torch.cat(torch.split(ngh_entity_embeddings, int(self.dim / self.n_head), dim=-1),
                                          dim=0)
        # [n_head*batch_size, -1, sample_size]
        att = torch.sum(W_rui_mut_Vh * ngh_entity_embeddings, dim=-1) / np.sqrt(
            float(self.dim) / float(self.n_head))

        # [n_head*batch_size, -1, sample_size]
        att_norm = F.softmax(att, dim=-1)
        # [n_head*batch_size, -1, sample_size, 1]
        att_norm = torch.unsqueeze(att_norm, dim=-1)
        # [n_head*batch_size, -1, dim/n_head]
        ego_embedding = torch.sum(att_norm * ngh_entity_embeddings, dim=2)
        # [batch_size, -1, dim]
        ego_embedding = torch.cat(torch.split(ego_embedding, self.batch_size, dim=0), dim=-1)

        return ego_embedding

    def forward(self, self_embeddings, ngh_user_embeddings, ngh_entity_embeddings,
                 user_embeddings, item_embeddings, parameters, is_item_layer):
        """
        :param self_embeddings:                 [batch_size, -1, dim]
        :param ngh_user_embeddings:             [batch_size, sample_size, dim]
        :param ngh_entity_embeddings:           [batch_size, -1, sample_size, dim]
        :param user_embeddings:                 [batch_size, dim]
        :param item_embeddings:                 [batch_size, dim]
        :param parameters:                      W_r, [batch_size, -1, sample_size, dim, dim]

        :return:                                [batch_size, -1, dim]
        """
        _, num, _ = self_embeddings.shape
        W_r, W_ui = parameters
        # parameter1 = [W_r, None]
        # parameter2 = [W_ui, None]

        # For user side
        # [batch_size, 1, dim]
        user_side_ego_embeddings, item_UI_embedding = self._compute_user_ego_embedding(item_embeddings,
                                                                                       ngh_user_embeddings, W_ui)

        if not is_item_layer:

            """
            If it's not the layer for items, we only aggregate the entity neighbors
            """
            # [batch_size, -1, dim]
            entity_ego_embeddings = self._compute_entity_ego_embedding(user_embeddings, item_UI_embedding,
                                                                       self_embeddings, ngh_entity_embeddings, W_r)
            # aggregate them up
            if self.agg_type == 'sum':
                # [-1, dim]
                output = torch.reshape(self_embeddings + entity_ego_embeddings, [-1, self.dim])
                output = self.linears(output)
                output = self.dropout(output)
                # self.weights = self.linears.weight
                # self.bias = self.linears.bias
            elif self.agg_type == 'concat':
                # [-1, dim * 2]
                output = torch.cat([self_embeddings, entity_ego_embeddings], dim=-1)
                output = self.linear1(output)
                output = self.dropout(output)
                # self.weight_1 = self.linear1.weight
                # self.bias_1 = self.linear1.bias
                output = torch.reshape(output, [-1, self.dim * 2])
            elif self.agg_type == 'ngh':
                # [-1, dim]
                output = torch.reshape(entity_ego_embeddings, [-1, self.dim])
                output = self.linears(output)
                output = self.dropout(output)
                # self.weights = self.linears.weight
                # self.bias = self.linears.bias
            else:
                raise NotImplementedError

            output = torch.reshape(output, [self.batch_size, -1, self.dim])

            # [batch_size, -1, dim]
            return self.act_f(output)

        else:
            """
            If it is the layer for items, we aggregate the user part and entity part together
            """
            # For entity side
            # [batch_size, -1, dim]
            entity_ego_embeddings = self._compute_entity_ego_embedding(user_embeddings, item_UI_embedding,
                                                                       self_embeddings, ngh_entity_embeddings, W_r)
            # aggregate them up
            if self.agg_type == 'sum':
                # [-1, dim]

                output = torch.reshape(self_embeddings + user_side_ego_embeddings + entity_ego_embeddings,
                                       [-1, self.dim])
                output = self.linears(output)
                output = self.dropout(output)
                # self.weights = self.linears.weight
                # self.bias = self.linears.bias
            elif self.agg_type == 'concat':
                output = torch.cat([self_embeddings, user_side_ego_embeddings, entity_ego_embeddings], dim=-1)
                # [-1, dim * 2]
                output = torch.reshape(output, [-1, self.dim * 3])
                output = self.linear2(output)
                output = self.dropout(output)
                # self.weight_2 = self.linear2.weight
                # self.bias_2 = self.linear2.bias
            elif self.agg_type == 'ngh':
                # [-1, dim]
                output = torch.reshape(entity_ego_embeddings + user_side_ego_embeddings, [-1, self.dim])
                output = self.linears(output)
                output = self.dropout(output)
                # self.weights = self.linears.weight
                # self.bias = self.linears.bias
            else:
                raise NotImplementedError

            output = torch.reshape(output, [self.batch_size, -1, self.dim])

            # [batch_size, -1 ,dim]
            return self.act_f(output)

    def get_weight(self):
        if self.agg_type in ['sum', 'ngh']:
            return self.linears.weight, None
        elif self.agg_type == 'concat':
            return self.linear1.weight, self.linear2.weight
        else:
            raise NotImplementedError

    def get_bias(self):
        if self.agg_type in ['sum', 'ngh']:
            return self.linears.bias, None
        elif self.agg_type == 'concat':
            return self.linear1.bias, self.linear2.bias
        else:
            raise NotImplementedError
