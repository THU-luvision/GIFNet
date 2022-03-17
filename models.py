import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907, add trainable mask.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('bnc,cd -> bnd', input, self.weight)
        output = torch.einsum('mn,bnd -> bmd', adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolutionLayer(nhid, nout, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj=None):
        """
        x in shape [batch_size, num_node, feat_dim]
        adj in shape [num_node, num_node]
        """
        if adj is None:
            batch_size, num_node, _ = x.size()
            adj = torch.ones((num_node, num_node)).to(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, use_face=False, encoder_lstm_layer=1):
        super(GCNEncoder, self).__init__()
        feat_dim_index = 1
        if use_face:
            self.neib_face_lstm_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
            feat_dim_index = 2
        self.neib_traj_lstm_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
        self.gcn_net = GCN(nfeat * feat_dim_index,
                           nhid * feat_dim_index,
                           nout * feat_dim_index,
                           dropout)
        self.nfeat = nfeat
        self.use_face = use_face

    def neib_lstm_init_hidden(self, batch):
        h = torch.zeros(1, batch, self.nfeat).cuda()
        c = torch.zeros(1, batch, self.nfeat).cuda()
        return h, c

    def forward(self, obs_traj_rel, neib_traj_self, **kwargs):
        """
        :param obs_traj_rel: (9, batch_size, 2)
        :param neib_traj_self: relative to themselves t-1: [(9, N_neib, 2), ... (batch_size)]
        """
        if self.use_face:
            assert 'obs_face' in kwargs.keys() and 'neib_face_abs' in kwargs.keys()
            obs_face = kwargs['obs_face']
            neib_face_abs = kwargs['neib_face_abs']

        graph_embeded_data = []
        for i in range(len(neib_traj_self)):
            curr_obs_traj_rel = obs_traj_rel[:, i]  # (9, 2)
            curr_obs_traj_rel = curr_obs_traj_rel.unsqueeze(dim=1)
            curr_neib_traj_self = neib_traj_self[i]  # (9, N_neib, 2)
            curr_neib_traj_self = torch.cat((curr_obs_traj_rel, curr_neib_traj_self), dim=1)

            batch = curr_neib_traj_self.shape[1]
            # neighbour_feat_lstm_h_t, neighbour_feat_lstm_c_t = self.init_hidden_neighbour_lstm(batch)
            neib_traj_state = self.neib_lstm_init_hidden(batch)
            # propagate for feat lstm
            neib_hidden_states, _ = self.neib_traj_lstm_encoder(curr_neib_traj_self, neib_traj_state)  # [9, N, 32]

            if self.use_face:
                curr_obs_face = obs_face[:, i]  # (9, 2)
                curr_obs_face = curr_obs_face.unsqueeze(dim=1)
                curr_neib_face_abs = neib_face_abs[i]  # (9, N_neib, 2)
                curr_neib_face_abs = torch.cat((curr_obs_face, curr_neib_face_abs), dim=1)

                neib_face_state = self.neib_lstm_init_hidden(batch)
                neib_face_hidden_states, _ = self.neib_face_lstm_encoder(curr_neib_face_abs, neib_face_state)
                neib_hidden_states = torch.cat((neib_hidden_states, neib_face_hidden_states), dim=-1)  # [9, N, 64]

            single_embedding = self.gcn_net(neib_hidden_states).squeeze()

            if len(single_embedding.shape) == 3:
                single_embedding = single_embedding[:, 0, :]

            graph_embeded_data.append(single_embedding)
        graph_embeded_data = torch.stack(graph_embeded_data).permute(1, 0, 2)  # [9, batch_size, 32/64]
        return graph_embeded_data


class SocialAttentionEncoder(nn.Module):
    def __init__(self, nfeat, mlp_bottleneck_dim, encoder_lstm_layer=1, use_face=False):
        super(SocialAttentionEncoder, self).__init__()
        feat_dim_index = 1
        if use_face:
            self.neib_face_abs_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
            self.neib_face_rel_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
            feat_dim_index = 2

        self.neib_traj_self_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
        self.neib_traj_rel_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)

        self.nfeat = nfeat
        self.use_face = use_face

        mlp_dims = [nfeat * feat_dim_index, mlp_bottleneck_dim * feat_dim_index]  # Fixme: only use 1-layer mlp
        self.mlp_pre_attn = make_mlp(mlp_dims)
        self.attn = nn.Linear(mlp_bottleneck_dim * feat_dim_index, 1)

    def neib_lstm_init_hidden(self, batch):
        h = torch.zeros(1, batch, self.nfeat).cuda()
        c = torch.zeros(1, batch, self.nfeat).cuda()
        return h, c

    def forward(self, obs_traj_rel, neib_traj_rel, neib_traj_self, **kwargs):
        if self.use_face:
            assert 'obs_face' in kwargs.keys()
            assert 'neib_face_abs' in kwargs.keys()
            assert 'neib_face_rel' in kwargs.keys()
            obs_face = kwargs['obs_face']
            neib_face_abs = kwargs['neib_face_abs']
            neib_face_rel = kwargs['neib_face_rel']

        graph_embeded_data = []
        for i in range(len(neib_traj_self)):
            curr_obs_traj_rel = obs_traj_rel[:, i]  # (9, 2)
            curr_obs_traj_rel = curr_obs_traj_rel.unsqueeze(dim=1)
            curr_neib_traj_rel = neib_traj_rel[i]  # (9, N_neib, 2)
            curr_neib_traj_self = neib_traj_self[i]  # (9, N_neib, 2)

            curr_neib_traj_rel = torch.cat((torch.zeros_like(curr_obs_traj_rel), curr_neib_traj_rel), dim=1)
            curr_neib_traj_self = torch.cat((curr_obs_traj_rel, curr_neib_traj_self), dim=1)

            batch = curr_neib_traj_self.shape[1]
            neib_traj_rel_state = self.neib_lstm_init_hidden(batch)
            neib_traj_self_state = self.neib_lstm_init_hidden(batch)

            # propagate for attn
            neib_hidden_attn, _ = self.neib_traj_rel_encoder(curr_neib_traj_rel, neib_traj_rel_state)  # [9, N, 32]
            # propagate for feat
            neib_hidden_feat, _ = self.neib_traj_self_encoder(curr_neib_traj_self, neib_traj_self_state)  # [9, N, 32]

            if self.use_face:
                curr_obs_face = obs_face[:, i]  # (9, 2)
                curr_obs_face = curr_obs_face.unsqueeze(dim=1)
                curr_neib_face_rel = neib_face_rel[i]  # (9, N_neib, 2)
                curr_neib_face_abs = neib_face_abs[i]  # (9, N_neib, 2)

                curr_neib_face_rel = torch.cat((torch.zeros_like(curr_obs_face), curr_neib_face_rel), dim=1)
                curr_neib_face_abs = torch.cat((curr_obs_face, curr_neib_face_abs), dim=1)

                neib_face_rel_state = self.neib_lstm_init_hidden(batch)
                neib_face_abs_state = self.neib_lstm_init_hidden(batch)

                neib_face_rel_hidden, _ = self.neib_face_rel_encoder(curr_neib_face_rel, neib_face_rel_state)
                neib_face_abs_hidden, _ = self.neib_face_abs_encoder(curr_neib_face_abs, neib_face_abs_state)

                neib_hidden_attn = torch.cat((neib_hidden_attn, neib_face_rel_hidden), dim=-1)  # [9, N, 64]
                neib_hidden_feat = torch.cat((neib_hidden_feat, neib_face_abs_hidden), dim=-1)  # [9, N, 64]

            if neib_hidden_attn.shape[1] == 1:  # no neib
                graph_embeded_data.append(neib_hidden_feat.squeeze())
            else:
                attn_h = self.mlp_pre_attn(neib_hidden_attn)  # [9, N, 32/64]
                attn_w = self.attn(attn_h).squeeze()  # [9, N]
                attn_w = F.softmax(attn_w, dim=1).unsqueeze(dim=2)  # [9, N, 1]
                single_embedding = torch.sum(neib_hidden_feat * attn_w, dim=1)  # [9, 32/64]
                graph_embeded_data.append(single_embedding)

        graph_embeded_data = torch.stack(graph_embeded_data).permute(1, 0, 2)  # [9, batch_size, 32/64]
        return graph_embeded_data


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, alpha, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)  # torch.Size([8, 4, 137, 137])  [obs_length, batch_size, node_num, node_num]
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.n_head)
                + " -> "
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in, f_out=n_units[i + 1],
                                             attn_dropout=dropout, alpha=alpha)
            )  # 4 32 16   1 64 32

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATWithAttention(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GATWithAttention, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout, alpha=alpha
                )
            )  # 4 32 16   1 64 32

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x, return_attn=False):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x if not return_attn else (x, attn)


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data


class GATEncoderWithAttention(nn.Module):
    def __init__(self, n_units, n_heads, nfeat, dropout, alpha, use_face=False, encoder_lstm_layer=1):
        super(GATEncoderWithAttention, self).__init__()

        if use_face:
            self.neib_face_abs_encoder = nn.LSTM(2, int(nfeat / 2), encoder_lstm_layer)
            self.neib_face_rel_encoder = nn.LSTM(2, int(nfeat / 2), encoder_lstm_layer)
            self.neib_traj_self_encoder = nn.LSTM(2, int(nfeat / 2), encoder_lstm_layer)
            self.neib_traj_rel_encoder = nn.LSTM(2, int(nfeat / 2), encoder_lstm_layer)
        else:
            self.neib_traj_self_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)
            self.neib_traj_rel_encoder = nn.LSTM(2, nfeat, encoder_lstm_layer)

        self.gat_net = GATWithAttention(n_units, n_heads, dropout, alpha)
        self.nfeat = nfeat
        self.use_face = use_face

        print('=================== GAT =====================')
        print('n_units', n_units)
        print('n_heads', n_heads)
        print('nfeat', int(nfeat / 2) if use_face else nfeat)
        print('============================================')

    def neib_lstm_init_hidden(self, batch):
        if self.use_face:
            h = torch.zeros(1, batch, int(self.nfeat / 2)).cuda()
            c = torch.zeros(1, batch, int(self.nfeat / 2)).cuda()
        else:
            h = torch.zeros(1, batch, self.nfeat).cuda()
            c = torch.zeros(1, batch, self.nfeat).cuda()
        return h, c

    def forward(self, obs_traj_rel, neib_traj_rel, neib_traj_self, **kwargs):
        """
        :param obs_traj_rel: (9, batch_size, 2)
        :param neib_traj_rel: relative to main person: [(9, N_neib, 2), ... (batch_size)]
        :param neib_traj_self: relative to themselves t-1: [(9, N_neib, 2), ... (batch_size)]
        """
        if self.use_face:
            assert 'obs_face' in kwargs.keys()
            assert 'neib_face_abs' in kwargs.keys()
            assert 'neib_face_rel' in kwargs.keys()
            obs_face = kwargs['obs_face']
            neib_face_abs = kwargs['neib_face_abs']
            neib_face_rel = kwargs['neib_face_rel']

        graph_embeded_data = []
        for i in range(len(neib_traj_rel)):
            curr_obs_traj_rel = obs_traj_rel[:, i]  # (9, 2)
            curr_obs_traj_rel = curr_obs_traj_rel.unsqueeze(dim=1)
            curr_neib_traj_rel = neib_traj_rel[i]  # (9, N_neib, 2)
            curr_neib_traj_self = neib_traj_self[i]  # (9, N_neib, 2)

            curr_neib_traj_rel = torch.cat((torch.zeros_like(curr_obs_traj_rel), curr_neib_traj_rel), dim=1)
            curr_neib_traj_self = torch.cat((curr_obs_traj_rel, curr_neib_traj_self), dim=1)

            batch = curr_neib_traj_rel.shape[1]
            neib_traj_rel_state = self.neib_lstm_init_hidden(batch)
            neib_traj_self_state = self.neib_lstm_init_hidden(batch)

            # propagate for attn
            neib_hidden_attn, _ = self.neib_traj_rel_encoder(curr_neib_traj_rel, neib_traj_rel_state)  # [9, N, 32]
            # propagate for feat
            neib_hidden_feat, _ = self.neib_traj_self_encoder(curr_neib_traj_self, neib_traj_self_state)  # [9, N, 32]

            if self.use_face:
                curr_obs_face = obs_face[:, i]  # (9, 2)
                curr_obs_face = curr_obs_face.unsqueeze(dim=1)
                curr_neib_face_rel = neib_face_rel[i]  # (9, N_neib, 2)
                curr_neib_face_abs = neib_face_abs[i]  # (9, N_neib, 2)

                curr_neib_face_rel = torch.cat((torch.zeros_like(curr_obs_face), curr_neib_face_rel), dim=1)
                curr_neib_face_abs = torch.cat((curr_obs_face, curr_neib_face_abs), dim=1)

                neib_face_rel_state = self.neib_lstm_init_hidden(batch)
                neib_face_abs_state = self.neib_lstm_init_hidden(batch)

                neib_face_rel_hidden, _ = self.neib_face_rel_encoder(curr_neib_face_rel, neib_face_rel_state)
                neib_face_abs_hidden, _ = self.neib_face_abs_encoder(curr_neib_face_abs, neib_face_abs_state)

                neib_hidden_attn = torch.cat((neib_hidden_attn, neib_face_rel_hidden), dim=-1)  # [9, N, 64]
                neib_hidden_feat = torch.cat((neib_hidden_feat, neib_face_abs_hidden), dim=-1)  # [9, N, 64]

            attn_feat, attn = self.gat_net(neib_hidden_attn, return_attn=True)  # [9, 1, N, N]

            single_embedding = torch.matmul(attn, neib_hidden_feat.unsqueeze(1)).squeeze()

            if len(single_embedding.shape) == 3:
                single_embedding = single_embedding[:, 0, :]

            graph_embeded_data.append(single_embedding)
        graph_embeded_data = torch.stack(graph_embeded_data).permute(1, 0, 2)  # [9, batch_size, 32]
        return graph_embeded_data


class ProxemicsFieldGenerator(nn.Module):
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            n_units,
            n_heads,
            graph_network_out_dims,
            dropout,
            alpha,
            graph_lstm_hidden_size,
            lstm_layers=1,
            noise_dim=(8,),
            noise_type="gaussian",
            graph_mode='gcn',
            use_face=False,
            feat_concat_samp_coef=1
    ):
        super(ProxemicsFieldGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.use_face = use_face
        self.feat_concat_samp_coef = feat_concat_samp_coef
        self.graph_mode = graph_mode

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size

        self.traj_lstm_input_size = traj_lstm_input_size

        if graph_mode == 'gcn':
            nfeat, nhid, nout = n_units
            self.gcnencoder = GCNEncoder(nfeat=nfeat, nhid=nhid, nout=nout, dropout=dropout, use_face=use_face)
        elif graph_mode == 'attn':
            nfeat, nhid, nout = n_units
            self.saencoder = SocialAttentionEncoder(nfeat=nfeat, mlp_bottleneck_dim=nfeat, use_face=use_face)
        elif graph_mode == 'gat':
            nfeat = n_units[-1]
            self.gatencoder = GATEncoderWithAttention(n_units, n_heads, nfeat, dropout, alpha, use_face=use_face)

        actual_lstm_hidden_size = int(self.traj_lstm_hidden_size / feat_concat_samp_coef) \
            if use_face else self.traj_lstm_hidden_size

        if use_face:
            self.face_lstm_encoder = nn.LSTM(traj_lstm_input_size, actual_lstm_hidden_size, lstm_layers)
            self.traj_lstm_encoder = nn.LSTM(traj_lstm_input_size, actual_lstm_hidden_size, lstm_layers)
            self.traj_hidden2pos = nn.Linear(int(2 * self.traj_lstm_hidden_size / feat_concat_samp_coef), 2)
            self.traj_gat_hidden2pos = nn.Linear(int(2 * self.traj_lstm_hidden_size / feat_concat_samp_coef)
                                                 + self.graph_lstm_hidden_size, 2)
            self.pred_lstm_hidden_size = (int(2 * self.traj_lstm_hidden_size / feat_concat_samp_coef)
                                          + self.graph_lstm_hidden_size + noise_dim[0])
        else:
            self.traj_lstm_encoder = nn.LSTM(traj_lstm_input_size, self.traj_lstm_hidden_size, lstm_layers)
            self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)
            self.traj_gat_hidden2pos = nn.Linear(self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2)
            self.pred_lstm_hidden_size = (self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0])

        self.graph_lstm_encoder = nn.LSTM(graph_network_out_dims, graph_lstm_hidden_size, lstm_layers)
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, self.pred_lstm_hidden_size)
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)

        print('============================================')
        print('use face: ', self.use_face)
        print('graph mode: ', self.graph_mode)
        print('feat concat samp coef', feat_concat_samp_coef)
        print('traj / face lstm encoder: {}-{}-{}'.format(traj_lstm_input_size, actual_lstm_hidden_size, lstm_layers))
        print('graph lstm encoder: {}-{}-{}'.format(graph_network_out_dims, graph_lstm_hidden_size, lstm_layers))
        print('pred lstm encoder: {}-{}   z: {}'.format(traj_lstm_input_size, self.pred_lstm_hidden_size, noise_dim[0]))
        print('============================================')

    def traj_lstm_init_hidden(self, batch):
        actual_lstm_hidden_size = int(self.traj_lstm_hidden_size / self.feat_concat_samp_coef)\
            if self.use_face else self.traj_lstm_hidden_size
        h = torch.zeros(1, batch, actual_lstm_hidden_size).cuda()
        c = torch.zeros(1, batch, actual_lstm_hidden_size).cuda()
        return h, c

    def graph_lstm_init_hidden(self, batch):
        h = torch.zeros(1, batch, self.graph_lstm_hidden_size).cuda()
        c = torch.zeros(1, batch, self.graph_lstm_hidden_size).cuda()
        return h, c

    def add_noise(self, _input, decoder_z=None):
        if decoder_z is None:
            noise_shape = (1,) + self.noise_dim
            z_decoder = get_noise(noise_shape, self.noise_type)
        else:
            z_decoder = decoder_z.cuda()
            if len(z_decoder.shape) == 1:
                z_decoder = torch.unsqueeze(z_decoder, 0)
        _vec = z_decoder[0].view(1, -1)
        _to_cat = _vec.repeat(_input.shape[0], 1)
        decoder_h = torch.cat([_input, _to_cat], dim=1)
        return decoder_h

    def forward(
            self,
            obs_traj_rel,
            obs_traj,
            obs_face,
            neib_traj_rel,
            neib_traj_self,
            neib_face_abs,
            neib_face_rel,
            training_step,
            teacher_forcing_ratio=0.5,
            decoder_z=None
    ):
        batch = obs_traj_rel.shape[1]
        traj_state = self.traj_lstm_init_hidden(batch)
        graph_state = self.graph_lstm_init_hidden(batch)

        pred_traj_rel = []

        if self.use_face:
            face_state = self.traj_lstm_init_hidden(batch)
            traj_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            face_hidden_states, _ = self.face_lstm_encoder(obs_face[: self.obs_len], face_state)
            step_1_hidden_states = torch.cat((traj_hidden_states, face_hidden_states), dim=-1)
            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_traj_rel = self.traj_hidden2pos(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        else:
            step_1_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_traj_rel = self.traj_hidden2pos(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        # training_step 2: use linear transform of concat of (traj lstm hidden, group lstm hidden) as output
        if training_step == 2 or training_step == 3:
            if self.graph_mode == 'gcn':
                if self.use_face:
                    graph_lstm_input = self.gcnencoder(obs_traj_rel[: self.obs_len],
                                                       neib_traj_self,
                                                       obs_face=obs_face,
                                                       neib_face_abs=neib_face_abs)  # [9, bs, 64]
                else:
                    graph_lstm_input = self.gcnencoder(obs_traj_rel[: self.obs_len], neib_traj_self)  # [9, bs, 32]
            elif self.graph_mode == 'attn':
                if self.use_face:
                    graph_lstm_input = self.saencoder(obs_traj_rel[: self.obs_len],
                                                      neib_traj_rel,
                                                      neib_traj_self,
                                                      obs_face=obs_face,
                                                      neib_face_abs=neib_face_abs,
                                                      neib_face_rel=neib_face_rel)  # [9, bs, 64]
                else:
                    graph_lstm_input = self.saencoder(obs_traj_rel[: self.obs_len],
                                                      neib_traj_rel,
                                                      neib_traj_self)  # [9, bs, 32]
            elif self.graph_mode == 'gat':
                if self.use_face:
                    graph_lstm_input = self.gatencoder(obs_traj_rel[: self.obs_len],
                                                       neib_traj_rel,
                                                       neib_traj_self,
                                                       obs_face=obs_face,
                                                       neib_face_abs=neib_face_abs,
                                                       neib_face_rel=neib_face_rel)  # [9, bs, 64]
                else:
                    graph_lstm_input = self.gatencoder(obs_traj_rel[: self.obs_len],
                                                       neib_traj_rel,
                                                       neib_traj_self)  # [9, bs, 32]

        if training_step == 2:
            graph_hidden_states, _ = self.graph_lstm_encoder(graph_lstm_input, graph_state)
            step_2_hidden_states = torch.cat((step_1_hidden_states, graph_hidden_states), dim=-1)
            pred_traj_rel = self.traj_gat_hidden2pos(step_2_hidden_states)

        # return linear transformation of lstm output as prediction result
        if training_step == 1 or training_step == 2:
            return pred_traj_rel
        else:
            graph_hidden_states, _ = self.graph_lstm_encoder(graph_lstm_input, graph_state)
            encoded_before_noise_hidden = torch.cat(
                (step_1_hidden_states[-1], graph_hidden_states[-1]), dim=-1)  # [bs, 64]

            pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, decoder_z=decoder_z)  # [bs, 80]
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()  # ct is initialized as 0
            traj_output = obs_traj_rel[self.obs_len - 1]
            if self.training:
                for i, input_t in enumerate(
                        obs_traj_rel[-self.pred_len:].chunk(obs_traj_rel[-self.pred_len:].size(0), dim=0)
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else traj_output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    traj_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [traj_output]
            else:
                # when validation, inference in auto-regressive manner
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        traj_output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    traj_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [traj_output]

            traj_outputs = torch.stack(pred_traj_rel)
            return traj_outputs


class ProxemicsFieldGenerator_WO_GAT(nn.Module):
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            noise_dim=(8,),
            noise_type="gaussian",
            lstm_layers=1,
            use_face=False,
            feat_concat_samp_coef=1
    ):
        super(ProxemicsFieldGenerator_WO_GAT, self).__init__()

        print('############## model ##############')
        print('use face: ', use_face)
        print('feat concat samp coef', feat_concat_samp_coef)
        print('###################################')
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.use_face = use_face
        self.feat_concat_samp_coef = feat_concat_samp_coef

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        if use_face:
            self.face_lstm_encoder = nn.LSTM(traj_lstm_input_size,
                                             int(traj_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.traj_lstm_encoder = nn.LSTM(traj_lstm_input_size,
                                             int(traj_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.pred_lstm_hidden_size = (int(2 * self.traj_lstm_hidden_size / feat_concat_samp_coef) + noise_dim[0])
            self.traj_hidden2pos = nn.Linear(int(2 * self.traj_lstm_hidden_size / feat_concat_samp_coef), 2)
        else:
            self.traj_lstm_encoder = nn.LSTM(traj_lstm_input_size, traj_lstm_hidden_size, lstm_layers)
            self.pred_lstm_hidden_size = (self.traj_lstm_hidden_size + noise_dim[0])
            self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)

        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, self.pred_lstm_hidden_size)
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)

    def traj_lstm_init_hidden(self, batch):
        if self.use_face:
            h = torch.zeros(1, batch, int(self.traj_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
            c = torch.zeros(1, batch, int(self.traj_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
        else:
            h = torch.zeros(1, batch, self.traj_lstm_hidden_size).cuda()
            c = torch.zeros(1, batch, self.traj_lstm_hidden_size).cuda()
        return h, c

    def add_noise(self, _input, decoder_z=None):
        if decoder_z is None:
            noise_shape = (1,) + self.noise_dim
            z_decoder = get_noise(noise_shape, self.noise_type)
        else:
            z_decoder = decoder_z.cuda()
            if len(z_decoder.shape) == 1:
                z_decoder = torch.unsqueeze(z_decoder, 0)
        _vec = z_decoder[0].view(1, -1)
        _to_cat = _vec.repeat(_input.shape[0], 1)
        decoder_h = torch.cat([_input, _to_cat], dim=1)
        return decoder_h

    def forward(
            self,
            obs_traj_rel,
            obs_face,
            training_step,
            teacher_forcing_ratio=0.5,
            decoder_z=None
    ):
        batch = obs_traj_rel.shape[1]
        traj_state = self.traj_lstm_init_hidden(batch)
        pred_traj_rel = []

        if self.use_face:
            face_state = self.traj_lstm_init_hidden(batch)
            traj_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            face_hidden_states, _ = self.face_lstm_encoder(obs_face[: self.obs_len], face_state)
            step_1_hidden_states = torch.cat((traj_hidden_states, face_hidden_states), dim=-1)
            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_traj_rel = self.traj_hidden2pos(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        else:
            step_1_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_traj_rel = self.traj_hidden2pos(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        # return linear transformation of lstm output as prediction result
        if training_step == 1:
            return pred_traj_rel
        elif training_step == 2:
            encoded_before_noise_hidden = step_1_hidden_states[-1]
            pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, decoder_z=decoder_z)  # [bs, 80]
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()  # ct is initialized as 0
            traj_output = obs_traj_rel[self.obs_len - 1]
            if self.training:
                for i, input_t in enumerate(
                        obs_traj_rel[-self.pred_len:].chunk(obs_traj_rel[-self.pred_len:].size(0), dim=0)
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else traj_output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    traj_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [traj_output]
            else:
                # when validation, inference in auto-regressive manner
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        traj_output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    traj_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [traj_output]

            traj_outputs = torch.stack(pred_traj_rel)
            return traj_outputs


class AttentionFieldGenerator(nn.Module):
    def __init__(
            self,
            obs_len,
            pred_len,
            face_lstm_input_size,
            face_lstm_hidden_size,
            n_units,
            n_heads,
            graph_network_out_dims,
            dropout,
            alpha,
            graph_lstm_hidden_size,
            lstm_layers=1,
            graph_mode='gcn',
            use_traj=False,
            feat_concat_samp_coef=1
    ):
        super(AttentionFieldGenerator, self).__init__()

        print('############## model ##############')
        print('use traj: ', use_traj)
        print('graph mode: ', graph_mode)
        print('feat concat samp coef', feat_concat_samp_coef)
        print('###################################')

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.use_traj = use_traj
        self.feat_concat_samp_coef = feat_concat_samp_coef
        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.face_lstm_hidden_size = face_lstm_hidden_size
        self.face_lstm_input_size = face_lstm_input_size
        self.lstm_layers = lstm_layers

        if graph_mode == 'gcn':
            nfeat, nhid, nout = n_units
            self.gcnencoder = GCNEncoder(nfeat=nfeat, nhid=nhid, nout=nout, dropout=dropout, use_face=use_traj)
        elif graph_mode == 'gat':
            nfeat = n_units[-1]
            self.gatencoder = GATEncoderWithAttention(n_units, n_heads, nfeat, dropout, alpha, use_face=use_traj)

        self.graph_mode = graph_mode

        if use_traj:
            self.traj_lstm_encoder = nn.LSTM(face_lstm_input_size,
                                             int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.face_lstm_encoder = nn.LSTM(face_lstm_input_size,
                                             int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.face_hidden2ori = nn.Linear(int(2 * face_lstm_hidden_size / feat_concat_samp_coef), face_lstm_input_size)
            self.face_gat_hidden2ori = nn.Linear(int(2 * face_lstm_hidden_size / feat_concat_samp_coef)
                                                 + self.graph_lstm_hidden_size, face_lstm_input_size)
            self.pred_lstm_hidden_size = (int(2 * self.face_lstm_hidden_size / feat_concat_samp_coef)
                                          + self.graph_lstm_hidden_size)
        else:
            self.face_lstm_encoder = nn.LSTM(face_lstm_input_size, face_lstm_hidden_size, lstm_layers)
            self.face_hidden2ori = nn.Linear(face_lstm_hidden_size, face_lstm_input_size)
            self.face_gat_hidden2ori = nn.Linear(face_lstm_hidden_size + self.graph_lstm_hidden_size, face_lstm_input_size)
            self.pred_lstm_hidden_size = (self.face_lstm_hidden_size + self.graph_lstm_hidden_size)

        self.graph_lstm_encoder = nn.LSTM(graph_network_out_dims, graph_lstm_hidden_size, lstm_layers)
        self.pred_lstm_model = nn.LSTMCell(face_lstm_input_size, self.pred_lstm_hidden_size)
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, face_lstm_input_size)

        print('============================================')
        print('use traj: ', self.use_traj)
        print('graph mode: ', self.graph_mode)
        print('feat concat samp coef', feat_concat_samp_coef)
        print('face / traj lstm encoder: {}-{}-{}'.format(face_lstm_input_size,
              int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers))
        print('graph lstm encoder: {}-{}-{}'.format(graph_network_out_dims, graph_lstm_hidden_size, lstm_layers))
        print('pred lstm encoder: {}-{}'.format(face_lstm_input_size, self.pred_lstm_hidden_size))
        print('============================================')

    def face_lstm_init_hidden(self, batch):
        if self.use_traj:
            h = torch.zeros(self.lstm_layers, batch, int(self.face_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
            c = torch.zeros(self.lstm_layers, batch, int(self.face_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
        else:
            h = torch.zeros(self.lstm_layers, batch, self.face_lstm_hidden_size).cuda()
            c = torch.zeros(self.lstm_layers, batch, self.face_lstm_hidden_size).cuda()
        return h, c

    def graph_lstm_init_hidden(self, batch):
        h = torch.zeros(1, batch, self.graph_lstm_hidden_size).cuda()
        c = torch.zeros(1, batch, self.graph_lstm_hidden_size).cuda()
        return h, c

    def forward(
            self,
            obs_traj_rel,
            obs_face_rel,
            neib_traj_rel,
            neib_traj_self,
            neib_face_abs,
            neib_face_rel,
            training_step,
            teacher_forcing_ratio=0.5
    ):
        batch = obs_face_rel.shape[1]
        face_state = self.face_lstm_init_hidden(batch)
        graph_state = self.graph_lstm_init_hidden(batch)

        pred_ori = []

        if self.use_traj:
            traj_state = self.face_lstm_init_hidden(batch)
            traj_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            face_hidden_states, _ = self.face_lstm_encoder(obs_face_rel[: self.obs_len], face_state)
            step_1_hidden_states = torch.cat((traj_hidden_states, face_hidden_states), dim=-1)

            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_ori = self.face_hidden2ori(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        else:
            step_1_hidden_states, _ = self.face_lstm_encoder(obs_face_rel[: self.obs_len], face_state)
            if training_step == 1:
                # training_step 1: use linear transform of traj lstm hidden as output
                pred_ori = self.face_hidden2ori(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        # training_step 2: use linear transform of concat of (traj lstm hidden, group lstm hidden) as output
        if training_step == 2 or training_step == 3:
            if self.graph_mode == 'gcn':
                if self.use_traj:
                    graph_lstm_input = self.gcnencoder(obs_face_rel[: self.obs_len],
                                                       neib_face_abs,
                                                       obs_face=obs_traj_rel,
                                                       neib_face_abs=neib_traj_self)  # [9, bs, 64]
                else:
                    graph_lstm_input = self.gcnencoder(obs_face_rel[: self.obs_len], neib_face_abs)  # [9, bs, 32]
            elif self.graph_mode == 'gat':
                if self.use_traj:
                    graph_lstm_input = self.gatencoder(obs_face_rel[: self.obs_len],
                                                       neib_face_rel,
                                                       neib_face_abs,
                                                       obs_face=obs_traj_rel,
                                                       neib_face_abs=neib_traj_self,
                                                       neib_face_rel=neib_traj_rel)  # [9, bs, 64]
                else:
                    graph_lstm_input = self.gatencoder(obs_face_rel[: self.obs_len],
                                                       neib_face_rel,
                                                       neib_face_abs)  # [9, bs, 32]

        if training_step == 2:
            graph_hidden_states, _ = self.graph_lstm_encoder(graph_lstm_input, graph_state)
            step_2_hidden_states = torch.cat((step_1_hidden_states, graph_hidden_states), dim=-1)
            pred_ori = self.face_gat_hidden2ori(step_2_hidden_states)

        # return linear transformation of lstm output as prediction result
        if training_step == 1 or training_step == 2:
            return pred_ori
        else:
            graph_hidden_states, _ = self.graph_lstm_encoder(graph_lstm_input, graph_state)
            pred_lstm_hidden = torch.cat(
                (step_1_hidden_states[-1], graph_hidden_states[-1]), dim=-1)  # [bs, 64]

            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()  # ct is initialized as 0
            ori_output = obs_face_rel[self.obs_len - 1]
            if self.training:
                for i, input_t in enumerate(
                        obs_face_rel[-self.pred_len:].chunk(obs_face_rel[-self.pred_len:].size(0), dim=0)
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else ori_output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    ori_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_ori += [ori_output]
            else:
                # when validation, inference in auto-regressive manner
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        ori_output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    ori_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_ori += [ori_output]

            ori_outputs = torch.stack(pred_ori)
            return ori_outputs


class AttentionFieldGenerator_WO_GAT(nn.Module):
    def __init__(
            self,
            obs_len,
            pred_len,
            face_lstm_input_size,
            face_lstm_hidden_size,
            lstm_layers=1,
            use_traj=False,
            feat_concat_samp_coef=1
    ):
        super(AttentionFieldGenerator_WO_GAT, self).__init__()

        print('############## face wo gat ##############')
        print('use traj: ', use_traj)
        print('feat_concat_samp_coef: ', feat_concat_samp_coef)
        print('###################################')

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.use_traj = use_traj
        self.feat_concat_samp_coef = feat_concat_samp_coef
        self.face_lstm_hidden_size = face_lstm_hidden_size
        self.face_lstm_input_size = face_lstm_input_size

        if use_traj:
            self.face_lstm_encoder = nn.LSTM(face_lstm_input_size,
                                             int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.traj_lstm_encoder = nn.LSTM(face_lstm_input_size,
                                             int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers)
            self.pred_lstm_hidden_size = int(2 * self.face_lstm_hidden_size / feat_concat_samp_coef)
            self.face_hidden2ori = nn.Linear(int(2 * face_lstm_hidden_size / feat_concat_samp_coef), 2)
        else:
            self.face_lstm_encoder = nn.LSTM(face_lstm_input_size, face_lstm_hidden_size, lstm_layers)
            self.face_hidden2ori = nn.Linear(face_lstm_hidden_size, 2)
            self.pred_lstm_hidden_size = self.face_lstm_hidden_size

        self.pred_lstm_model = nn.LSTMCell(face_lstm_input_size, self.pred_lstm_hidden_size)
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)

        print('============================================')
        print('use traj: ', self.use_traj)
        print('feat concat samp coef', feat_concat_samp_coef)
        print('face / traj lstm encoder: {}-{}-{}'.format(face_lstm_input_size,
              int(face_lstm_hidden_size / feat_concat_samp_coef), lstm_layers))
        print('pred lstm encoder: {}-{}'.format(face_lstm_input_size, self.pred_lstm_hidden_size))
        print('============================================')

    def face_lstm_init_hidden(self, batch):
        if self.use_traj:
            h = torch.zeros(1, batch, int(self.face_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
            c = torch.zeros(1, batch, int(self.face_lstm_hidden_size / self.feat_concat_samp_coef)).cuda()
        else:
            h = torch.zeros(1, batch, self.face_lstm_hidden_size).cuda()
            c = torch.zeros(1, batch, self.face_lstm_hidden_size).cuda()

        return h, c

    def forward(
            self,
            obs_traj_rel,
            obs_face_rel,
            training_step,
            teacher_forcing_ratio=0.5
    ):
        batch = obs_face_rel.shape[1]
        face_state = self.face_lstm_init_hidden(batch)
        pred_ori = []

        if self.use_traj:
            traj_state = self.face_lstm_init_hidden(batch)
            traj_hidden_states, _ = self.traj_lstm_encoder(obs_traj_rel[: self.obs_len], traj_state)
            face_hidden_states, _ = self.face_lstm_encoder(obs_face_rel[: self.obs_len], face_state)
            step_1_hidden_states = torch.cat((traj_hidden_states, face_hidden_states), dim=-1)
            if training_step == 1:
                pred_ori = self.face_hidden2ori(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        else:
            step_1_hidden_states, _ = self.face_lstm_encoder(obs_face_rel[: self.obs_len], face_state)
            if training_step == 1:
                pred_ori = self.face_hidden2ori(step_1_hidden_states)  # torch.Size([9, 1024, 2])

        # return linear transformation of lstm output as prediction result
        if training_step == 1:
            return pred_ori
        elif training_step == 2:
            pred_lstm_hidden = step_1_hidden_states[-1]
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()  # ct is initialized as 0
            ori_output = obs_face_rel[self.obs_len - 1]
            if self.training:
                for i, input_t in enumerate(
                        obs_face_rel[-self.pred_len:].chunk(obs_face_rel[-self.pred_len:].size(0), dim=0)
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else ori_output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    ori_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_ori += [ori_output]
            else:
                # when validation, inference in auto-regressive manner
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        ori_output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    ori_output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_ori += [ori_output]

            ori_outputs = torch.stack(pred_ori)
            return ori_outputs
