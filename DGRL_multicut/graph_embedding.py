import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pytorch_structure2vec.s2v_lib.s2v_lib import S2VLIB
from pytorch_structure2vec.s2v_lib.pytorch_util import weights_init
from cmd_args import cmd_args


class S2VGraph(object):

    def __init__(self, g):
        self.g = g
        self.num_nodes = len(g)
        u, v = zip(*g.edges())
        self.num_edges = len(u)
        
        self.edge_pair = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pair[:, 0] = min(u, v)
        self.edge_pair[:, 1] = max(u, v)
        self.edge_pairs = self.edge_pair.flatten()

        self.weights = [None] * self.num_edges
        for i in range(self.num_edges):
            self.weights[i] = g.edges[(self.edge_pair[i, 0], self.edge_pair[i, 1])]['weight']

        self.xs = [None] * self.num_edges
        for i in range(self.num_edges):
            self.xs[i] = 0.0

        self.labels_inverse = {}
        for i in range(self.num_edges):
            self.labels_inverse[(self.edge_pair[i, 0], self.edge_pair[i, 1])] = i  # dict: set --> node_label

    def to_networkx(self):

        return self.g


class S2VCG(object):

    def __init__(self, g):
        self.g = g
        self.num_nodes = len(g)  # useful for s2v_lib.py, do not delete.
        self.num_edges = len(g.edges())

        if self.num_edges > 0:
            x, y = zip(*g.edges())
            self.edge_pair = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pair[:, 0] = min(x, y)
            self.edge_pair[:, 1] = max(x, y)
            self.edge_pairs = self.edge_pair.flatten()
        else:
            self.edge_pairs = np.ndarray(shape=(0, 2), dtype=np.int32)

    def to_networkx(self):

        return self.g


'''
class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):

        ctx.save_for_backward(sp_mat, dense_mat)
        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):

        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(torch.mm(grad_output.data, dense_mat.data.t()))
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2
'''


def spmm(sp_mat, dense_mat):

    # return MySpMM.apply(sp_mat, dense_mat)
    # return torch.sparse.mm(sp_mat, dense_mat)
    return torch.mm(sp_mat, dense_mat)  # most fast


class EmbedMeanField(nn.Module):

    def __init__(self, latent_dim, num_edge_feats, max_lv):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim, bias=False)

        self.conv_params_0 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.conv_params_1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.conv_params_2 = nn.Linear(latent_dim, latent_dim, bias=False)

        weights_init(self)

    def forward(self, graph_list, edge_feat):  # edge_feat: [a,a,b,b,...,z,z]

        if cmd_args.is_weighted_gnn:
            n2n_sp, e2n_sp = S2VLIB.PrepareMeanField(graph_list, edge_feat)
        else:
            n2n_sp, e2n_sp = S2VLIB.PrepareMeanField(graph_list, None)

        if cmd_args.ctx == 'gpu':
            n2n_sp = n2n_sp.cuda(cmd_args.cuda)
            e2n_sp = e2n_sp.cuda(cmd_args.cuda)

        if edge_feat is not None:
            edge_feat = Variable(edge_feat)

        n2n_sp = Variable(n2n_sp, requires_grad=False)
        e2n_sp = Variable(e2n_sp, requires_grad=False)

        h = self.mean_field(edge_feat, n2n_sp, e2n_sp)

        return h

    def mean_field(self, edge_feat, n2n_sp, e2n_sp):

        input_edge_linear = F.relu(self.w_e2l(edge_feat))
        e2n_pool = spmm(e2n_sp, input_edge_linear)
        static_message_conv = self.conv_params_0(e2n_pool)

        cur_message_layer = F.relu(static_message_conv)
        lv = 1
        while lv < self.max_lv:
            n2n_pool = spmm(n2n_sp, cur_message_layer)
            merged_linear = static_message_conv + self.conv_params_1(cur_message_layer) + self.conv_params_2(n2n_pool)
            cur_message_layer = F.relu(merged_linear)
            lv += 1

        return cur_message_layer


class EmbedGNN(nn.Module):

    def __init__(self, latent_dim, num_edge_feats, max_lv):
        super(EmbedGNN, self).__init__()
        self.latent_dim = latent_dim
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim, bias=False)

        self.conv_params_0 = nn.Linear(latent_dim, latent_dim, bias=False)
        if self.max_lv >= 2:
            self.conv_params_1_1 = nn.Linear(latent_dim, latent_dim, bias=False)
            self.conv_params_2_1 = nn.Linear(latent_dim, latent_dim, bias=False)
        if self.max_lv >= 3:
            self.conv_params_1_2 = nn.Linear(latent_dim, latent_dim, bias=False)
            self.conv_params_2_2 = nn.Linear(latent_dim, latent_dim, bias=False)
        if self.max_lv >= 4:
            self.conv_params_1_3 = nn.Linear(latent_dim, latent_dim, bias=False)
            self.conv_params_2_3 = nn.Linear(latent_dim, latent_dim, bias=False)

        weights_init(self)

    def forward(self, graph_list, edge_feat):  # edge_feat: [a,a,b,b,...,z,z]

        if cmd_args.is_weighted_gnn:
            n2n_sp, e2n_sp = S2VLIB.PrepareMeanField(graph_list, edge_feat)
        else:
            n2n_sp, e2n_sp = S2VLIB.PrepareMeanField(graph_list, None)

        if cmd_args.ctx == 'gpu':
            n2n_sp = n2n_sp.cuda(cmd_args.cuda)
            e2n_sp = e2n_sp.cuda(cmd_args.cuda)

        if edge_feat is not None:
            edge_feat = Variable(edge_feat)

        n2n_sp = Variable(n2n_sp, requires_grad=False)
        e2n_sp = Variable(e2n_sp, requires_grad=False)

        h = self.mean_field(edge_feat, n2n_sp, e2n_sp)

        return h

    def mean_field(self, edge_feat, n2n_sp, e2n_sp):

        input_edge_linear = F.relu(self.w_e2l(edge_feat))
        e2n_pool = spmm(e2n_sp, input_edge_linear)
        static_message = self.conv_params_0(e2n_pool)

        cur_message_layer = F.relu(static_message)
        if self.max_lv >= 2:
            n2n_pool = spmm(n2n_sp, cur_message_layer)
            merged_linear = static_message + self.conv_params_1_1(cur_message_layer) + self.conv_params_2_1(n2n_pool)
            cur_message_layer = F.relu(merged_linear)
        if self.max_lv >= 3:
            n2n_pool = spmm(n2n_sp, cur_message_layer)
            merged_linear = static_message + self.conv_params_1_2(cur_message_layer) + self.conv_params_2_2(n2n_pool)
            cur_message_layer = F.relu(merged_linear)
        if self.max_lv >= 4:
            n2n_pool = spmm(n2n_sp, cur_message_layer)
            merged_linear = static_message + self.conv_params_1_3(cur_message_layer) + self.conv_params_2_3(n2n_pool)
            cur_message_layer = F.relu(merged_linear)

        return cur_message_layer
