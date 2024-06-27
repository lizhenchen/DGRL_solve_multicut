import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_structure2vec.s2v_lib.pytorch_util import weights_init
from graph_embedding import EmbedMeanField, S2VCG, EmbedGNN
from cmd_args import cmd_args


class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        if cmd_args.is_shared_GNN_layers:
            model = EmbedMeanField
        else:
            model = EmbedGNN
        if cmd_args.graph_embed_type == 'BiLevel':
            self.gnn_cg = model(latent_dim=cmd_args.latent_dim, num_edge_feats=1, max_lv=cmd_args.cg_layers)
            self.gnn_sg = model(latent_dim=cmd_args.latent_dim, num_edge_feats=2, max_lv=cmd_args.sg_layers)
        elif cmd_args.graph_embed_type == 'CG':
            self.gnn_cg = model(latent_dim=cmd_args.latent_dim * 2, num_edge_feats=1, max_lv=cmd_args.cg_layers)
        elif cmd_args.graph_embed_type == 'SG':
            self.gnn_sg = model(latent_dim=cmd_args.latent_dim * 2, num_edge_feats=2, max_lv=cmd_args.sg_layers)

        self.concat_linear = nn.Linear(cmd_args.latent_dim * 2 + 1, cmd_args.latent_dim)

        self.weight_linear = nn.Linear(1, cmd_args.latent_dim)
        self.nodes_linear = nn.Linear(cmd_args.latent_dim, cmd_args.latent_dim)  # # #

        self.state_embed_1 = nn.Linear(cmd_args.latent_dim + 1, cmd_args.state_latent_dim)
        self.state_embed_out = nn.Linear(cmd_args.state_latent_dim, 1)

        self.action_embed_1 = nn.Linear(cmd_args.latent_dim * 3 + 1, cmd_args.action_latent_dim)
        self.action_embed_out = nn.Linear(cmd_args.action_latent_dim, 1)

        weights_init(self)

    def get_matrix_big_small(self, state_graph, contracted_graph, num_state_graph_nodes, f='sum'):

        NUM_SMALL_NODES = 0
        index0, index1, vals = [], [], []
        coeff = None
        num_contains = [0.0] * sum(num_state_graph_nodes)

        for k in range(len(state_graph)):

            for i in contracted_graph[k].nodes:

                i_contain = contracted_graph[k].nodes[i]['contain']
                len_i_contain = len(i_contain)
                if f == 'sum':
                    coeff = 1.0
                elif f == 'mean':
                    coeff = 1.0 / len_i_contain

                index0.extend([i + NUM_SMALL_NODES] * len_i_contain)
                index1.extend([j + NUM_SMALL_NODES for j in list(i_contain)])
                vals.extend([coeff] * len_i_contain)
                num_contains[i + NUM_SMALL_NODES] = len_i_contain

            NUM_SMALL_NODES += state_graph[k].num_nodes

        MBS = torch.sparse.FloatTensor(torch.LongTensor([index0, index1]), torch.FloatTensor(vals), torch.Size([NUM_SMALL_NODES, NUM_SMALL_NODES]))
        if cmd_args.ctx == 'gpu':
            MBS = MBS.cuda(cmd_args.cuda)

        num_contains = torch.FloatTensor(num_contains).unsqueeze(1)
        if cmd_args.ctx == 'gpu':
            num_contains = num_contains.cuda(cmd_args.cuda)

        return MBS, num_contains

    def get_matrix_graph_pooling(self, num_state_graph_nodes, num_contracted_graph_edges, num_contracted_graph_nodes, cg_contracted_nodes_list, f='sum'):

        l1 = sum(num_contracted_graph_edges)
        l2 = sum(num_state_graph_nodes)
        index0, index1, vals = [], [], []
        start1, start2 = 0, 0
        num_nodes = []

        if f == 'sum':
            for i in range(len(num_state_graph_nodes)):
                for k in range(start2, start2 + num_state_graph_nodes[i]):
                    if (k - start2) not in cg_contracted_nodes_list[i]:
                        for j in range(start1, start1 + num_contracted_graph_edges[i]):
                            index0.append(j)
                            index1.append(k)
                vals.extend([1.0] * num_contracted_graph_nodes[i] * num_contracted_graph_edges[i])
                start1 += num_contracted_graph_edges[i]
                start2 += num_state_graph_nodes[i]
                num_nodes.extend([num_contracted_graph_nodes[i]] * num_contracted_graph_edges[i])
        elif f == 'mean':
            for i in range(len(num_state_graph_nodes)):
                for k in range(start2, start2 + num_state_graph_nodes[i]):
                    if (k - start2) not in cg_contracted_nodes_list[i]:
                        for j in range(start1, start1 + num_contracted_graph_edges[i]):
                            index0.append(j)
                            index1.append(k)
                t = 1.0 / num_contracted_graph_nodes[i]
                vals.extend([t] * num_contracted_graph_nodes[i] * num_contracted_graph_edges[i])  # pay attention!
                start1 += num_contracted_graph_edges[i]
                start2 += num_state_graph_nodes[i]
                num_nodes.extend([num_contracted_graph_nodes[i]] * num_contracted_graph_edges[i])

        MGP = torch.sparse.FloatTensor(torch.LongTensor([index0, index1]), torch.FloatTensor(vals), torch.Size([l1, l2]))
        if cmd_args.ctx == 'gpu':
            MGP = MGP.cuda(cmd_args.cuda)

        num_nodes = torch.FloatTensor(num_nodes).unsqueeze(1)
        if cmd_args.ctx == 'gpu':
            num_nodes = num_nodes.cuda(cmd_args.cuda)

        return MGP, num_nodes

    def get_tensor_nodes_to_edge(self, state_graph, contracted_graph, f):

        if f == 'sum':
            NUM_SMALL_NODES = 0
            index0, index1 = [], []
            counter = 0
            for k in range(len(state_graph)):
                for (u, v) in contracted_graph[k].edges:
                    index0.extend([counter] * 2)
                    index1.extend([u + NUM_SMALL_NODES, v + NUM_SMALL_NODES])
                    counter += 1
                NUM_SMALL_NODES += state_graph[k].num_nodes

            TNTE = torch.sparse.FloatTensor(torch.LongTensor([index0, index1]), torch.FloatTensor([1.0, 1.0] * counter), torch.Size([counter, NUM_SMALL_NODES])).to_dense()
            if cmd_args.ctx == 'gpu':
                TNTE = TNTE.cuda(cmd_args.cuda)

        elif f == 'max':
            NUM_SMALL_NODES = 0
            index0, index1, index2 = [], [], []
            counter = 0
            for k in range(len(state_graph)):
                for (u, v) in contracted_graph[k].edges:
                    index0.extend([0, 1])
                    index1.extend([counter] * 2)
                    index2.extend([u + NUM_SMALL_NODES, v + NUM_SMALL_NODES])
                    counter += 1
                NUM_SMALL_NODES += state_graph[k].num_nodes

            TNTE = torch.sparse.FloatTensor(torch.LongTensor([index0, index1, index2]), torch.FloatTensor([1.0, 1.0] * counter), torch.Size([2, counter, NUM_SMALL_NODES])).to_dense()
            if cmd_args.ctx == 'gpu':
                TNTE = TNTE.cuda(cmd_args.cuda)

        return TNTE

    def get_matrix_piecewise_mean(self, num_coarse_edges_list):

        l = sum(num_coarse_edges_list)
        index0, index1, vals = [], [], []
        start = 0

        for i in range(len(num_coarse_edges_list)):
            for j in range(start, start + num_coarse_edges_list[i]):
                for k in range(start, start + num_coarse_edges_list[i]):
                    index0.append(j)
                    index1.append(k)
                    vals.append(1.0/(num_coarse_edges_list[i]))
            start += num_coarse_edges_list[i]

        MPM = torch.sparse.FloatTensor(torch.LongTensor([index0, index1]), torch.FloatTensor(vals), torch.Size([l, l]))
        if cmd_args.ctx == 'gpu':
            MPM = MPM.cuda(cmd_args.cuda)

        return MPM

    def PrepareEdgeFeatures(self, state_graph):

        sg_wt, sg_xs = [], []

        for i in range(len(state_graph)):
            for j in range(len(state_graph[i].xs)):
                sg_wt.extend([state_graph[i].weights[j]] * 2)
                sg_xs.extend([state_graph[i].xs[j]] * 2)

        if cmd_args.ctx == 'gpu':
            edge_feat = torch.cuda.FloatTensor([sg_wt, sg_xs], device=torch.device(cmd_args.cuda)).t()
        else:
            edge_feat = torch.FloatTensor([sg_wt, sg_xs]).t()

        return edge_feat

    def CG_features(self, state_graph, contracted_graph):

        cg = copy.deepcopy(contracted_graph)
        weights = []
        weights_single = []
        cg_contract_nodes_list = []

        for k in range(len(state_graph)):

            contracted_nodes_list = []
            for j in range(state_graph[k].num_nodes):
                if j not in cg[k].nodes():
                    contracted_nodes_list.append(j)
            cg[k].add_nodes_from(contracted_nodes_list)
            cg_contract_nodes_list.append(contracted_nodes_list)
            for u, v, w in cg[k].edges(data=True):
                weights.extend([w['weight'], w['weight']])
                weights_single.append(w['weight'])

        if cmd_args.ctx == 'gpu':
            weights_feature = torch.cuda.FloatTensor(weights, device=torch.device(cmd_args.cuda)).unsqueeze(1)
            weights_single_feature = torch.cuda.FloatTensor(weights_single, device=torch.device(cmd_args.cuda)).unsqueeze(1)
        else:
            weights_feature = torch.FloatTensor(weights).unsqueeze(1)
            weights_single_feature = torch.FloatTensor(weights_single).unsqueeze(1)

        return weights_feature, cg, weights_single_feature, cg_contract_nodes_list

    def forward(self, state_graph, contracted_graph):

        edge_feat = self.PrepareEdgeFeatures(state_graph)

        num_state_graph_nodes = [state_graph[k].num_nodes for k in range(len(state_graph))]
        num_state_graph_edges = [state_graph[k].num_edges for k in range(len(state_graph))]
        num_contracted_graph_nodes = [len(contracted_graph[k]) for k in range(len(contracted_graph))]
        num_contracted_graph_edges = [len(contracted_graph[k].edges) for k in range(len(contracted_graph))]

        if cmd_args.ctx == 'gpu':
            q_value = torch.zeros((max(num_state_graph_edges), len(state_graph)), device=torch.device(cmd_args.cuda))
        else:
            q_value = torch.zeros((max(num_state_graph_edges), len(state_graph)))

        MBS, num_contains = self.get_matrix_big_small(state_graph, contracted_graph, num_state_graph_nodes, f='sum')
        weights_feature, cg, weights_single_feature, cg_contract_nodes_list = self.CG_features(state_graph, contracted_graph)

        if cmd_args.graph_embed_type == 'BiLevel':
            tmp_embed_sg = self.gnn_sg(state_graph, edge_feat)
            tmp_embed_sg = torch.mm(MBS, tmp_embed_sg)

            tmp_embed = self.gnn_cg([S2VCG(cg[k]) for k in range(len(cg))], weights_feature)

            tmp_embed = torch.cat((tmp_embed, tmp_embed_sg, num_contains), dim=1)

        elif cmd_args.graph_embed_type == 'CG':
            tmp_embed = self.gnn_cg([S2VCG(cg[k]) for k in range(len(cg))], weights_feature)

            tmp_embed = torch.cat((tmp_embed, num_contains), dim=1)

        elif cmd_args.graph_embed_type == 'SG':
            tmp_embed_sg = self.gnn_sg(state_graph, edge_feat)
            tmp_embed_sg = torch.mm(MBS, tmp_embed_sg)

            tmp_embed = torch.cat((tmp_embed_sg, num_contains), dim=1)

        tmp_embed = self.concat_linear(tmp_embed)

        # state
        MGP, num_nodes = self.get_matrix_graph_pooling(num_state_graph_nodes, num_contracted_graph_edges, num_contracted_graph_nodes, cg_contract_nodes_list, f='sum')
        tmp_graph_embed = torch.mm(MGP, tmp_embed)
        tmp_graph_embed = torch.cat((tmp_graph_embed, num_nodes), dim=1)
        tmp_graph_embed = F.relu(tmp_graph_embed)

        # action
        '''
        TNTE = self.get_tensor_nodes_to_edge(state_graph, contracted_graph, f='max')
        tmp_embed = torch.matmul(TNTE, tmp_embed)
        nodes_part = torch.max(tmp_embed, 0)[0]
        # nodes_part = self.nodes_linear(nodes_part)
        '''
        TNTE = self.get_tensor_nodes_to_edge(state_graph, contracted_graph, f='sum')
        nodes_part = torch.mm(TNTE, tmp_embed)
        nodes_part = self.nodes_linear(nodes_part)
        # '''
        edge_part = self.weight_linear(weights_single_feature)
        tmp_embed = torch.cat((nodes_part, edge_part), dim=1)
        tmp_embed = F.relu(tmp_embed)

        v_embed = F.relu(self.state_embed_1(tmp_graph_embed))
        v_value = self.state_embed_out(v_embed)

        assert cmd_args.isDueling is True
        action_embed = torch.cat((tmp_embed, tmp_graph_embed), dim=1)  # torch.Size([291, 128])
        action_embed = F.relu(self.action_embed_1(action_embed))  # torch.Size([291, 32])
        adv_value = self.action_embed_out(action_embed)  # torch.Size([291, 1])

        MPM = self.get_matrix_piecewise_mean(num_contracted_graph_edges)
        adv_mean = torch.mm(MPM, adv_value)
        tmp_pred = v_value + adv_value - adv_mean

        pred_list = []
        ncge = 0
        for i in range(len(num_contracted_graph_edges)):
            ncgei = num_contracted_graph_edges[i]
            q_value_i = tmp_pred[ncge: ncge + ncgei]
            q_value[0: ncgei, i] = q_value_i.view(1, ncgei)
            pred_list.append(q_value_i)
            ncge += ncgei

        return pred_list, q_value
