from copy import deepcopy
from cmd_args import cmd_args
import numpy as np


class GraphEdgeEnv(object):

    def setup(self, g_list, g_original_nx):

        self.g_list = deepcopy(g_list)
        self.cg_list = deepcopy(g_original_nx)
        self.original_list = g_original_nx
        for i in range(len(self.cg_list)):
            for j in range(len(self.cg_list[i])):
                self.cg_list[i].nodes[j]['contain'] = {j}

        self.rewards = [None] * len(self.cg_list)
        self.dones_before = [False] * len(self.cg_list)
        self.dones_after = [False] * len(self.cg_list)
        self.counts_before = 0
        self.counts_after = 0

    def step(self, action_lists, train=True):

        picked_edges = action_lists

        for i in range(len(self.cg_list)):
            self.dones_before[i] = self.dones_after[i]

        self.counts_before = self.counts_after

        for i in range(len(self.cg_list)):

            if self.dones_before[i] is True:
                self.rewards[i] = 0.0
                continue

            u = min(picked_edges[i])  # node_preserved
            v = max(picked_edges[i])  # node_contracted

            O = self.original_list[i]  # original input graph
            G = self.cg_list[i]
            H = G.copy()

            self.rewards[i] = G.edges[(u, v)]['weight']

            new_edges = [(u, w) for x, w, d in G.edges(v, data=True) if w != u]

            v_data_contain = H.nodes[v]['contain']
            for j in H.nodes[u]['contain']:
                for k in v_data_contain:
                    if O.has_edge(j, k):
                        l = self.g_list[i].labels_inverse[(min(j, k), max(j, k))]
                        self.g_list[i].xs[l] = 1.0

            H.remove_node(v)

            H.nodes[u]['contain'] = H.nodes[u]['contain'] | v_data_contain

            new_edges_weights = []
            for e in new_edges:
                if G.has_edge(e[0], e[1]):
                    new_edges_weights.append((e[0], e[1], {'weight': G.edges[(u, e[1])]['weight'] + G.edges[(v, e[1])]['weight']}))
                else:
                    new_edges_weights.append((e[0], e[1], {'weight': G.edges[(v, e[1])]['weight']}))

            H.add_edges_from(new_edges_weights)
            self.cg_list[i] = H

            weights = [w['weight'] for u, v, w in H.edges(data=True) if u != v]

            if cmd_args.isDone:
                if len(weights) == 0 or max(weights) <= 0:
                    self.dones_after[i] = True
                    self.counts_after += 1
            else:
                if len(weights) <= 1 or max(weights) <= 0:
                    self.dones_after[i] = True
                    self.counts_after += 1

        if train is True:
            return self.rewards, self.g_list, self.cg_list, self.dones_before, self.dones_after, self.counts_before, self.counts_after

        if train is False:
            return self.dones_after, self.counts_after

    def getState(self):

        return self.g_list, self.cg_list

    def objective(self):

        obj_lists = []
        for i in range(len(self.cg_list)):
            weight_array = np.array([w['weight'] for (u, v, w) in self.cg_list[i].edges(data=True)])
            obj_lists.append(sum(weight_array))

        return obj_lists
