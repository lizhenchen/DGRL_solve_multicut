import networkx as nx
import random
import math
import numpy as np
import itertools
from graph_embedding import S2VGraph
from cmd_args import cmd_args


def Random_Graph(num_nodes):

    if cmd_args.graph_form == 'er':
        edges = itertools.combinations(range(num_nodes), 2)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for e in edges:
            if np.random.uniform() < cmd_args.ER_edge_rate:
                G.add_edge(*e)
        ncc = nx.number_connected_components(G)
        if ncc > 1:
            cc_list = []
            cc = nx.connected_components(G)
            for i in cc:
                cc_list.append(i)
            for j in range(ncc-1):
                u = random.choice(tuple(cc_list[j]))
                v = random.choice(tuple(cc_list[j+1]))
                G.add_edge(u, v)
        assert nx.number_connected_components(G) == 1
        if cmd_args.edge_weight == 'log':
            for edge in G.edges():
                p = random.uniform(0.001, 0.999)
                G.edges[edge]['weight'] = math.log((1 - p) / p)
        elif cmd_args.edge_weight == 'uniform':
            for edge in G.edges():
                G.edges[edge]['weight'] = random.uniform(-1.0, 1.0)

    elif cmd_args.graph_form == 'ba':
        G = nx.random_graphs.barabasi_albert_graph(num_nodes, 4, seed=np.random)
        if cmd_args.edge_weight == 'log':
            for edge in G.edges():
                p = random.uniform(0.001, 0.999)
                G.edges[edge]['weight'] = math.log((1 - p) / p)
        elif cmd_args.edge_weight == 'uniform':
            for edge in G.edges():
                G.edges[edge]['weight'] = random.uniform(-1.0, 1.0)

    elif cmd_args.graph_form == 'ws':
        G = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, 6, 0.2, tries=500, seed=None)
        if cmd_args.edge_weight == 'log':
            for edge in G.edges():
                p = random.uniform(0.001, 0.999)
                G.edges[edge]['weight'] = math.log((1 - p) / p)
        elif cmd_args.edge_weight == 'uniform':
            for edge in G.edges():
                G.edges[edge]['weight'] = random.uniform(-1.0, 1.0)
    # print(G.number_of_edges())
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    return G


def Generate_dataset(N, NUM_NODES):

    out = []
    for i in range(N):
        G = Random_Graph(NUM_NODES)
        out.append(G)

    return out


def load_graphs(graph_lists):

    glist_original_nx = graph_lists
    glist = [S2VGraph(graph_lists[j]) for j in range(len(graph_lists))]

    return glist, glist_original_nx
