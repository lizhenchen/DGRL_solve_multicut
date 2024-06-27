import numpy as np
import networkx as nx
import pickle
import h5py
import os


def allkeys(obj):
    'Recursively find all keys in an h5py.Group.'
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys


def build_graph(num_states, factors, function_values):

    assert num_states.shape[0] == np.max(num_states) == np.min(num_states)
    assert int(factors.shape[0] / 5) - factors.shape[0] / 5 == 0.0
    assert function_values.shape[0] == int(factors.shape[0] / 5) * 2
    num_of_nodes = num_states.shape[0]
    num_of_edges = int(factors.shape[0] / 5)
    G = nx.Graph()
    edge_list = []
    # nnnnn = []
    for i in range(num_of_edges):
        assert factors[5 * i] == i
        edge_list.append((factors[5 * i + 3], factors[5 * i + 4], function_values[2 * i + 1] - function_values[2 * i]))
        # nnnnn.append([factors[5 * i + 3], factors[5 * i + 4]])
    # nnnnn = np.array(nnnnn)
    # mmmmm = np.unique(nnnnn, axis=0, return_counts=True, return_index=True)
    G.add_weighted_edges_from(edge_list)
    # assert G.number_of_edges() == num_of_edges, print(G.number_of_edges(), num_of_edges, len(edge_list), nnnnn.shape, mmmmm)
    assert G.number_of_nodes() == num_of_nodes
    assert min(G.nodes) == 0
    assert max(G.nodes) == num_of_nodes - 1
    print(G.number_of_nodes())

    return G


def modularity_clustering(sub_num_nodes):

    for name in ['football', 'karate', 'lesmis']:
        h = h5py.File('../%s/%s.h5' % ('modularity-clustering', name), 'r')
        num_states = h['gm/numbers-of-states'][:]
        factors = h['gm/factors'][:]
        function_values = h['gm/function-id-16006/values'][:]
        G = build_graph(num_states, factors, function_values)
        SG_list = []
        for j in range(600):
            sample_index = np.random.choice(G.number_of_nodes(), sub_num_nodes, replace=False).tolist()
            SG = nx.subgraph(G, sample_index)
            SG_new = nx.relabel_nodes(SG, mapping=dict(zip(SG, range(SG.number_of_nodes()))))
            if j == 0:
                print(SG_new.nodes)
                print(SG_new.edges)
                print(SG_new.number_of_nodes())
                print(SG_new.number_of_edges())
            SG_list.append(SG_new)
        if os.path.exists('../modularity-clustering/%s' % name) is False:
            os.mkdir('../modularity-clustering/%s' % name)
        pickle.dump(SG_list[:500], open('../modularity-clustering/%s/train_nx_graphs.pkl' % name, 'wb'), protocol=4)
        pickle.dump(SG_list[500:550], open('../modularity-clustering/%s/valid_nx_graphs.pkl' % name, 'wb'), protocol=4)
        pickle.dump(SG_list[550:600], open('../modularity-clustering/%s/test_nx_graphs.pkl' % name, 'wb'), protocol=4)


def knott_3d(sub_min_num_node):

    # for name in ['knott-3d-150', 'knott-3d-300', 'knott-3d-450']:
    for name in ['knott-3d-150']:
        file_name_list = os.listdir('../%s' % name)
        G_list = []
        for file_name in file_name_list:
            if file_name == 'knott-3d-150':
                continue
            h = h5py.File('../%s/%s' % (name, file_name), 'r')
            num_states = h['gm/numbers-of-states'][:]
            factors = h['gm/factors'][:]
            function_values = h['gm/function-id-16006/values'][:]
            G = build_graph(num_states, factors, function_values)
            G_list.append(G)
        G = nx.disjoint_union_all(G_list)
        G = nx.relabel_nodes(G, mapping=dict(zip(G, range(G.number_of_nodes()))))
        print(G.number_of_nodes())
        print(G.number_of_edges())
        SG_list = []
        sample_index = np.random.choice(G.number_of_nodes(), 600, replace=False).tolist()
        for j in range(600):
            index = sample_index[j]
            node_dict_1 = nx.single_source_shortest_path_length(G, index, cutoff=1)
            node_dict_2 = nx.single_source_shortest_path_length(G, index, cutoff=2)
            node_dict_3 = nx.single_source_shortest_path_length(G, index, cutoff=3)
            sub_node_list = []
            if len(node_dict_1) >= sub_min_num_node:
                sub_node_list.append(index)
                node_dict_1_list = [i for i in node_dict_1 if node_dict_1[i] == 1]
                sub_node_list.extend(node_dict_1_list[:(sub_min_num_node - 1)])
            elif len(node_dict_1) < sub_min_num_node and len(node_dict_2) >= sub_min_num_node:
                node_dict_1_list = [i for i in node_dict_1]
                sub_node_list.extend(node_dict_1_list)
                node_dict_2_list = [i for i in node_dict_2 if node_dict_2[i] == 2]
                sub_node_list.extend(node_dict_2_list[:(sub_min_num_node - len(node_dict_1_list))])
            elif len(node_dict_2) < sub_min_num_node and len(node_dict_3) >= sub_min_num_node:
                node_dict_2_list = [i for i in node_dict_2]
                sub_node_list.extend(node_dict_2_list)
                node_dict_3_list = [i for i in node_dict_3 if node_dict_3[i] == 3]
                sub_node_list.extend(node_dict_3_list[:(sub_min_num_node - len(node_dict_2_list))])
            else:
                print('WARNING.')
            SG = nx.subgraph(G, sub_node_list)
            assert SG.number_of_nodes() == sub_min_num_node
            SG_new = nx.relabel_nodes(SG, mapping=dict(zip(SG, range(SG.number_of_nodes()))))
            print(SG_new.number_of_nodes(), SG_new.number_of_edges())
            SG_list.append(SG_new)
        if os.path.exists('../%s/%s' % (name, name)) is False:
            os.mkdir('../%s/%s' % (name, name))
        pickle.dump(SG_list[:500], open('../%s/%s/train_nx_graphs.pkl' % (name, name), 'wb'), protocol=4)
        pickle.dump(SG_list[500:550], open('../%s/%s/valid_nx_graphs.pkl' % (name, name), 'wb'), protocol=4)
        pickle.dump(SG_list[550:600], open('../%s/%s/test_nx_graphs.pkl' % (name, name), 'wb'), protocol=4)


def image_seg(sub_min_num_node):

    file_name_list = os.listdir('../%s' % 'image-seg')
    G_list = []
    for file_name in file_name_list:
        if file_name == 'image-seg':
            continue
        print(file_name)
        h = h5py.File('../%s/%s' % ('image-seg', file_name), 'r')
        num_states = h['gm/numbers-of-states'][:]
        factors = h['gm/factors'][:]
        function_values = h['gm/function-id-16006/values'][:]
        G = build_graph(num_states, factors, function_values)
        G_list.append(G)
    G = nx.disjoint_union_all(G_list)
    G = nx.relabel_nodes(G, mapping=dict(zip(G, range(G.number_of_nodes()))))
    print(G.number_of_nodes())
    print(G.number_of_edges())
    SG_list = []
    sample_index = np.random.choice(G.number_of_nodes(), 600, replace=False).tolist()
    for j in range(600):
        index = sample_index[j]
        node_dict_1 = nx.single_source_shortest_path_length(G, index, cutoff=1)
        node_dict_2 = nx.single_source_shortest_path_length(G, index, cutoff=2)
        node_dict_3 = nx.single_source_shortest_path_length(G, index, cutoff=3)
        node_dict_4 = nx.single_source_shortest_path_length(G, index, cutoff=4)
        node_dict_5 = nx.single_source_shortest_path_length(G, index, cutoff=5)
        sub_node_list = []
        if len(node_dict_1) >= sub_min_num_node:
            sub_node_list.append(index)
            node_dict_1_list = [i for i in node_dict_1 if node_dict_1[i] == 1]
            sub_node_list.extend(node_dict_1_list[:(sub_min_num_node - 1)])
        elif len(node_dict_1) < sub_min_num_node and len(node_dict_2) >= sub_min_num_node:
            node_dict_1_list = [i for i in node_dict_1]
            sub_node_list.extend(node_dict_1_list)
            node_dict_2_list = [i for i in node_dict_2 if node_dict_2[i] == 2]
            sub_node_list.extend(node_dict_2_list[:(sub_min_num_node - len(node_dict_1_list))])
        elif len(node_dict_2) < sub_min_num_node and len(node_dict_3) >= sub_min_num_node:
            node_dict_2_list = [i for i in node_dict_2]
            sub_node_list.extend(node_dict_2_list)
            node_dict_3_list = [i for i in node_dict_3 if node_dict_3[i] == 3]
            sub_node_list.extend(node_dict_3_list[:(sub_min_num_node - len(node_dict_2_list))])
        elif len(node_dict_3) < sub_min_num_node and len(node_dict_4) >= sub_min_num_node:
            node_dict_3_list = [i for i in node_dict_3]
            sub_node_list.extend(node_dict_3_list)
            node_dict_4_list = [i for i in node_dict_4 if node_dict_4[i] == 4]
            sub_node_list.extend(node_dict_4_list[:(sub_min_num_node - len(node_dict_3_list))])
        elif len(node_dict_4) < sub_min_num_node and len(node_dict_5) >= sub_min_num_node:
            node_dict_4_list = [i for i in node_dict_4]
            sub_node_list.extend(node_dict_4_list)
            node_dict_5_list = [i for i in node_dict_5 if node_dict_5[i] == 5]
            sub_node_list.extend(node_dict_5_list[:(sub_min_num_node - len(node_dict_4_list))])
        else:
            print('WARNING.')
        SG = nx.subgraph(G, sub_node_list)
        assert SG.number_of_nodes() == sub_min_num_node
        SG_new = nx.relabel_nodes(SG, mapping=dict(zip(SG, range(SG.number_of_nodes()))))
        print(SG_new.number_of_nodes(), SG_new.number_of_edges())
        SG_list.append(SG_new)
    if os.path.exists('../%s/%s' % ('image-seg', 'image-seg')) is False:
        os.mkdir('../%s/%s' % ('image-seg', 'image-seg'))
    pickle.dump(SG_list[:500], open('../%s/%s/train_nx_graphs.pkl' % ('image-seg', 'image-seg'), 'wb'), protocol=4)
    pickle.dump(SG_list[500:550], open('../%s/%s/valid_nx_graphs.pkl' % ('image-seg', 'image-seg'), 'wb'), protocol=4)
    pickle.dump(SG_list[550:600], open('../%s/%s/test_nx_graphs.pkl' % ('image-seg', 'image-seg'), 'wb'), protocol=4)


if __name__ == '__main__':

    modularity_clustering(sub_num_nodes=25)
    # knott_3d(sub_min_num_node=50)
    # image_seg(sub_min_num_node=50)

    '''
    # G = nx.Graph()
    # edge_list = [(1,4,6),(3,1,2)]
    # G.add_weighted_edges_from(edge_list)
    # print(max(G.nodes), 1)

    # h = h5py.File('../%s/69020.bmp.h5' % 'image-seg', 'r')
    # h = h5py.File('../%s/241004.bmp.h5' % 'image-seg', 'r')
    h = h5py.File('../%s/football.h5' % 'modularity-clustering', 'r')
    # h = h5py.File('../%s/gm_knott_3d_032.h5' % 'knott-3d-150', 'r')
    # h = h5py.File('../%s/gm_knott_3d_079.h5' % 'knott-3d-300', 'r')
    # h = h5py.File('../%s/gm_knott_3d_103.h5' % 'knott-3d-450', 'r')
    print(h.keys())
    print(allkeys(h))
    gm_factors = h['gm/factors']
    print(gm_factors, gm_factors[:])
    gm_header = h['gm/header']
    print(gm_header, gm_header[:])
    gm_number = h['gm/numbers-of-states']
    print(gm_number, gm_number[:])
    gm_function_indices = h['gm/function-id-16006/indices']
    print(gm_function_indices, gm_function_indices[:])
    gm_function_values = h['gm/function-id-16006/values']
    print(gm_function_values, gm_function_values[:].shape)
    a = h['gm/factors'][:]
    print(type(a), np.min(a), np.max(a))
    b = h['gm/function-id-16006/values'][:]
    print(type(b), np.min(b), np.max(b))
    c = h['gm/function-id-16006/indices'][:]
    print(type(c), np.min(c), np.max(c))
    G, q, p, f, nnnnn = build_graph(h['gm/numbers-of-states'][:], gm_factors[:], h['gm/function-id-16006/values'][:])

    # gm_number = h['gm/numbers-of-states']
    # print(gm_number, gm_number[:])
    # gm_function_indices = h['gm/function-id-16007/indices']
    # print(gm_function_indices, gm_function_indices[:])
    # gm_function_values = h['gm/function-id-16007/values']
    # print(gm_function_values, gm_function_values[:])
    # a = h['gm/factors'][:]
    # print(type(a), np.min(a), np.max(a))
    # b = h['gm/function-id-16007/values'][:]
    # print(type(b), np.min(b), np.max(b))
    # c = h['gm/function-id-16007/indices'][:]
    # print(type(c), np.min(c), np.max(c))
    '''

    # g_original_nx = pickle.load(open('../%s/test_nx_graphs.pkl' % 'knott-3d-450/knott-3d-450', 'rb'))
    # g_original_nx = pickle.load(open('../%s/test_nx_graphs.pkl' % 'image-seg/image-seg', 'rb'))
    # l = []
    # for i in g_original_nx:
    #     # print(i.number_of_nodes(), i.number_of_edges())
    #     l.append(i.number_of_edges())
    # print(sum(l)/len(l))
    # print(max(l))
