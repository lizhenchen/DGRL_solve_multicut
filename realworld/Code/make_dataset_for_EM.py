import numpy as np
import networkx as nx
import pickle
import joblib


def em_data(path, is_train, sub_min_num_node):
    G = nx.Graph()
    edge_list = []
    if is_train is True:
        uv_ids = joblib.load(open('./' + path + '/train/uv_ids.pkl', 'rb'))
        edge_costs = joblib.load(open('./' + path + '/train/edge_costs.pkl', 'rb'))
    else:
        uv_ids = joblib.load(open('./' + path + '/test/uv_ids.pkl', 'rb'))
        edge_costs = joblib.load(open('./' + path + '/test/edge_costs.pkl', 'rb'))
    for i in range(uv_ids.shape[0]):
        edge_list.append((uv_ids[i][0], uv_ids[i][1], edge_costs[i]))
    G.add_weighted_edges_from(edge_list)
    print(np.max(uv_ids), np.min(uv_ids))
    print(G.number_of_nodes(), G.number_of_edges())
    G = nx.relabel_nodes(G, mapping=dict(zip(G, range(G.number_of_nodes()))))
    print(G.number_of_nodes(), G.number_of_edges())
    SG_list = []
    if is_train is True:
        num = 550
    else:
        num = 50
    sample_index = np.random.choice(G.number_of_nodes(), num, replace=False).tolist()
    for j in range(num):
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
    if is_train is True:
        pickle.dump(SG_list[:500], open('./%s/train_nx_graphs.pkl' % path, 'wb'), protocol=4)
        pickle.dump(SG_list[500:550], open('./%s/valid_nx_graphs.pkl' % path, 'wb'), protocol=4)
    else:
        pickle.dump(SG_list[0:50], open('./%s/test_nx_graphs.pkl' % path, 'wb'), protocol=4)


if __name__ == '__main__':

    # em_data(path='SNEMI3D', is_train=False, sub_min_num_node=50)
    # em_data(path='CREMI-C', is_train=False, sub_min_num_node=50)
    em_data(path='FIB25', is_train=False, sub_min_num_node=50)

    # g_original_nx = pickle.load(open('./%s/test_nx_graphs.pkl' % 'SNEMI3D', 'rb'))  # 223.56
    # g_original_nx = pickle.load(open('./%s/test_nx_graphs.pkl' % 'CREMI-C', 'rb'))  # 224.92
    g_original_nx = pickle.load(open('./%s/test_nx_graphs.pkl' % 'FIB25', 'rb'))  # 216.94

    l = []
    for i in g_original_nx:
        print(i.number_of_nodes(), i.number_of_edges())
        l.append(i.number_of_edges())
    print(sum(l)/len(l))
    print(max(l))

    # input_path = 'SNEMI3D'
    # uv_ids = joblib.load(open('./' + input_path + '/test/uv_ids.pkl', 'rb'))
    # edge_costs = joblib.load(open('./' + input_path + '/test/edge_costs.pkl', 'rb'))
    # print(edge_costs.shape, type(edge_costs))
