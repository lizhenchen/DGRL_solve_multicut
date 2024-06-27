import torch
import pickle
import joblib
import csv
import numpy as np
from message import Generate_dataset, load_graphs
from rl_common import GraphEdgeEnv
from graph_embedding import S2VGraph
from dqn import Agent
from cmd_args import cmd_args

# PATH = 'realworld/modularity-clustering/polbooks'
# PATH = 'realworld/knott-3d-450/knott-3d-450'
# PATH = 'realworld/image-seg/image-seg'
# PATH = 'realworld/SNEMI3D'
# PATH = 'realworld/CREMI-C'
PATH = 'realworld/FIB25'
# PATH = 'BA-60-20000-a'
# PATH = 'ER-20-0175'
# PATH = 'ER-60-0125-20000'
# PATH = 'BA40-BL22-SH'
# PATH = 'BA40-CG4-SH'
# PATH = 'BA40-SG4-SH'
# ORIGINAL = 'ER-20-0175'
ORIGINAL = PATH
NUM_VALID_GRAPHS = 50
NUM_VALID_NODES = 60
IS_TO_GENERATE = False
IS_GPU = False
CUDA = 'cuda:0'
print(PATH, ORIGINAL)


if __name__ == '__main__':

    # '''
    argsm = np.load('../%s/args.npz.npy' % PATH, allow_pickle=True)
    for i in range(argsm.shape[0]):
        para_value = None
        if str(argsm[i][0]) in {'ctx', 'cuda', 'folder_name', 'graph_form', 'edge_weight', 'graph_embed_type', 'big_small', 'graph_coarse'}:  # 8
            para_value = argsm[i][1]
        elif str(argsm[i][0]) in {'is_to_generate', 'is_weighted_gnn', 'isPretrain', 'isDone', 'isDueling', 'isDouble', 'is_shared_GNN_layers'} and str(argsm[i][1]) == 'True':  # 7
            para_value = True
        elif str(argsm[i][0]) in {'is_to_generate', 'is_weighted_gnn', 'isPretrain', 'isDone', 'isDueling', 'isDouble', 'is_shared_GNN_layers'} and str(argsm[i][1]) == 'False':  # 7
            para_value = False
        elif str(argsm[i][0]) in {'ER_edge_rate', 'target_replace_soft', 'memory_learn_start_rate', 'epsilon',
                                  'min_epsilon', 'epsilon_step_rate', 'gamma', 'learning_rate', 'lr_decay_rate',
                                  'min_learning_rate'}:  # 10
            para_value = float(argsm[i][1])
        elif str(argsm[i][0]) in {'num_graphs', 'num_nodes', 'num_sub_dataset', 'latent_dim', 'state_latent_dim',
                                  'action_latent_dim', 'cg_layers', 'sg_layers', 'big_cycles', 'memory_capacity',
                                  'batch_size', 'learn_freq', 'update_freq', 'N'}:  # 14
            para_value = int(argsm[i][1])
        setattr(cmd_args, str(argsm[i][0]), para_value)

    print(cmd_args)

    if IS_TO_GENERATE:
        output = Generate_dataset(NUM_VALID_GRAPHS, NUM_VALID_NODES)
        g_list, g_original_nx = load_graphs(output)
        pickle.dump(g_original_nx, open('../%s/valid_nx_graphs.pkl' % ORIGINAL, 'wb'), protocol=4)
    else:
        g_original_nx = pickle.load(open('../%s/valid_nx_graphs.pkl' % ORIGINAL, 'rb'))
        g_list = [S2VGraph(g_original_nx[j]) for j in range(len(g_original_nx))]

    env = GraphEdgeEnv()
    print("len g_list:", len(g_list))
    agent = Agent(g_list, g_original_nx, env)
    print("\nStarting Validation Loop\n")

    csvfile = open('../%s/results_valid.csv' % PATH, 'w', newline='')
    writer = csv.writer(csvfile)

    obj_sum = []

    # for i in range(16000,20000):
    for i in range(cmd_args.big_cycles):
    # for i in range(4):

        if i % 20 == 0:
            print(i)
        
        if IS_GPU:
            agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH, i + 1), map_location=lambda storage, loc: storage.cuda(CUDA)))
        else:
            agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH, i + 1), map_location=torch.device('cpu')))
        objectives = agent.test()

        writer.writerow(['#' + str(i + 1)] + objectives + [sum(objectives)])
        obj_sum.append([i + 1, sum(objectives)])

    csvfile.close()
    joblib.dump(obj_sum, '../%s/obj_sum.pkl' % PATH)

    sorted_obj_sum = sorted(obj_sum, key=lambda x: x[1], reverse=False)
    joblib.dump(sorted_obj_sum, '../%s/sorted_valid.pkl' % PATH)
    print(sorted_obj_sum[:10])
    # '''

    '''
    obj_sum_1 = joblib.load(open('../%s/obj_sum_1.pkl' % PATH, 'rb'))
    obj_sum_2 = joblib.load(open('../%s/obj_sum_2.pkl' % PATH, 'rb'))
    obj_sum_3 = joblib.load(open('../%s/obj_sum_3.pkl' % PATH, 'rb'))
    obj_sum_4 = joblib.load(open('../%s/obj_sum_4.pkl' % PATH, 'rb'))
    obj_sum_5 = joblib.load(open('../%s/obj_sum_5.pkl' % PATH, 'rb'))
    obj_sum = obj_sum_1 + obj_sum_2 + obj_sum_3 + obj_sum_4 + obj_sum_5
    sorted_obj_sum = sorted(obj_sum, key=lambda x: x[1], reverse=False)
    joblib.dump(sorted_obj_sum, '../%s/sorted_valid.pkl' % PATH)
    print(sorted_obj_sum[:10])
    '''
