import torch
import pickle
import joblib
import csv
import time
import numpy as np
from message import Generate_dataset, load_graphs
from graph_embedding import S2VGraph
from rl_common import GraphEdgeEnv
from dqn import Agent
from cmd_args import cmd_args


# PATH = 'OLD-20230511/BA-40-CG2-Weighted'
# PATH = 'BA-60-20000'
# PATH = 'realworld/image-seg/image-seg'
# PATH = 'realworld/knott-3d-150/knott-3d-150'
# PATH = 'realworld/modularity-clustering/adjnoun'
# PATH = 'realworld/SNEMI3D'
PATH = 'realworld/CREMI-C'
# PATH = 'realworld/FIB25'
print(PATH)
ORIGINAL = PATH
NUM_TEST_GRAPHS = 50
NUM_TEST_NODES = None
IS_TO_GENERATE = False
IS_GPU = False
CUDA = 'cuda:0'


if __name__ == '__main__':

    argsm = np.load('../%s/args.npz.npy' % PATH, allow_pickle=True)
    for i in range(argsm.shape[0]):
        para_value = None
        if str(argsm[i][0]) in {'ctx', 'cuda', 'folder_name', 'graph_form', 'edge_weight', 'graph_embed_type', 'big_small', 'graph_coarse'}:  # 8
            para_value = argsm[i][1]
        elif str(argsm[i][0]) in {'is_to_generate', 'is_weighted_gnn', 'isPretrain', 'isDone', 'isDueling', 'isDouble', 'is_shared_GNN_layers'} and str(argsm[i][1]) == 'True':  # 6
            para_value = True
        elif str(argsm[i][0]) in {'is_to_generate', 'is_weighted_gnn', 'isPretrain', 'isDone', 'isDueling', 'isDouble', 'is_shared_GNN_layers'} and str(argsm[i][1]) == 'False':  # 6
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
        output = Generate_dataset(NUM_TEST_GRAPHS, NUM_TEST_NODES)
        g_list, g_original_nx = load_graphs(output)
        pickle.dump(g_original_nx, open('../%s/test_nx_graphs.pkl' % ORIGINAL, 'wb'), protocol=4)
    else:
        g_original_nx = pickle.load(open('../%s/test_nx_graphs.pkl' % ORIGINAL, 'rb'))
        g_list = [S2VGraph(g_original_nx[j]) for j in range(len(g_original_nx))]

    env = GraphEdgeEnv()
    print("len g_list:", len(g_list))
    agent = Agent(g_list, g_original_nx, env)
    print("\nStarting Testing Loop\n")

    sorted_obj_sum = joblib.load(open('../%s/sorted_valid.pkl' % PATH, 'rb'))
    for top_num in range(1, len(sorted_obj_sum)):
        if sorted_obj_sum[top_num][1] != sorted_obj_sum[0][1]:
            break
    print(top_num)

    csvfile = open('../%s/results_test.csv' % PATH, 'w', newline='')
    writer = csv.writer(csvfile)

    for i in range(30):
    # for i in range(max(top_num, 30)):
    # for i in range(2000):

        if i % 10 == 0:
            print(i)

        j = sorted_obj_sum[i][0]
        # j = sorted_obj_sum[top_num - 1 - i][0]
        # j = len(sorted_obj_sum) - i

        if IS_GPU:
            agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH, j), map_location=lambda storage, loc: storage.cuda(CUDA)))
        else:
            agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH, j), map_location=torch.device('cpu')))
        start = time.time()
        objectives = agent.test()
        end = time.time()

        writer.writerow(['#' + str(j)] + objectives + [''] + [''] + [str(end - start)] + [sum(objectives)])

    csvfile.close()
