import torch
import pickle
import joblib
import csv
import time
import numpy as np
from graph_embedding import S2VGraph
from rl_common import GraphEdgeEnv
from dqn import Agent
from cmd_args import cmd_args


IS_GPU = False
CUDA = 'cuda:0'


if __name__ == '__main__':

    # inputs = [['BA-20', 'BA-40', 'BA-60'], [3, 4, 3], 'larger/BA-80']
    # inputs = [['BA-20', 'BA-40', 'BA-60'], [3, 4, 3], 'larger/BA-100']
    # inputs = [['BA-20', 'BA-40', 'BA-60'], [3, 4, 3], 'larger/BA-120']
    # inputs = [['ER-20', 'ER-40', 'ER-60'], [3, 4, 3], 'larger/ER-80']
    # inputs = [['ER-20', 'ER-40', 'ER-60'], [3, 4, 3], 'larger/ER-100']
    # inputs = [['ER-20', 'ER-40', 'ER-60'], [3, 4, 3], 'larger/ER-120']
    # inputs = [['WS-20', 'WS-40', 'WS-60'], [3, 4, 3], 'larger/WS-80']
    # inputs = [['WS-20', 'WS-40', 'WS-60'], [3, 4, 3], 'larger/WS-100']
    inputs = [['WS-20', 'WS-40', 'WS-60'], [3, 4, 3], 'larger/WS-120']

    PATH = inputs[0]
    NUM = inputs[1]
    ORIGINAL = inputs[2]
    print(PATH, NUM, ORIGINAL)

    csvfile = open('../%s/our_results.csv' % ORIGINAL, 'w', newline='')
    writer = csv.writer(csvfile)

    g_original_nx = pickle.load(open('../%s/test_nx_graphs.pkl' % ORIGINAL, 'rb'))
    g_list = [S2VGraph(g_original_nx[j]) for j in range(len(g_original_nx))]

    for k in range(len(PATH)):

        argsm = np.load('../%s/args.npz.npy' % PATH[k], allow_pickle=True)
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
        # print(cmd_args)

        env = GraphEdgeEnv()
        agent = Agent(g_list, g_original_nx, env)
        print("\nStarting Testing Loop\n")

        sorted_obj_sum = joblib.load(open('../%s/sorted_valid.pkl' % PATH[k], 'rb'))

        for i in range(int(NUM[k])):
            print(i)

            j = sorted_obj_sum[i][0]

            if IS_GPU:
                agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH[k], j), map_location=lambda storage, loc: storage.cuda(CUDA)))
            else:
                agent.net.load_state_dict(torch.load('../%s/savemodels/checkpoint-%05d.pth' % (PATH[k], j), map_location=torch.device('cpu')))
            start = time.time()
            objectives = agent.test()
            end = time.time()

            writer.writerow(['#' + str(j)] + objectives + [''] + [''] + [str(end - start)] + [sum(objectives)])

    csvfile.close()
