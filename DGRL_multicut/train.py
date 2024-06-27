import pickle
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from message import Generate_dataset, load_graphs
from rl_common import GraphEdgeEnv
from graph_embedding import S2VGraph
from cmd_args import cmd_args
from dqn import Agent


PATH = cmd_args.folder_name
GLOBAL_NUM_GRAPHS = cmd_args.num_graphs
NUM_NODES = cmd_args.num_nodes
# ORIGINAL = 'BA-40'
ORIGINAL = PATH
print(PATH, ORIGINAL)


if __name__ == '__main__':

    # if not os.path.exists(PATH):
        # os.makedirs(PATH)
    argsm = cmd_args._get_kwargs()
    np.savetxt("../%s/args.csv" % PATH, np.array(argsm), delimiter=',', fmt='%s')
    np.save("../%s/args.npz" % PATH, argsm)

    if cmd_args.is_to_generate:
        output = Generate_dataset(GLOBAL_NUM_GRAPHS, NUM_NODES)
        g_list, g_original_nx = load_graphs(output)
        pickle.dump(g_original_nx, open('../%s/train_nx_graphs.pkl' % ORIGINAL, 'wb'), protocol=4)
    else:
        g_original_nx = pickle.load(open('../%s/train_nx_graphs.pkl' % ORIGINAL, 'rb'))
        g_list = [S2VGraph(g_original_nx[j]) for j in range(len(g_original_nx))]

    env = GraphEdgeEnv()
    print("len g_list:", len(g_list))

    agent = Agent(g_list, g_original_nx, env)
    print("\nStarting Training Loop\n")
    agent.train()



    # argsm = cmd_args._get_kwargs()
    # np.savetxt("../%s/args.csv" % PATH, np.array(argsm), delimiter=',', fmt='%s')
    # np.save("../%s/args.npz" % PATH, argsm)
    #
    # output = Generate_dataset(GLOBAL_NUM_GRAPHS, NUM_NODES)
    # g_list, g_original_nx = load_graphs(output)
    # pickle.dump(g_original_nx, open('../%s/test_nx_graphs.pkl' % ORIGINAL, 'wb'), protocol=4)
