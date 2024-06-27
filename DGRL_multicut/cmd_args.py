import argparse

cmd_opt = argparse.ArgumentParser(description='Argparsers for deep graph reinforcement learning for multicut problem')

cmd_opt.add_argument('-ctx', type=str, default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-cuda', type=str, default='cuda:2', help='#cuda')
cmd_opt.add_argument('-folder_name', type=str, default='realworld/CREMI-C', help='record file')  # realworld/modularity-clustering/adjnoun knott-3d-450/knott-3d-450 image-seg/image-seg
# cmd_opt.add_argument('-folder_name', type=str, default='BA-40-CG3-UnWeighted', help='record file')
# NEW1 (BA-40) -- batch_size: 64->96, update_freq: 1->5, epsilon: 0.5--0.1->0.3--0.05
cmd_opt.add_argument('-is_to_generate', type=bool, default=False, help='is generate dataset?')
cmd_opt.add_argument('-graph_form', type=str, default='real', help='er/ba/ws/real')
cmd_opt.add_argument('-ER_edge_rate', type=float, default=0.100, help='edge rate')  # 0.175/0.150/0.125
cmd_opt.add_argument('-edge_weight', type=str, default='uniform', help='edge weight distribution, log/uniform')
cmd_opt.add_argument('-num_graphs', type=int, default=500, help='number of train graphs')
cmd_opt.add_argument('-num_nodes', type=int, default=50, help='number of graph nodes')  # 20/40/60
cmd_opt.add_argument('-num_sub_dataset', type=int, default=10, help='graph num of sub-dataset')

# q_net.py
cmd_opt.add_argument('-is_weighted_gnn', type=bool, default=False, help='is weighted GNN embedding?')  # UnWeighted better!
cmd_opt.add_argument('-is_shared_GNN_layers', type=bool, default=True, help='is shared GNN layers as RNN?') # True better
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent features')
cmd_opt.add_argument('-state_latent_dim', type=int, default=16, help='dimension of V-value latent features')
cmd_opt.add_argument('-action_latent_dim', type=int, default=32, help='mlp hidden layer size')
cmd_opt.add_argument('-graph_embed_type', type=str, default='BiLevel', help='graph embedding type: BiLevel/CG/SG')
cmd_opt.add_argument('-cg_layers', type=int, default=2, help='max rounds of message passing-1')
cmd_opt.add_argument('-sg_layers', type=int, default=2, help='max rounds of message passing-2')
cmd_opt.add_argument('-big_small', type=str, default='sum', help='mean/sum')  # Sum
cmd_opt.add_argument('-graph_coarse', type=str, default='sum', help='mean/sum')  # Sum

# dqn.py
cmd_opt.add_argument('-isPretrain', type=bool, default=False, help='need pretrain weights?')
cmd_opt.add_argument('-big_cycles', type=int, default=20000, help='number of big cycles')  # 5W/2W/2W
cmd_opt.add_argument('-target_replace_soft', type=float, default=1.0, help='target replace soft rate')  # 1.0
cmd_opt.add_argument('-memory_capacity', type=int, default=15*10000, help='memory pool capacity')  # 15W/15W/20W
cmd_opt.add_argument('-memory_learn_start_rate', type=float, default=0.25, help='memory pool capacity')
cmd_opt.add_argument('-batch_size', type=int, default=96, help='minibatch size')  # 64/96/128
cmd_opt.add_argument('-learn_freq', type=int, default=10, help='learning frequency')  # 5/10/50
cmd_opt.add_argument('-update_freq', type=int, default=5, help='updating frequency')  # 5
cmd_opt.add_argument('-isDone', type=bool, default=True, help='consider Done samples for reward calculation?')  # True

cmd_opt.add_argument('-epsilon', type=float, default=0.3, help='make random action probability')
cmd_opt.add_argument('-min_epsilon', type=float, default=0.05, help='min random exploration rate')
cmd_opt.add_argument('-epsilon_step_rate', type=float, default=1.0, help='range of epsilon changing steps')

cmd_opt.add_argument('-N', type=int, default=3, help='N for N-step DQN')
cmd_opt.add_argument('-gamma', type=float, default=0.95, help='DQN decay-factor for balance now and future')
cmd_opt.add_argument('-isDueling', type=bool, default=True, help='is Dueling DQN?')  # True
cmd_opt.add_argument('-isDouble', type=bool, default=True, help='is Double DQN?')  # True

cmd_opt.add_argument('-learning_rate', type=float, default=0.005, help='initial learning rate')
cmd_opt.add_argument('-lr_decay_rate', type=float, default=0.98, help='learning rate decay rate')
cmd_opt.add_argument('-min_learning_rate', type=float, default=0.001, help='min learning rate')


cmd_args, _ = cmd_opt.parse_known_args()
print(cmd_args)

if __name__ == '__main__':
    import joblib
    import pickle

    PATH = 'BA-60'
    sorted_obj_sum = pickle.load(open('../%s/sorted_valid.pkl' % PATH, 'rb'))
    print(sorted_obj_sum[:10])

