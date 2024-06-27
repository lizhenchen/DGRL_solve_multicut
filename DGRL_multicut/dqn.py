# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# CUDA_VISIBLE_DEVICES = 3
import torch
torch.set_num_threads(1)
# torch.set_num_interop_threads(8)
# print(torch.get_num_threads())
# print(torch.get_num_interop_threads())

import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import shutil
import os
import warnings
warnings.filterwarnings("ignore")
from q_net import QNet
from cmd_args import cmd_args


PATH = cmd_args.folder_name
BATCH_SIZE = cmd_args.batch_size
GAMMA = cmd_args.gamma
EPSILON = cmd_args.epsilon
N = cmd_args.N
MEMORY_CAPACITY = cmd_args.memory_capacity
GLOBAL_NUM_STEPS = cmd_args.big_cycles
GLOBAL_NUM_GRAPHS = cmd_args.num_graphs
NUM_NODES = cmd_args.num_nodes


class Agent(object):
    def __init__(self, g_list, g_original_nx, env):

        self.g_list = g_list
        self.g_original_nx = g_original_nx

        self.env = env
        self.net = QNet()
        self.old_net = QNet()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=cmd_args.learning_rate)

        self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                     base_lr=cmd_args.min_learning_rate, max_lr=cmd_args.learning_rate,
                                                     step_size_up=250, step_size_down=250, mode='exp_range', gamma=cmd_args.lr_decay_rate,
                                                     scale_mode='cycle', cycle_momentum=False)  # cycle_momentum=False

        if cmd_args.ctx == 'gpu':
            self.net = self.net.cuda(cmd_args.cuda)
            self.old_net = self.old_net.cuda(cmd_args.cuda)

        self.memory_counter = 0
        self.memory = {}
        self.main_step = 0
        self.n_step_memory = {}
        self.rewards_memory = {}

        self.learn_step_counter = 0
        self.loss_func = nn.MSELoss()

        self.eps_start = cmd_args.epsilon
        self.eps_end = cmd_args.min_epsilon
        self.eps_step = cmd_args.epsilon_step_rate * GLOBAL_NUM_STEPS

    def store_transition(self, s0, s1, act_inds, r, s_0, s_1, dones_before, dones_after):

        for i in range(len(r)):
            if dones_before[i] is False:
                transition = (s0[i], s1[i], act_inds[i], r[i], s_0[i], s_1[i], dones_after[i])
                index = self.memory_counter % MEMORY_CAPACITY
                self.memory[index] = transition
                self.memory_counter += 1

    def make_actions(self, cur_state_g, cur_state_cg, dones_after, greedy=False):

        eps = self.eps_end + max(0., (self.eps_start - self.eps_end) * (self.eps_step - self.main_step) / self.eps_step)

        if greedy is False:
            action_lists = []
            action_index_lists = []
            if np.random.uniform() > eps:
                actions_q_values, _ = self.net(cur_state_g, cur_state_cg)
                for i in range(len(cur_state_cg)):
                    if dones_after[i] is False:
                        action_index = torch.max(actions_q_values[i], 0)[1].cpu().data.numpy()[0]
                        action_lists.append(list(cur_state_cg[i].edges())[action_index])
                        action_index_lists.append(action_index)
                    else:
                        action_lists.append(None)
                        action_index_lists.append(None)
            else:
                for i in range(len(cur_state_cg)):
                    if dones_after[i] is False:
                        num_actions = cur_state_cg[i].number_of_edges()
                        action_index = np.random.randint(num_actions)
                        action_lists.append(list(cur_state_cg[i].edges())[action_index])
                        action_index_lists.append(action_index)
                    else:
                        action_lists.append(None)
                        action_index_lists.append(None)
        else:
            action_lists = []
            action_index_lists = []
            actions_q_values, _ = self.net(cur_state_g, cur_state_cg)
            for i in range(len(cur_state_cg)):
                if dones_after[i] is False:
                    action_index = torch.max(actions_q_values[i], 0)[1].cpu().data.numpy()[0]
                    action_lists.append(list(cur_state_cg[i].edges())[action_index])
                    action_index_lists.append(action_index)
                else:
                    action_lists.append(None)
                    action_index_lists.append(None)

        return action_lists, action_index_lists

    def learn(self):

        if self.learn_step_counter == 0 and self.main_step % cmd_args.update_freq == 0:

            tmp_state_dict = self.net.state_dict().copy()

            soft = cmd_args.target_replace_soft
            if soft == 1.0:
                pass
            else:
                for item in tmp_state_dict:
                    tmp_state_dict[item] = soft * tmp_state_dict[item] + (1.0 - soft) * self.old_net.state_dict()[item]

            self.old_net.load_state_dict(tmp_state_dict)

        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE, replace=False).tolist()

        batch_state_g = [self.memory[i][0] for i in sample_index]
        batch_state_cg = [self.memory[i][1] for i in sample_index]
        batch_a_index = [self.memory[i][2] for i in sample_index]
        batch_r = torch.FloatTensor([[self.memory[i][3]] for i in sample_index])
        if cmd_args.ctx == 'gpu':
            batch_r = batch_r.cuda(cmd_args.cuda)
        batch_state_g_ = [self.memory[i][4] for i in sample_index]
        batch_state_cg_ = [self.memory[i][5] for i in sample_index]
        batch_done_samples = [self.memory[i][6] for i in sample_index]

        _, q_eval = self.net(batch_state_g, batch_state_cg)

        if cmd_args.ctx == 'gpu':
            q_eval = torch.gather(q_eval, dim=0, index=torch.LongTensor([batch_a_index]).cuda(cmd_args.cuda)).view(BATCH_SIZE, 1)
        else:
            q_eval = torch.gather(q_eval, dim=0, index=torch.LongTensor([batch_a_index])).view(BATCH_SIZE, 1)

        if cmd_args.isDouble:
            q_next_max = []
            next_q_values, _ = self.old_net(batch_state_g_, batch_state_cg_)
            q_next_list, _ = self.net(batch_state_g_, batch_state_cg_)

            if cmd_args.isDone:
                for i in range(BATCH_SIZE):
                    if batch_done_samples[i] is True:  # if H[i].shape[0] == 0:
                        q_next_max.append(0)
                    else:
                        argmax_actions = q_next_list[i].max(0)[1].detach().item()
                        q_next_max.append(next_q_values[i][argmax_actions].detach())
            else:
                for i in range(BATCH_SIZE):
                    argmax_actions = q_next_list[i].max(0)[1].detach().item()
                    q_next_max.append(next_q_values[i][argmax_actions].detach())

            q_next_max = torch.FloatTensor(q_next_max).view(BATCH_SIZE, 1)

        else:
            H, _ = self.old_net(batch_state_g_, batch_state_cg_)

            if cmd_args.isDone:
                q_next_max = []
                for i in range(BATCH_SIZE):
                    if batch_done_samples[i] is True:  # if H[i].shape[0] == 0:
                        q_next_max.append([0])
                    else:
                        q_next_max.append([H[i].detach().max(0)[0].cpu().data.numpy()[0]])
                q_next_max = torch.FloatTensor(q_next_max)
            else:
                q_next_max = torch.FloatTensor([[H[j].detach().max(0)[0].cpu().data.numpy()[0]] for j in range(BATCH_SIZE)])

        if cmd_args.ctx == 'gpu':
            q_next_max = q_next_max.cuda(cmd_args.cuda)

        q_target = batch_r + (GAMMA ** N) * q_next_max  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)

        if self.learn_step_counter == 1:
            print('Loss:  ', loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_simulation(self):

        self.net.train()
        self.old_net.eval()

        # ordinal = math.floor(self.main_step/cmd_args.change_dataset) % cmd_args.batch_sub_dataset
        # self.env.setup(self.g_list[ordinal * cmd_args.num_sub_dataset: (ordinal+1) * cmd_args.num_sub_dataset], self.g_original_nx[ordinal * cmd_args.num_sub_dataset: (ordinal+1) * cmd_args.num_sub_dataset])
        graph_indexes = np.random.choice(GLOBAL_NUM_GRAPHS, cmd_args.num_sub_dataset, replace=False).tolist()
        self.env.setup([self.g_list[i] for i in graph_indexes], [self.g_original_nx[i] for i in graph_indexes])

        ep_r = [0] * cmd_args.num_sub_dataset
        small_step = 0
        dones_after = [False] * cmd_args.num_sub_dataset

        cur_state_g, cur_state_cg = self.env.getState()

        while True:

            list_actions, list_action_indexs = self.make_actions(cur_state_g, cur_state_cg, dones_after)
            self.n_step_memory[small_step % N] = (deepcopy(cur_state_g), deepcopy(cur_state_cg), list_action_indexs.copy())

            rewards, cur_state_g, cur_state_cg, dones_before, dones_after, counts_before, counts_after = self.env.step(list_actions)
            self.rewards_memory[small_step] = rewards.copy()

            small_step += 1

            if small_step >= N:
                num = small_step % N
                accumulated_rewards = np.sum([[j * (GAMMA ** (i - small_step + N)) for j in self.rewards_memory[i]] for i in range(small_step - N, small_step)], axis=0).tolist()
                # del self.rewards_memory[small_step - N]
                self.store_transition(self.n_step_memory[num][0], self.n_step_memory[num][1], self.n_step_memory[num][2], accumulated_rewards, deepcopy(cur_state_g), deepcopy(cur_state_cg), dones_before, dones_after)

            ep_r = np.sum([ep_r, rewards], axis=0).tolist()

            if self.memory_counter >= (cmd_args.memory_learn_start_rate * MEMORY_CAPACITY) and (small_step - 1) % cmd_args.learn_freq == 0:
                self.learn()

            if counts_after >= len(rewards):
                print('\nBig cycle (Train): ', self.main_step, '| Total reward: ', round(sum(ep_r), 3))
                break

        self.learn_step_counter = 0
        self.main_step += 1

        return sum(ep_r)

    def train(self):

        pbar = tqdm(range(GLOBAL_NUM_STEPS), unit='steps')

        if os.path.exists('../%s/savemodels/' % PATH):
            shutil.rmtree('../%s/savemodels/' % PATH)
        os.mkdir('../%s/savemodels/' % PATH)

        sum_ep_r_lists = []

        if cmd_args.isPretrain:
            pretrain_weights = torch.load('../%s/pretrain.pth' % PATH, map_location=lambda storage, loc: storage.cuda(cmd_args.cuda))
            self.net.load_state_dict(pretrain_weights)

        for self.step in pbar:

            sum_ep_r = self.run_simulation()
            sum_ep_r_lists.append(sum_ep_r)

            if (self.step + 1) % 1 == 0:  # 100
                torch.save(self.net.state_dict(), '../%s/savemodels/checkpoint-%05d.pth' % (PATH, self.step + 1))

            self.lr_scheduler.step()

    def test(self):

        self.net.eval()

        self.env.setup(self.g_list, self.g_original_nx)
        small_step = 0
        dones_after = [False] * len(self.g_list)

        cur_state_g, cur_state_cg = self.env.getState()
        while True:
            small_step += 1

            list_actions, list_action_indexs = self.make_actions(cur_state_g, cur_state_cg, dones_after, greedy=True)

            dones_after, counts_after = self.env.step(list_actions, train=False)

            if counts_after == len(self.g_list):
                break

        objective = self.env.objective()

        return objective
