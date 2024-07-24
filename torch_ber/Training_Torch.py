
# I think it is the right thing. You know it.
from __future__ import  division
import sys
sys.path.append('/home/vic/文档/Projects/PRIMAL/od_mstar3')
import gym
import matplotlib
import numpy as np
import random

import torch
import torch.nn
import torch.nn.functional as F

import matliblib.pyplot as plt
import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing

import mapf_gym as mapf_gym
import pickle
import imageio

# Change hte {ACNet} to [ActorCritic}
from torch_ber.ACNet_torch import ActorCritic


# List available CUDA devices
cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

# Print CUDA devices
print("CUDA Devices:")
for i, device in enumerate(cuda_devices):
    print(f"Device {i}: {device}")

# Check if CUDA is available
if torch.cuda.is_available():
    print("\nCUDA is available")
else:
    print("\nCUDA is not available")

def make_git():
    '''TODO'''

# 更新目标图，将from的梯度都分配给to，也就是进行一个梯度的更换，在torch里面使用statedict可以实现。
def update_target_graph():
    '''TODO'''

def discount(x, gamma):
        return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x, gamma)


'''Parameters'''
num_workers = 0 # 暂时是0，后面训练代码会进行更新这个变量。

# Learning parameters
max_episode_length     = 64
# max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,70)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,.5) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size            = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10

# 2024-07-10 11:34:58
# NUM_META_AGENTS        = 3
# NUM_THREADS            = 8 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))

NUM_META_AGENTS        = 2
NUM_THREADS            = 3 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_Q                   = 2.e-5 #8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = 'model_primal'
gifs_path              = 'gifs_primal'
train_path             = 'train_primal'
GLOBAL_NET_SCOPE       = 'global'

#Imitation options
PRIMING_LENGTH         = 0    # number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB     = 0.5  # probability of training on a demonstration per episode

# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 0

# Shared arrays for tensorboard
episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_wrong_blocking = [ [] for _ in range(NUM_META_AGENTS) ]
rollouts               = [ None for _ in range(NUM_META_AGENTS)]
demon_probs=[np.random.rand() for _ in range(NUM_META_AGENTS)]
# episode_steps_on_goal  = [ [] for _ in range(NUM_META_AGENTS) ]
printQ                 = False # (for headless)
swarm_reward           = [0]*NUM_META_AGENTS


'''
Worker: worker thread. 

'''
# TODO understand how the worker thread works
class Worker:

    def __init__(self,game,metaAgentID,workerID,a_size,groupLock):
        self.env = game
        self.metaAgentID = metaAgentID
        self.workerID = workerID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1
        self.groupLock = groupLock

        # self.nextGIF = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        # TODO
        #  这里local_AC 是一个本地的网络，与global的网络要进行区分开。
        self.global_AC = ActorCritic(a_size,TRAINING)
        self.local_AC =ActorCritic(a_size,TRAINING)

    # 判断是否应该继续跑下去
    def shouldRun(self,episode_count):
        return episode_count < NUM_EXPS
    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if (not hasattr(self, "lock_bool")):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool

    def parse_path(self, path):
        '''needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)'''
        result = [[] for i in range(num_workers)]
        for t in range(len(path[:-1])):
            observations = []
            move_queue = list(range(num_workers))
            for agent in range(1, num_workers + 1):
                observations.append(self.env._observe(agent))
            steps = 0
            while len(move_queue) > 0:
                steps += 1
                i = move_queue.pop(0)
                o = observations[i]
                pos = path[t][i]
                newPos = path[t + 1][i]  # guaranteed to be in bounds by loop guard
                direction = (newPos[0] - pos[0], newPos[1] - pos[1])
                a = self.env.world.getAction(direction)
                # 这里我很不了理解为什么是i+1？agent 0的在t的动作a，为什么传给agent 1？
                # 明白了，这里是看a是不是对agent 1有阻塞？但是为什么只有agent 1？
                # 因为agent是逐个行动的？agent 0 agent 1 agent 2.
                state, reward, done, nextActions, on_goal, blocking, valid_action = self.env._step((i + 1, a))
                if steps > num_workers ** 2:
                    # if we have a very confusing situation where lots of agents move
                    # in a circle (difficult to parse and also (mostly) impossible to learn)
                    return None
                if not valid_action:
                    # the tie must be broken here
                    move_queue.append(i)
                    continue
                result[i].append([o[0], o[1], a])
        return result

    def train(self,rollout,gamma,bootstrap_value,rnn_state0,imitation=False):
        global episode_count
        if imitation:
            # Get data from rollout
            rollout = np.array(rollout)
            inputs = rollout[:,0]
            goal_pos = rollout[:,1]
            optimal_actions = rollout[:,2]
            state_in = [[] for i in range(2)]
            state_in[0] = rnn_state0[0]
            state_in[1] = rnn_state0[1]

            # initialize the hidden  variable
            hidden = net.init_hidden(1)

            # local_AC算法的输出数值
            policy, value, (hx, cx), blocking, on_goal,valids = self.local_AC(inputs,goal_pos,hidden)

            # Compute the imitation loss use the labels to change the one-hot encoding
            self.optimal_actions_onehot = F.one_hot(self.optimal_actions, num_classes=a_size).float()
            action_labels = torch.argmax(self.optimal_actions_onehot, dim=1)
            imitation_loss = F.cross_entropy(action_labels,policy)

            return imitation_loss

        # 这里是代码出现的最大的逻辑bug，这里的rollout最多只有3个，因为parse_path是这样传递的。
        # 那么下面的这些rollout，observation,goals是怎么得来的呢？ 是从与环境交互的episode中得来的
        # rollout是环境交互的经验池，此时得到下列变量。
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        goals = rollout[:, -2]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]
        valids = rollout[:, 6]
        blockings = rollout[:, 10]
        on_goals = rollout[:, 8]
        train_value = rollout[:, -1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages, gamma)

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save


        # 定义变量名称
        target_v = np.stack(discounted_rewards)
        inputs = np.stack(observations)
        goal_pos = np.stack(goals)
        train_valid = np.stack(valids)
        target_blocking = blockings
        target_on_goal = on_goals
        state_in = [[] for i in range(2)]
        state_in[0] = rnn_state0[0]
        state_in[1] = rnn_state0[1]

        # initialize the hidden  variable
        hidden = net.init_hidden(1)

        # 网络进行输出
        policy, value, (hx, cx), blocking, on_goal,valids = self.local_AC(inputs, goal_pos, hidden)

        # 对需要计算损失的变量进行tensor化，并计算损失

        # value_loss
        train_value = torch.tensor(train_value, dtype=torch.float32)
        target_v    = torch.tensor(target_v, dtype=torch.float32)
        reshaped_value = self.value.view(-1)  # 等价于 tf.reshape(self.value, shape=[-1])
        squared_diff = torch.square(self.target_v - reshaped_value)  # 等价于 tf.square(self.target_v - reshaped_value)

        value_loss = torch.sum(self.train_value * squared_diff)  # 等价于 tf.reduce_sum(self.train_value * squared_diff)

        # policy_loss
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        actions_onehot = F.one_hot(self.actions, num_classes=self.a_size).float()
        responsible_outputs = torch.sum(self.policy * self.actions_onehot, dim=1)
        clipped_outputs = torch.clamp(self.responsible_outputs, min=1e-15, max=1.0)
        log_outputs = torch.log(clipped_outputs)

        policy_loss = -torch.sum(log_outputs * self.advantages)

        #valid_loss
        train_valids = torch.tensor(train_valids, dtype=torch.float32)
        valids = torch.tensor(valids, dtype=torch.float32)
        clipped_valids = torch.clamp(valids, min=1e-10, max=1.0)
        clipped_one_minus_valids = torch.clamp(1 - valids, min=1e-10, max=1.0)
        log_valids = torch.log(clipped_valids)
        log_one_minus_valids = torch.log(clipped_one_minus_valids)

        valid_loss = -torch.sum(
            log_valids * self.train_valid + log_one_minus_valids * (1 - self.train_valid)
        )

        # entropy_loss
        clipped_policy = torch.clamp(policy, min=1e-10, max=1.0)
        log_policy = torch.log(clipped_policy)

        entropy_loss = -torch.sum(self.policy * log_policy)

        # blocking_loss
        target_blocking = torch.tensor(target_blockings, dtype=torch.float32)
        blocking = torch.tensor(blocking, dtype=torch.float32)
        clipped_blocking = torch.clamp(blocking, min=1e-10, max=1.0)
        clipped_one_minus_blocking = torch.clamp(1 - blocking, min=1e-10, max=1.0)

        log_blocking = torch.log(clipped_blocking)
        log_one_minus_blocking = torch.log(clipped_one_minus_blocking)

        # 计算损失
        blocking_loss = -torch.sum(
            target_blockings * log_blocking +
            (1 - target_blockings) * log_one_minus_blocking
        )

        # on_goal_loss
        target_on_goal = torch.tensor(target_on_goals, dtype=torch.float32)
        on_goal = torch.tensor(on_goal, dtype=torch.float32)
        clipped_on_goal = torch.clamp(on_goal, min=1e-10, max=1.0)
        clipped_one_minus_on_goal = torch.clamp(1 - on_goal, min=1e-10, max=1.0)

        # 计算对数
        log_on_goal = torch.log(clipped_on_goal)
        log_one_minus_on_goal = torch.log(clipped_one_minus_on_goal)

        # 计算损失
        on_goal_loss = -torch.sum(
            target_on_goals * log_on_goal +
            (1 - target_on_goals) * log_one_minus_on_goal
        )

        return value_loss/len(rollout), policy_loss/len(rollout), valid_loss/len(
            rollout),entropy_loss/len(rollout),blocking_loss/len(rollout),on_goal_loss/len(rollout)


    def work(self, max_episode_length, gamma):
