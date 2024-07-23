import sys
sys.path.append('/home/vic/文档/Projects/PRIMAL/od_mstar3')
import gym
import matplotlib
import numpy as np
import random

import torch
import torch.nn


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
from torch_ber.ACNet_torch import ACNet


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
a_size                 = 5 + int(DIAG_MVMT)*4
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
MODEL_NUMBER           = 313000

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




class Worker:

    def __init__(self,game,metaAgentID,workerID,a_size,groupLock):
        self.env = game
        self.metaAgentID = metaAgentID
        self.workerID = workerID
        self.name = "worker"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1
        self.groupLock = groupLock

        # self.nextGIF = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        # TODO
        #  这里local_AC 是一个本地的网络，与global的网络要进行区分开。

        self.local_AC = ACNet(self.name, a_size, trainer, True, GRID_SIZE, GLOBAL_NET_SCOPE)

        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)


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

