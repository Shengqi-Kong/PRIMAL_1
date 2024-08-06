
# I think it is the right thing. You know it.
# Holy, you know how hard for me to translate from tensorflow to pytorch




# ----- 设置系统路径 -----
from __future__ import  division
import sys
sys.path.append('/od_mstar3')
import numpy as np

# ----- 取消UserWarning and np.VisibleDeprecationWaring
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)
# ----- 引入torch框架 -----
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
# ----- 引入cpp mstar -----
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import scipy.signal as signal
import os
import GroupLock
# ----- 引入环境-引入网络 -----
import mapf_gym as mapf_gym
from ACNet_torch import ActorCritic
# ------ 设置系统路径 ------
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# ----- CUDA调试 -----
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
test_cuda = False
if test_cuda:
    # List available CUDA devices
    cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # Print CUDA devices
    print("CUDA Devices:")
    for i, device_i in enumerate(cuda_devices):
        print(f"Device {i}: {device_i}")
if torch.cuda.is_available():
    print("\nCUDA is available")
else:
    print("\nCUDA is not available")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)

# 暂时不写gif代码
def make_gif():
    '''TODO'''

# 更新目标图，将from的梯度都分配给to，也就是进行一个梯度的更换，在torch里面使用statedict可以实现。
def update_target_graph():
    '''TODO'''


# ----- discount的含义是
def discount(x, gamma):
        return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x, gamma)


# ----- 超参数 -----
num_workers = 4 # 暂时是4，后面训练代码会进行更新这个变量。
# Learning parameters
max_episode_length     = 64
# max_episode_length     = 256
# episode_count          = 0
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = 0.95 # discount rate for advantage estimation and reward discounting
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,70)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,0.5) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size            = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10

# 2024-07-10 11:34:58
# NUM_META_AGENTS        = 3
# NUM_THREADS            = 8 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
# NUM_META_AGENTS        = 2
# NUM_THREADS            = 3 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))

# 2024-08-01 23:15:44 假设如果不涉及线程同步，即A2C
NUM_META_AGENTS        = 1
NUM_THREADS            = 1

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
printQ                 = True # (for headless)
swarm_reward           = [0]*NUM_META_AGENTS


'''
Worker: worker thread. 

'''
# TODO understand how the worker thread works
class Worker:

    # def __init__(self, game, metaAgentID, workerID, a_size, groupLock):
    #
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock,global_network):

        self.env = game
        self.metaAgentID = metaAgentID
        self.workerID = workerID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1
        self.groupLock = groupLock
        self.a_size = a_size
        # 2024-08-01 21:48:14
        self.global_AC = global_network
        self.local_AC = ActorCritic(a_size, TRAINING)

        self.local_AC.to(device)
        self.global_AC.to(device)
        self.optimizer = optim.Adam(self.local_AC.parameters(),0.001)
        # self.nextGIF = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        # TODO
        #  这里local_AC 是一个本地的网络，与global的网络要进行区分开。
        # self.global_AC = ActorCritic(a_size,TRAINING)



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



        raise ValueError("Input must be a nested list or numpy array.")

    def train(self,rollout,gamma,bootstrap_value,rnn_state0,imitation=False):
        global episode_count
        if imitation:
            # Get data from rollout
            # print("rollout is:")
            # print('-'*20)
            # print('rollout.len:',len(rollout))
            # print('rollout[0].len:',len(rollout[0]))
            # print('rollout[0][0].len:',len(rollout[0][0]))
            # print('rollout[0][0][0].len:',len(rollout[0][0][0]))
            # print('rollout[0][0][0][0].len:',len(rollout[0][0][0][0]))

            # print(rollout)
            rollout = np.array(rollout)
            inputs = rollout[:,0]
            # print(inputs.dtype)
            inputs = np.array([np.array(sublist,dtype=np.float32) for sublist in inputs],dtype=np.float32)
            goal_pos = rollout[:,1]
            goal_pos = np.array([np.array(sublist,dtype=np.float32) for sublist in goal_pos],dtype=np.float32)
            optimal_actions = rollout[:,2]
            optimal_actions = [sub_optimal for sub_optimal in optimal_actions ]
            optimal_actions = torch.tensor(optimal_actions,dtype=torch.float32)
            state_in = [[] for i in range(2)]
            state_in[0] = rnn_state0[0]
            state_in[1] = rnn_state0[1]

            # initialize the hidden  variable
            # hidden = self.local_AC.init_hidden(1)
            # TODO 这里需要对输入进行处理，这里的batch_size是多少呢？

            # 2024-08-03 17:25:07 这里我把input改换成了tensor
            # input_tensor = [[] for i in range(len(inputs))]
            # for i_len in range(len(inputs)):
            #     for j_len in range(len(inputs[i_len])):
            #         input_tensor[i_len].append(inputs[i_len][j_len])
            # inputs
            inputs_tensor  = torch.tensor(inputs)
            inputs = inputs_tensor
            # print(input_tensor)

            # goal_tensor = [[] for i in range(len(goal_pos))]
            goal_tensor = torch.tensor(goal_pos,dtype=torch.float32)
            # for i_len in range(len(goal_pos)):
            #     goal_tensor.append(goal_pos[i_len])

            print_flag = False
            if print_flag:
                print(inputs_tensor.shape)
                print('-'*20)
                print(torch.tensor(goal_tensor).shape)
                print('-'*20)
                print(type(goal_pos))
                print(np.array(goal_pos).shape)
                print(goal_pos)
            # goal_tensor = torch.tensor(goal_tensor,dtype=torch.float32)
            goal_pos = goal_tensor

            print("-"*20)
            print(type(state_in))
            # 将数据转移到cuda上
            inputs = inputs.to(device)
            goal_pos = goal_pos.to(device)
            state_in = [state_in[0].to(device),state_in[1].to(device)]
            # local_AC算法的输出数值

            policy, value, (hx, cx), blocking, on_goal,valids = self.local_AC(inputs,goal_pos,state_in)

            # Compute the imitation loss use the labels to change the one-hot encoding
            self.optimal_actions_onehot = F.one_hot(optimal_actions, num_classes=a_size).float()
            action_labels = torch.argmax(self.optimal_actions_onehot, dim=1)
            imitation_loss = F.cross_entropy(action_labels,policy)

            return imitation_loss

        # 这里是代码出现的最大的逻辑bug，这里的rollout最多只有3个，因为parse_path是这样传递的。
        # 那么下面的这些rollout，observation,goals是怎么得来的呢？ 是从与环境交互的episode中得来的
        # rollout是环境交互的经验池，此时得到下列变量。

        # 2024-08-03 17:48:35 下面的这些问题都可以被适当解决
        # rollout = np.array(rollout)
        # rollout = np.array(rollout.detach().numpy(), dtype=np.float32)
        # print

        rollout = np.array(rollout)
        # observation & goals : stay mess
        observations = rollout[:, 0]
        goals = rollout[:, -2]
        # actions : list
        actions = rollout[:, 1]
        actions = [action for action in actions]

        # reward : list
        rewards = rollout[:, 2]
        rewards = [reward for reward in rewards]
        # values : tensor
        values = rollout[:, 5]
        values = [value for value in values]
        values = torch.stack(values).to(device)
        # valids : tensor
        valids = rollout[:, 6]
        valids = [valid for valid in valids]
        valids = torch.stack(valids).to(device)

        # blockings : list
        blockings = rollout[:, 10]
        blockings = [blocking for blocking in blockings]
        # on_goals : list
        on_goals = rollout[:, 8]
        on_goals = [on_goal_sub for on_goal_sub in on_goals]

        train_value = rollout[:, -1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        # self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        rewards_plus = np.asarray(rewards + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        # self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = np.array(rewards) + gamma * value_plus[1:] - value_plus[:-1]
        advantages = good_discount(advantages, gamma)
        advantages = np.array([advantage for advantage in advantages])
        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # 定义变量名称
        # 2024-08-04 10:39:04 这里包括上面的变量在设置的时候都是以numpy的array形式出现的，在进行运算时都需要进行
        # tensor处理。
        target_v = np.stack(discounted_rewards)
        target_v = torch.tensor(target_v, dtype=torch.float32)

        #
        inputs = np.stack(observations)
        inputs_tensor = torch.tensor(inputs,dtype=torch.float32)
        goal_pos = np.stack(goals)
        goal_pos_tensor = torch.tensor(goal_pos,dtype=torch.float32)

        # train_valid = np.stack(valids)
        train_valid = valids
        # target_blocking = blockings
        target_blocking = torch.tensor(blockings)
        # target_on_goal = on_goals
        target_on_goal = torch.tensor(on_goals)
        state_in = [[] for i in range(2)]

        state_in = [rnn_state0[0].to(device), rnn_state0[1].to(device)]

        # 网络进行输出
        policy, value, (hx, cx), blocking, on_goal,valids = self.local_AC(inputs_tensor, goal_pos_tensor, state_in)

        # value_loss
        # 2024-08-01 22:20:28 value_loss应该用现成的误差函数来计算，否则看起来不太对。
        # TODO train_value是用来干什么的？
        # criterion = nn.MSEloss()

        # train_value = torch.tensor(train_value, dtype=torch.float32)
        # target_v    = torch.tensor(target_v, dtype=torch.float32)
        # reshaped_value = value.view(-1)  # 等价于 tf.reshape(self.value, shape=[-1])
        # squared_diff = torch.square(self.target_v - reshaped_value)  # 等价于 tf.square(self.target_v - reshaped_value)

        # value_loss = torch.sum(self.train_value * squared_diff)  # 等价于 tf.reduce_sum(self.train_value * squared_diff)
        value_loss = F.mse_loss(value.view(-1), target_v)


        # policy_loss
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        actions_onehot = F.one_hot(actions, num_classes=self.a_size).float()
        responsible_outputs = torch.sum(policy * actions_onehot, dim=1)
        clipped_outputs = torch.clamp(responsible_outputs, min=1e-15, max=1.0)
        log_outputs = torch.log(clipped_outputs)

        policy_loss = -torch.sum(log_outputs * advantages)

        #valid_loss
        # train_valid = torch.tensor(train_valid, dtype=torch.float32)
        valids = torch.tensor(valids, dtype=torch.float32)
        clipped_valids = torch.clamp(valids, min=1e-10, max=1.0)
        clipped_one_minus_valids = torch.clamp(1 - valids, min=1e-10, max=1.0)
        log_valids = torch.log(clipped_valids)
        log_one_minus_valids = torch.log(clipped_one_minus_valids)

        valid_loss = -torch.sum(
            log_valids * train_valid + log_one_minus_valids * (1 - train_valid)
        )

        # entropy_loss
        clipped_policy = torch.clamp(policy, min=1e-10, max=1.0)
        log_policy = torch.log(clipped_policy)

        entropy_loss = -torch.sum(policy * log_policy)

        # blocking_loss
        # target_blocking = torch.tensor(target_blocking, dtype=torch.float32)
        # blocking = torch.tensor(blocking, dtype=torch.float32)
        clipped_blocking = torch.clamp(blocking, min=1e-10, max=1.0)
        clipped_one_minus_blocking = torch.clamp(1 - blocking, min=1e-10, max=1.0)

        log_blocking = torch.log(clipped_blocking)
        log_one_minus_blocking = torch.log(clipped_one_minus_blocking)

        # 计算损失
        blocking_loss = -torch.sum(
            target_blocking * log_blocking.view(-1) +
            (1 - target_blocking) * log_one_minus_blocking.view(-1)
        )

        # on_goal_loss
        # target_on_goal = torch.tensor(target_on_goal, dtype=torch.float32)
        # on_goal = torch.tensor(on_goal, dtype=torch.float32)
        clipped_on_goal = torch.clamp(on_goal, min=1e-10, max=1.0)
        clipped_one_minus_on_goal = torch.clamp(1 - on_goal, min=1e-10, max=1.0)

        # 计算对数
        log_on_goal = torch.log(clipped_on_goal)
        log_one_minus_on_goal = torch.log(clipped_one_minus_on_goal)

        # 计算损失
        on_goal_loss = -torch.sum(
            target_on_goal * log_on_goal.view(-1) +
            (1 - target_on_goal) * log_one_minus_on_goal.view(-1)
        )
        print('train is returning')
        return value_loss/len(rollout), policy_loss/len(rollout), valid_loss/len(
            rollout),entropy_loss/len(rollout),blocking_loss/len(rollout),on_goal_loss/len(rollout)

    def pull_global(self, global_AC, local_AC):
        # 将全局参数更新到本地模型中
        for global_param, local_param in zip(global_AC.parameters(), local_AC.parameters()):
            local_param.data.copy_(global_param.data)

    def work(self, max_episode_length, gamma):
        global episode_count, swarm_reward, \
            episode_rewards, episode_lengths, \
            episode_mean_values, episode_invalid_ops, \
            episode_wrong_blocking  # , episode_invalid_goals
        total_steps, i_buf = 0, 0
        episode_buffers, s1Values = [[] for _ in range(NUM_BUFFERS)], [[] for _ in range(NUM_BUFFERS)]
        # torch.save(self.global_AC.state_dict(), f'{model_path}/model-{int(episode_count)}.pt')

        while(self.shouldRun(episode_count)):
            # 把global网络的参数pull到local网络中

            # def get_model_device(model):
            #     # 迭代模型参数，找到第一个参数的设备
            #     for param in model.parameters():
            #         return param.device
            #
            # # 输出模型的设备信息
            # model_device = get_model_device(self.global_AC)
            # print(f'The model is on device: {model_device}')
            # torch.save({'model_state_dict': self.global_AC.state_dict()}, 'global_ac_checkpoint.pth')
            # torch.save(self.global_AC.state_dict(), f'{model_path}/model-{int(episode_count)}.pt')

            self.local_AC.load_state_dict(self.global_AC.state_dict())
            # self.local_AC.load_state_dict(torch.load(f'{model_path}/model-{int(episode_count)}.pt'))
            episode_buffer, episode_values = [], []
            episode_reward = episode_step_count = episode_inv_count = 0
            d = False
            print("Episode {}/{}".format(episode_count,NUM_EXPS))
            # Initial state from the environment
            if self.agentID == 1:
                self.env._reset(self.agentID)
            # self.synchronize()  # synchronize starting time of the threads
            validActions = self.env._listNextValidActions(self.agentID)
            s = self.env._observe(self.agentID)
            blocking = False
            p = self.env.world.getPos(self.agentID)


            on_goal = self.env.world.goals[p[0], p[1]] == self.agentID
            s = self.env._observe(self.agentID)
            rnn_state = self.local_AC.state_init
            rnn_state0 = rnn_state
            RewardNb = 0
            wrong_blocking = 0
            wrong_on_goal = 0

            if self.agentID == 1:
                global demon_probs
                demon_probs[self.metaAgentID] = np.random.rand()
            # self.synchronize()  # synchronize starting time of the threads


            # reset swarm_reward (for tensorboard)
            swarm_reward[self.metaAgentID] = 0
            if episode_count < PRIMING_LENGTH or demon_probs[self.metaAgentID] < DEMONSTRATION_PROB:
                # for the first PRIMING_LENGTH episodes, or with a certain probability
                # don't train on the episode and instead observe a demonstration from M*
                if self.workerID == 1 and episode_count % 100 == 0:
                    # saver.save(sess, model_path + '/model-' + str(int(episode_count)) + '.cptk')
                    torch.save(self.global_AC.state_dict(), f'{model_path}/model-{int(episode_count)}.pt')
                global rollouts
                rollouts[self.metaAgentID] = None
                if (self.agentID == 1):
                    world = self.env.getObstacleMap()
                    start_positions = tuple(self.env.getPositions())
                    goals = tuple(self.env.getGoals())
                    try:
                        mstar_path = cpp_mstar.find_path(world, start_positions, goals, 2, 5)
                        rollouts[self.metaAgentID] = self.parse_path(mstar_path)
                    except OutOfTimeError:
                        # M* timed out
                        print("timeout", episode_count)
                    except NoSolutionError:
                        print("nosol????", episode_count, start_positions)

                # self.synchronize()
                if rollouts[self.metaAgentID] is not None:
                    imitation_loss = self.train(rollouts[self.metaAgentID][self.agentID - 1], gamma, None, rnn_state0,
                                     imitation=True)
                    episode_count += 1. / num_workers
                    self.optimizer.zero_grad()
                    imitation_loss.backward()
                    self.optimizer.step()
                    # if self.agentID == 1:
                        # summary = tf.Summary()
                        # summary.value.add(tag='Losses/Imitation loss', simple_value=i_l)
                        # global_summary.add_summary(summary, int(episode_count))
                        # global_summary.flush()
                    # TODO 这里应该对imitation_loss的进行反向传播，第一个问题，A3C算法是隔一段时间
                    # 把local的parameters更新到global，还是把local的梯度更新到global？

                    # 如果是local梯度来更新global，torch中是如何实现的？

                    continue
                continue
            saveGIF = False

            if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                saveGIF = True
                self.nextGIF = episode_count + 64
                GIF_episode = int(episode_count)
                episode_frames = [self.env._render(mode='rgb_array', screen_height=900, screen_width=900)]
            print_training = 0
            while(not self.env.finished):
                if print_training % 100 == 0:
                    print("Training...")

                print_training += 1
                inputs = [s[0]]
                goal_pos = [s[1]]
                state_in = [[] for i in range(2)]
                state_in[0] = rnn_state[0]
                state_in[1] = rnn_state[1]
                # hidden = self.local_AC.init_hidden(1)

                inputs = torch.tensor(inputs,dtype=torch.float32)
                goal_pos = torch.tensor(goal_pos,dtype=torch.float32)
                # TODO 2024-08-03 11:46:39 修改了state_in的代码
                a_dist, v, rnn_state, pred_blocking, pred_on_goal,_ = self.local_AC(inputs,goal_pos,state_in)

                if (not (torch.argmax(a_dist.flatten()) in validActions)):
                    episode_inv_count += 1
                train_valid = torch.zeros(a_size)
                train_valid[validActions] = 1

                # valid_dist = np.array([a_dist[0, validActions]])
                # valid_dist = np.array([a_dist[0, validActions]])
                valid_dist = np.array([a_dist[0, validActions].detach().numpy()])
                valid_dist /= np.sum(valid_dist)
                # train_val 这里都是1，可以认为是没有价值的
                if TRAINING:
                    if (pred_blocking.flatten()[0] < 0.5) == blocking:
                        wrong_blocking += 1
                    if (pred_on_goal.flatten()[0] < 0.5) == on_goal:
                        wrong_on_goal += 1
                    a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                    train_val = 1.
                else:
                    a = np.argmax(a_dist.flatten())
                    if a not in validActions or not GREEDY:
                        a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                    train_val = 1.

                _, r, _, _, on_goal, blocking, _ = self.env._step((self.agentID, a), episode=episode_count)
                # self.synchronize()  # synchronize threads
                # Get common observation for all agents after all individual actions have been performed
                s1 = self.env._observe(self.agentID)
                validActions = self.env._listNextValidActions(self.agentID, a, episode=episode_count)
                d = self.env.finished

                if saveGIF:
                    episode_frames.append(self.env._render(mode='rgb_array', screen_width=900, screen_height=900))

                # 2024-08-04 20:41:44 这里我对变量进行了detach（）
                episode_buffer.append(
                    [s[0], a, r, s1, d, v[0, 0].detach(), train_valid.detach(), pred_on_goal.detach(), int(on_goal), pred_blocking.detach(),
                     int(blocking), s[1], train_val])
                # with torch.no_grad():
                #     episode_values.append(v[0, 0])
                episode_values.append(v[0,0].detach().numpy())
                episode_reward += r
                s = s1
                total_steps += 1
                episode_step_count += 1

                if r > 0:
                    RewardNb += 1
                if d == True:
                    print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

                if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                    # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                    if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                        episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                    else:
                        episode_buffers[i_buf] = episode_buffer[:]

                    if d:
                        s1Values[i_buf] = 0
                    else:
                        hidden = self.local_AC.init_hidden(1)
                        _,s1Values[i_buf], _,_,_,_ = self.local_AC(torch.tensor(np.array([s[0]])),\
                                                                   torch.tensor([s[1]]),
                                                                   hidden
                                                                   )
                    if (episode_count - EPISODE_START) < NUM_BUFFERS:
                        i_rand = np.random.randint(i_buf + 1)
                    else:
                        i_rand = np.random.randint(NUM_BUFFERS)
                        tmp = np.array(episode_buffers[i_rand])
                        while tmp.shape[0] == 0:
                            i_rand = np.random.randint(NUM_BUFFERS)
                            tmp = np.array(episode_buffers[i_rand])

                    v_l,p_l, valid_l,e_l,b_l,og_l = self.train(episode_buffers[i_rand],gamma,s1Values[i_rand],rnn_state0)
                    total_loss = 0.5 * v_l + p_l + 0.5 * valid_l - 0.01 * e_l + 0.5 * b_l + 0.5 * og_l
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    i_buf = (i_buf + 1) % NUM_BUFFERS
                    rnn_state0 = rnn_state
                    episode_buffers[i_buf] = []
                # self.synchronize()  # synchronize threads
                # sess.run(self.pull_global)
                if episode_step_count >= max_episode_length or d:
                    break
            # episode_values = np.array([episode_values.detach().numpy()])
            episode_lengths[self.metaAgentID].append(episode_step_count)
            episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
            episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
            episode_wrong_blocking[self.metaAgentID].append(wrong_blocking)

            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                print('                                                                                   ',
                      end='\r')
                print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb), end='\r')

            swarm_reward[self.metaAgentID] += episode_reward
            # self.synchronize()  # synchronize threads

            episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

            if not TRAINING:
                mutex.acquire()
                if episode_count < NUM_EXPS:
                    plan_durations[episode_count] = episode_step_count
                if self.workerID == 1:
                    episode_count += 1
                    print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count,
                                                                                          self.workerID,
                                                                                          episode_step_count,
                                                                                          episode_reward,
                                                                                          episode_inv_count))
                GIF_episode = int(episode_count)
                mutex.release()
            else:
                episode_count += 1. / num_workers

                if episode_count % SUMMARY_WINDOW == 0:
                    if episode_count % 100 == 0:
                        print('Saving Model', end='\n')
                        torch.save(self.global_AC.state_dict(), f'{model_path}/model-{int(episode_count)}.pt')
                        # saver.save(sess, model_path + '/model-' + str(int(episode_count)) + '.cptk')
                        print('Saved Model', end='\n')
                    SL = SUMMARY_WINDOW * num_workers
                    mean_reward = np.nanmean(episode_rewards[self.metaAgentID][-SL:])
                    mean_length = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                    mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                    mean_invalid = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                    mean_wrong_blocking = np.nanmean(episode_wrong_blocking[self.metaAgentID][-SL:])
                    # current_learning_rate = sess.run(lr, feed_dict={global_step: episode_count})

                    # if True:
                    #     summary = tf.Summary()
                    #     summary.value.add(tag='Perf/Learning Rate', simple_value=current_learning_rate)
                    #     summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                    #     summary.value.add(tag='Perf/Length', simple_value=mean_length)
                    #     summary.value.add(tag='Perf/Valid Rate',
                    #                       simple_value=(mean_length - mean_invalid) / mean_length)
                    #     summary.value.add(tag='Perf/Blocking Prediction Accuracy',
                    #                       simple_value=(mean_length - mean_wrong_blocking) / mean_length)
                    #
                    #     summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                    #     summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                    #     summary.value.add(tag='Losses/Blocking Loss', simple_value=b_l)
                    #     summary.value.add(tag='Losses/On Goal Loss', simple_value=og_l)
                    #     summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                    #     summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                    #     summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                    #     global_summary.add_summary(summary, int(episode_count))
                    #
                    #     global_summary.flush()

                    if printQ:
                        print('{} Tensorboard updated ({})'.format(episode_count, self.workerID), end='\r')
            # if saveGIF:
            #     # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
            #     time_per_step = 0.1
            #     images = np.array(episode_frames)
            #     if TRAINING:
            #         make_gif(images,
            #                  '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode, episode_step_count,
            #                                                           swarm_reward[self.metaAgentID]))
            #     else:
            #         make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path, GIF_episode, episode_step_count),
            #                  duration=len(images) * time_per_step, true_image=True, salience=False)
            # if SAVE_EPISODE_BUFFER:
            #     with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
            #         pickle.dump(episode_buffer, file)



# Training Part
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)


if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

# Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO  这里global-network是怎么和local_network进行协调的？



global_network = ActorCritic(a_size,True).to(device)
lr = 0.001

# optimizer = optim.Adam(global_network.parameters(),lr)

if TRAINING:
    num_workers = NUM_THREADS  # Set workers # = # of available CPU threads
else:
    num_workers = NUM_THREADS
    NUM_META_AGENTS = 1

gameEnvs, workers, groupLocks = [], [], []
n = 1  # counter of total number of agents (for naming)
for ma in range(NUM_META_AGENTS):
    num_agents = NUM_THREADS
    gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                               observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
    gameEnvs.append(gameEnv)

    # Create groupLock
    workerNames = ["worker_" + str(i) for i in range(n, n + num_workers)]
    groupLock = GroupLock.GroupLock([workerNames, workerNames])
    groupLocks.append(groupLock)

    # Create worker classes
    workersTmp = []
    for i in range(ma * num_workers + 1, (ma + 1) * num_workers + 1):
        workersTmp.append(Worker(gameEnv, ma, n, a_size, groupLock,global_network=global_network))
        n += 1
    workers.append(workersTmp)

if load_model == True:
    print('Loading Model...')
    if not TRAINING:
        # 创建检查点文件
        with open(os.path.join(model_path, 'checkpoint'), 'w') as file:
            file.write(f'model_checkpoint_path: "model-{int(episode_count)}.pt"')
            file.close()
        # 获取检查点文件路径
    checkpoint_path = os.path.join(model_path, f'model-{int(episode_count)}.pt')

    # 检查文件是否存在
    if os.path.isfile(checkpoint_path):
        # 加载模型权重
        global_network.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")

        # 提取模型编号
        p = os.path.basename(checkpoint_path)
        p = p[p.find('-') + 1:]
        p = p[:p.find('.')]
        episode_count = int(p)
        print("episode_count set to ", episode_count)
    else:
        print("No checkpoint found at", checkpoint_path)
    # if RESET_TRAINER:
    #     optimizer = optim.Adam(global_network.parameters(), lr=lr)

worker_threads = []

# for ma in range(NUM_META_AGENTS):
#     for worker in workers[ma]:
#         groupLocks[ma].acquire(0, worker.name)  # synchronize starting time of the threads
#         # worker_work = lambda: worker.work(max_episode_length, gamma)
#
#         # 2024-08-02 16:42:54 根据资料进行修改
#         worker_work = lambda max_episode_length, gamma: worker.work(max_episode_length, gamma)
#
#         p = mp.Process(target=worker_work, args=(max_episode_length, gamma))
#         p.start()
#         worker_threads.append(p)
#
# for p in worker_threads:
#     p.join()

workers[0][0].work(max_episode_length,gamma)

print('All workers')



