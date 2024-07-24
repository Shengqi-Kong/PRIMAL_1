import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# parameters for training
GRAD_CLIP = 1000.0
KEEP_PROB1 = 1 # was 0.5
KEEP_PROB2 = 1 # was 0.7
RNN_SIZE = 512
GOAL_REPR_SIZE = 12


# 进行列归一化，作用在参数变量上
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(0, keepdim=True))
    return out

# 对变量的权重进行初始化，分别对Conv和Linear两个权重进行初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# Used to initialize weights for policy and value output layers
# def normalized_columns_initializer(weights, std=1.0):
#     out = np.random.randn(*weights.shape).astype(np.float32)
#     out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#     return torch.tensor(out)

'''

action_dim: a_size,action tensor dimention,5 or 9(diag)
grid_size: GRID_SIZE,state tensor dimention,10


'''
class ActorCritic(nn.Module):
    def __init__(self,action_dim,training=True):
        super(ActorCritic, self).__init__()
        # 处理observation的两个卷积块
        self.conv1 = nn.Conv2d(4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1, padding=0)

        # 处理goal_position的全连接层
        self.fc_goal = nn.Linear(3, GOAL_REPR_SIZE)
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)

        self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)
        self.policy_linear = nn.Linear(RNN_SIZE, action_dim)
        self.value_linear = nn.Linear(RNN_SIZE,1)
        self.blocking_linear = nn.Linear(RNN_SIZE,1)
        self.ongoal_linear = nn.Linear(RNN_SIZE,1)

        self.train = training
    '''
    这里的inputs应该是(batch_size,grid_size,grid_size,channels)
    goal_pos的size是(3,1),3是因为(x,y,magitude) 或者（1，3））
    '''
    def forward(self,inputs,goal_pos,hidden):

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        goal_layer = F.relu(self.fc_goal(goal_pos))
        hidden_input = torch.cat([x, goal_layer], 1)
        h1 = F.relu(self.fc1(hidden_input))

        if self.train:
            h1 = F.dropout(h1, p=KEEP_PROB1)
        h2 = self.fc2(h1)
        if self.train:
            h2 = F.dropout(h2, p=KEEP_PROB2)
        h3 = F.relu(h2 + hidden_input)

        hx, cx = self.lstm(h3, hidden)

        # 这里初始化隐藏状态
        # hx, cx = self.lstm(h3, self.init_hidden(1))
        rnn_out = hx

        policy = F.softmax(self.policy_linear(rnn_out), dim=1)
        value = self.value_linear(rnn_out)
        blocking = torch.sigmoid(self.blocking_linear(rnn_out))
        on_goal = torch.sigmoid(self.ongoal_linear(rnn_out))

        # valids? the writter called rhe policy_sig the valids and returned it.
        policy_sig = F.sigmoid(self.policy_linear(rnn_out),dim=1)

        return policy, value, (hx, cx), blocking, on_goal,policy_sig

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, RNN_SIZE), torch.zeros(batch_size, RNN_SIZE))



action_dim = 5
TRAINING = True
GRID_SIZE = 10
net = ActorCritic(action_dim, TRAINING)

inputs = torch.randn(1,4,GRID_SIZE, GRID_SIZE)
goal_pos = torch.randn(1, 3)
hidden = net.init_hidden(1)
policy, value, hidden_out, blocking, on_goal = net(inputs, goal_pos, hidden)

# class ACNet(nn.Module):
#     def __init__(self, a_size, TRAINING,GRID_SIZE):
#         super(ACNet, self).__init__()
#         self.conv1 = nn.Conv2d(4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
#         self.conv1a = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
#         self.conv1b = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
#         self.conv2a = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
#         self.conv2b = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1, padding=0)
#
#         self.fc_goal = nn.Linear(3, GOAL_REPR_SIZE)
#         self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
#         self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
#
#         self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)
#         self.fc_policy = nn.Linear(RNN_SIZE, a_size)
#         self.fc_value = nn.Linear(RNN_SIZE, 1)
#         self.fc_blocking = nn.Linear(RNN_SIZE, 1)
#         self.fc_on_goal = nn.Linear(RNN_SIZE, 1)
#
#         self.train = TRAINING
#
#     def forward(self, inputs, goal_pos, hidden):
#         x = F.relu(self.conv1(inputs))
#         x = F.relu(self.conv1a(x))
#         x = F.relu(self.conv1b(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2a(x))
#         x = F.relu(self.conv2b(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = torch.flatten(x, start_dim=1)
#
#         goal_layer = F.relu(self.fc_goal(goal_pos))
#         hidden_input = torch.cat([x, goal_layer], 1)
#         h1 = F.relu(self.fc1(hidden_input))
#         if self.train:
#             h1 = F.dropout(h1, p=KEEP_PROB1)
#         h2 = self.fc2(h1)
#         if self.train:
#             h2 = F.dropout(h2, p=KEEP_PROB2)
#         h3 = F.relu(h2 + hidden_input)
#
#         hx, cx = self.lstm(h3, hidden)
#         rnn_out = hx
#
#         policy = F.softmax(self.fc_policy(rnn_out), dim=1)
#         value = self.fc_value(rnn_out)
#         blocking = torch.sigmoid(self.fc_blocking(rnn_out))
#         on_goal = torch.sigmoid(self.fc_on_goal(rnn_out))
#
#         return policy, value, (hx, cx), blocking, on_goal
#
#     def init_hidden(self, batch_size):
#         return (torch.zeros(batch_size, RNN_SIZE), torch.zeros(batch_size, RNN_SIZE))


# Example usage

