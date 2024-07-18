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

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(weights, std=1.0):
    out = np.random.randn(*weights.shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return torch.tensor(out)

class ACNet(nn.Module):
    def __init__(self, a_size, TRAINING, GRID_SIZE):
        super(ACNet, self).__init__()
        self.conv1 = nn.Conv2d(4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1, padding=0)

        self.fc_goal = nn.Linear(3, GOAL_REPR_SIZE)
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)

        self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)
        self.fc_policy = nn.Linear(RNN_SIZE, a_size)
        self.fc_value = nn.Linear(RNN_SIZE, 1)
        self.fc_blocking = nn.Linear(RNN_SIZE, 1)
        self.fc_on_goal = nn.Linear(RNN_SIZE, 1)

        self.train = TRAINING

    def forward(self, inputs, goal_pos, hidden):
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
        rnn_out = hx

        policy = F.softmax(self.fc_policy(rnn_out), dim=1)
        value = self.fc_value(rnn_out)
        blocking = torch.sigmoid(self.fc_blocking(rnn_out))
        on_goal = torch.sigmoid(self.fc_on_goal(rnn_out))

        return policy, value, (hx, cx), blocking, on_goal

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, RNN_SIZE), torch.zeros(batch_size, RNN_SIZE))

# Example usage
a_size = 5
TRAINING = True
GRID_SIZE = 8
net = ACNet(a_size, TRAINING, GRID_SIZE)
inputs = torch.randn(1, 4, GRID_SIZE, GRID_SIZE)
goal_pos = torch.randn(1, 3)
hidden = net.init_hidden(1)
policy, value, hidden_out, blocking, on_goal = net(inputs, goal_pos, hidden)
