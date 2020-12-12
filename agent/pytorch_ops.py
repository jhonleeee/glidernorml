import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as d
import numpy as np



class myRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0.01)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0.01] = 0
        return grad_input


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_p):
        super(PolicyNet, self).__init__()
        # state dim and action dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # drop out probability
        self.dropout_prob = dropout_p

        # policy network
        self.l1 = nn.Linear(self.state_dim, 128, bias=False)
        self.l2 = nn.Linear(128, 512, bias=False)
        self.l3 = nn.Linear(512, 128, bias=False)
        self.l4 = nn.Linear(128, self.action_dim, bias=False)

    def forward(self, x):
        x = self.l1(x)
        # x = F.dropout(x,p=self.dropout_prob)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = F.softmax(x)
        return x


class DQN(nn.Module):
    def __init__(self, state_dim, in_channels, action_dim, activation_fn=F.relu):
        super(DQN, self).__init__()
        self.input_channels = in_channels
        self.num_actions = action_dim
        self.activation_fn = activation_fn

        # pytorch only support NCHW, so may need transpose
        self.conv1 = nn.Conv1d(in_channels=self.input_channels,
                               out_channels=32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(64)

        # flatten
        def flatten_size(insize, kernel_size=2, stride=1, padding=0, dilation=1):
            return (insize + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        size = flatten_size(state_dim)
        size = flatten_size(size, kernel_size=1, stride=1)

        self.l1 = nn.Linear(in_features=size*64, out_features=512, bias=True)
        self.l2 = nn.Linear(in_features=512, out_features=self.num_actions, bias=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.activation_fn(self.bn1(self.conv1(x)))
        x = self.activation_fn(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.l1(x))
        x = self.l2(x)
        return x

    def feature_size(self, input):
        x = torch.zeros(1, input, input)
        return self.l2(self.l1(x)).view(1, -1).size(1)


class EntropyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EntropyNet, self).__init__()
        self.h_size = 512
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1 = nn.Linear(self.state_dim, self.h_size)
        self.l2 = nn.Linear(self.h_size, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
