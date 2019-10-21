import numpy as np
import torch
import torch.nn as nn
from spinup.algos.rdpg.util import *

activations = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'softmax': nn.Softmax,
    'leakyrelu': nn.LeakyReLU,
    None: None
}


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, init_w=3e-3, activation='relu', output_activation='tanh'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 20)
        self.fc2 = nn.Linear(20, 50)
        self.lstm = nn.LSTMCell(50, 50)
        self.fc3 = nn.Linear(50, act_dim)
        self.activation = activations[activation]()
        self.output_activation = activations[output_activation]()
        self.init_weights(init_w)

        # self.cx = torch.zeros((1, 50), requires_grad=True)
        # self.hx = torch.zeros((1, 50), requires_grad=True)
        self.cx = Variable(torch.zeros(1, 50))
        self.hx = Variable(torch.zeros(1, 50))

    def init_weights(self, init_w):
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def reset_lstm_hidden_state(self, done=True):
        # if done:
        #     self.cx = torch.zeros((1, 50), requires_grad=True)
        #     self.hx = torch.zeros((1, 50), requires_grad=True)
        # else:
        #     self.cx = self.cx.detach()
        #     self.hx = self.hx.detach()
        if done == True:
            self.cx = Variable(torch.zeros(1, 50))
            self.hx = Variable(torch.zeros(1, 50))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

    def forward(self, x, hidden_states=None):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        if hidden_states is None:
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.lstm(x, hidden_states)

        x = hx
        x = self.output_activation(self.fc3(x))
        return x, (hx, cx)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, init_w=3e-3, activation='relu'):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 20)
        self.fc2 = nn.Linear(20 + act_dim, 50)
        self.fc3 = nn.Linear(50, 1)
        self.activation = activations[activation]()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(torch.cat((out, a), dim=1)))
        out = self.fc3(out)
        return out
