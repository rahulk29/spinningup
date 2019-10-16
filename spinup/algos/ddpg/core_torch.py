import numpy as np
import torch
import torch.nn as nn

activations = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'softmax': nn.Softmax,
    'leakyrelu': nn.LeakyReLU,
    None: None
}


class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes=(32,), activation='tanh',
                 output_activation=None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        self.layers = nn.ModuleList([nn.Linear(in_features=in_features,
                                               out_features=hidden_sizes[0])])
        for i, h in enumerate(hidden_sizes[1:]):
            self.layers.append(activations[activation]())
            self.layers.append(nn.Linear(in_features=hidden_sizes[i],
                                         out_features=hidden_sizes[i + 1]))
        if output_activation is not None:
            self.layers.append(activations[output_activation]())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_scale * x
        return x.squeeze() if self.output_squeeze else x


"""
Actor-Critics
"""


class ActorCritic(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_sizes=(400, 300),
                 activation='relu',
                 output_activation='tanh',
                 action_space=None):
        super(ActorCritic, self).__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.policy = MLP(in_features=in_features,
                          hidden_sizes=list(hidden_sizes) + [act_dim],
                          activation=activation,
                          output_activation=output_activation,
                          output_scale=act_limit)
        self.q = MLP(in_features=in_features + act_dim,
                     hidden_sizes=list(hidden_sizes) + [1],
                     activation=activation,
                     output_activation=None,
                     output_squeeze=True)

    def forward(self, x, a):
        pi = self.policy(x)
        q = self.q(torch.cat((x, a), dim=1))
        q_pi = self.q(torch.cat((x, pi), dim=1))
        return pi, q, q_pi
