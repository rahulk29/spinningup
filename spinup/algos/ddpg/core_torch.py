import numpy as np
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
    def __init__(self, in_features, hidden_sizes=(32,), activation='tanh', output_activation=None, scale=1):
        super(MLP, self).__init__()
        self.scale = scale
        self.ml = nn.ModuleList([nn.Linear(in_features=in_features, out_features=hidden_sizes[0])])
        for i, h in enumerate(hidden_sizes[1:]):
            self.ml.append(activations[activation]())
            self.ml.append(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]))
        if output_activation is not None:
            self.ml.append(activations[output_activation]())

    def forward(self, x):
        for m in self.ml:
            x = m(x)
        x = self.scale * x
        return x


"""
Actor-Critics
"""


def mlp_actor_critic(obs_dim,
                     act_dim,
                     hidden_sizes=(400, 300),
                     activation='relu',
                     output_activation='tanh',
                     action_space=None):
    act_limit = action_space.high[0]
    mlp_pi = MLP(in_features=obs_dim,
                 hidden_sizes=list(hidden_sizes) + [act_dim],
                 activation=activation,
                 output_activation=output_activation,
                 scale=act_limit)
    mlp_q = MLP(in_features=obs_dim + act_dim,
                hidden_sizes=list(hidden_sizes) + [1],
                activation=activation,
                output_activation=None)
    return mlp_pi, mlp_q
