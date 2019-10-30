import torch
import torch.nn as nn
from copy import deepcopy

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
            self.layers.append(nn.LayerNorm((hidden_sizes[i])))
            self.layers.append(activations[activation]())
            self.layers.append(nn.Linear(in_features=hidden_sizes[i],
                                         out_features=hidden_sizes[i + 1]))
        if output_activation is not None:
            self.layers.append(nn.LayerNorm((hidden_sizes[-1])))
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


class ParameterNoise(object):

    def __init__(self, actor, param_noise_stddev=0.1, desired_action_stddev=0.1, adaption_coefficient=1.01):

        self.actor = actor
        self.perturbed_actor = deepcopy(self.actor)
        self.param_noise_stddev = param_noise_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaption_coefficient = adaption_coefficient
        self.set_perturbed_actor_updates()

    def get_perturbable_parameters(self, model):
        # Removing parameters that don't require parameter noise
        parameters = []
        for name, params in model.named_parameters():
            parameters.append(params)

        return parameters

    def set_perturbed_actor_updates(self):
        """
        Update the perturbed actor parameters
        :return:
        """
        # actor_perturbable_parameters = self.get_perturbable_parameters(self.actor)
        # perturbed_actor_perturbable_parameters = self.get_perturbable_parameters(self.perturbed_actor)

        for params, perturbed_params in zip(self.actor.parameters(), self.perturbed_actor.parameters()):
            # Update the parameters
            perturbed_params.data.copy_(params
                                        # + torch.normal(mean=torch.zeros(params.shape), std=self.param_noise_stddev)
                                        )

    def adapt_param_noise(self, obs):
        if self.param_noise_stddev is None:
            return 0.
        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        self.set_perturbed_actor_updates()
        with torch.no_grad():
            adaptive_noise_distance = torch.pow(torch.mean(torch.pow(self.actor(obs) - self.perturbed_actor(obs), 2)),
                                                0.5)
        if adaptive_noise_distance > self.desired_action_stddev:
            # Decrease stddev.
            self.param_noise_stddev /= self.adaption_coefficient
        else:
            # Increase stddev.
            self.param_noise_stddev *= self.adaption_coefficient
