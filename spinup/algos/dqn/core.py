import numpy as np
import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Actor-Critics
"""


def mlp_action_value(x, act_dim, hidden_sizes=(400, 300), activation=tf.nn.relu):
    with tf.variable_scope('q'):
        q = mlp(x, list(hidden_sizes) + [act_dim], activation, None)
    return q
