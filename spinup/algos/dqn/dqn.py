import time

import gym
import numpy as np
import tensorflow as tf
from spinup.algos.dqn import core
from spinup.algos.dqn.core import get_vars, mlp_action_value
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def dqn(env_fn, action_value=core.mlp_action_value, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        q_lr=1e-3, batch_size=100, start_steps=10000, update_period=10,
        eps_start=1, eps_end=0.1, eps_step=1e-4, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Inputs to computation graph
    x_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, obs_dim, None, None)
    a_ph = tf.placeholder(tf.int32)

    with tf.variable_scope('main'):
        q = mlp_action_value(x_ph, act_dim, **ac_kwargs)

    with tf.variable_scope('target'):
        q_targ = mlp_action_value(x2_ph, act_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main'])
    print('\nNumber of parameters: %d\n' % var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * tf.reduce_max(q_targ, axis=1))

    loss = tf.reduce_mean((tf.reduce_sum(tf.one_hot(a_ph, act_dim) * q, 1) - backup) ** 2)

    # Separate train ops for q
    optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_q_op = optimizer.minimize(loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'q': q})

    def get_action(o, eps):
        if np.random.random() < eps:
            return np.squeeze(sess.run(tf.argmax(q, axis=1), feed_dict={x_ph: np.expand_dims(o, 0)}))
        else:
            return env.action_space.sample()

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch

    for t in range(total_steps):
        eps = max(eps_start - t * eps_step, eps_end)
        if t > 1:
            a = get_action(o, eps)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store(o, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DQN updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             a_ph: a
                             }

                # Q-learning update
                outs = sess.run([loss, q, train_q_op], feed_dict)
                logger.store(Loss=outs[0], QVals=outs[1])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t % update_period == 0:
            sess.run([target_update])

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--c', type=str, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--eps_start', type=str, default=1)
    parser.add_argument('--eps_end', type=str, default=0.1)
    parser.add_argument('--eps_step', type=str, default=1e-4)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda: gym.make(args.env), action_value=core.mlp_action_value,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, update_period=args.c, eps_start=args.eps_start,
        eps_end=args.eps_end, eps_step=args.eps_step)
