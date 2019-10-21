import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
from copy import deepcopy
from spinup.algos.rdpg.memory import ReplayBuffer
from spinup.algos.rdpg.model import Actor, Critic, soft_update
from spinup.utils.logx_torch import EpochLogger
from torch.autograd import Variable


def rdpg(env_fn, ac_kwargs=dict(), seed=0, visualize=False, trajectory_length=5,
         steps_per_epoch=5000, epochs=100, replay_size=int(1e7), gamma=0.99,
         polyak=0.999, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         act_noise=0.1, max_ep_len=500, logger_kwargs=dict(), save_freq=1,
         epsilon_decay=2e-5):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and act, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        visualize (bool): Whether visualize during training

        trajectory_length (int): Length of trajectory (T in paper)

        steps_per_epoch (int): Number of steps of interaction (state-act pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random act selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        epsilon_decay (int): Decay of epsilon per step.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about act space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main outputs from computation graph
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic = Critic(obs_dim=obs_dim, act_dim=act_dim).to(device)

    # Target networks
    actor_target = Actor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic_target = Critic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    actor_target.eval()
    critic_target.eval()

    # Experience buffer
    replay_buffer = ReplayBuffer(capacity=replay_size, max_episode_length=max_ep_len)

    # Count variables
    var_counts = tuple(sum([np.prod(param.size()) for param in network.parameters()])
                       for network in [actor, critic])
    print(
        '\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % (var_counts[0], var_counts[1], sum(var_counts)))

    # Separate train ops for pi, q
    actor_optimizer = torch.optim.Adam(params=actor.parameters(), lr=pi_lr)
    critic_optimizer = torch.optim.Adam(params=critic.parameters(), lr=q_lr)

    # Initialize target network parameters
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    epsilon = 1

    def get_action(o, noise_scale):
        with torch.no_grad():
            pi, _ = actor(torch.Tensor(o[None, :]))
            a = pi.cpu().numpy()[0] + epsilon * noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def update_policy():
        actor.train()
        critic.train()

        experiences = replay_buffer.sample(batch_size)
        if len(experiences) == 0:  # not enough samples
            return

        pi_loss_total = 0
        q_loss_total = 0
        q_vals = []

        target_cx = Variable(torch.zeros(batch_size, 50))
        target_hx = Variable(torch.zeros(batch_size, 50))

        cx = Variable(torch.zeros(batch_size, 50))
        hx = Variable(torch.zeros(batch_size, 50))

        for t in range(len(experiences) - 1):  # iterate over episodes
            # target_cx = torch.zeros(batch_size, 50, requires_grad=True)
            # target_hx = torch.zeros(batch_size, 50, requires_grad=True)
            #
            # cx = torch.zeros(batch_size, 50, requires_grad=True)
            # hx = torch.zeros(batch_size, 50, requires_grad=True)

            target_cx = Variable(torch.zeros(batch_size, 50))
            target_hx = Variable(torch.zeros(batch_size, 50))

            cx = Variable(torch.zeros(batch_size, 50))
            hx = Variable(torch.zeros(batch_size, 50))

            # we first get the data out of the sampled experience
            obs1 = np.stack(tuple(trajectory.obs1 for trajectory in experiences[t]))
            # act = np.expand_dims(np.stack((trajectory.act for trajectory in experiences[t])), axis=1)
            act = np.stack(tuple(trajectory.act for trajectory in experiences[t]))
            done = np.expand_dims(np.stack(tuple(trajectory.done for trajectory in experiences[t])), axis=1)
            rew = np.expand_dims(np.stack(tuple(trajectory.rew for trajectory in experiences[t])), axis=1)
            # rew = np.stack((trajectory.rew for trajectory in experiences[t]))
            obs2 = np.stack(tuple(trajectory.obs1 for trajectory in experiences[t + 1]))

            with torch.no_grad():
                target_action, (target_hx, target_cx) = actor_target(torch.Tensor(obs2).to(device),
                                                                     (target_hx, target_cx))
                next_q_value = critic_target(torch.Tensor(obs2).to(device), target_action)
                target_q = torch.Tensor(rew).requires_grad_().to(device) + \
                           (1 - torch.Tensor(done).to(device)) * gamma * next_q_value

            # Critic update
            q = critic(torch.Tensor(obs1).to(device), torch.Tensor(act).to(device))
            q_vals.append(q.detach().cpu().numpy())

            # value_loss = criterion(q_batch, target_q_batch)
            q_loss = F.smooth_l1_loss(q, target_q)
            q_loss /= len(experiences)  # divide by trajectory length
            q_loss_total += q_loss

            # Actor update
            act, (hx, cx) = actor(torch.Tensor(obs1).to(device), (hx, cx))
            pi_loss = -critic(torch.Tensor(obs1).to(device), act)
            pi_loss /= len(experiences)  # divide by trajectory length
            pi_loss_total += pi_loss.mean()

            # update per trajectory
            critic.zero_grad()
            q_loss.backward()
            critic_optimizer.step()

            actor.zero_grad()
            pi_loss = pi_loss.mean()
            pi_loss.backward()
            actor_optimizer.step()

        logger.store(LossQ=q_loss_total, LossPi=pi_loss_total, QVals=np.asarray(q_vals))
        soft_update(actor_target, actor, polyak)
        soft_update(critic_target, critic, polyak)

    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0

            cx = Variable(torch.zeros(batch_size, 50))
            hx = Variable(torch.zeros(batch_size, 50))

            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                # a, (hx, cx) = actor(torch.Tensor(o).to(device), (hx, cx))
                # o, r, d, _ = test_env.step(a.detach().cpu().numpy())

                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
                if visualize:
                    env.render()
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    step = episode = trajectory_steps = 0
    o, ep_len, ep_ret = deepcopy(env.reset()), 0, 0
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    while step < total_steps:

        if step <= start_steps:
            a = env.action_space.sample()
        else:
            a = get_action(o, noise_scale=act_noise)
            epsilon -= epsilon_decay

        o2, r, d, _ = env.step(a)
        o2 = deepcopy(o2)

        if visualize:
            env.render()

        # agent observe and update policy
        replay_buffer.append(o, a, r, d)

        step += 1
        ep_len += 1
        trajectory_steps += 1
        ep_ret += r

        o = deepcopy(o2)

        if trajectory_steps >= trajectory_length:
            actor.reset_lstm_hidden_state(done=False)
            trajectory_steps = 0
            if step > start_steps:
                update_policy()

        if d or ep_len > max_ep_len:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = deepcopy(env.reset()), 0, False, 0, 0
            ep_len = 0
            ep_ret = 0

        if step > start_steps and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, {'actor': actor, 'critic': critic}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', step)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--trajectory_length', type=int, default=5)
    parser.add_argument('--epsilon_decay', type=float, default=2e-5)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    rdpg(
        lambda: gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        visualize=args.visualize,
        trajectory_length=args.trajectory_length,
        logger_kwargs=logger_kwargs)
