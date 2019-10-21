from collections import deque, namedtuple
import numpy as np
import random

Experience = namedtuple('Experience', 'obs1 act rew obs2 done')


class RingBuffer:
    def __init__(self, max_size):
        # Max number of transitions possible will be the memory capacity, could be much less
        self.size = 0
        self.max_size = max_size
        self.ptr = 0
        self.buf = []

    def append(self, item):
        if self.size < self.max_size:
            self.buf.append(None)
        self.buf[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __len__(self):
        return self.size


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class ReplayBuffer:
    def __init__(self, capacity, max_episode_length):
        self.max_episode_length = max_episode_length
        self.num_episodes = capacity // max_episode_length
        self.memory = RingBuffer(self.num_episodes)
        self.trajectory = []  # Temporal list of episode

    def sample(self, batch_size, maxlen=None):
        batch = [self.sample_trajectory(maxlen=maxlen) for _ in range(batch_size)]
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def sample_trajectory(self, maxlen=None):
        e = random.randrange(len(self.memory))
        mem = self.memory[e]
        T = len(mem)
        if T > 0:
            # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
            if maxlen is not None and T > maxlen + 1:
                t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
                return mem[t:t + maxlen + 1]
            else:
                return mem

    def append(self, obs1, act, rew, done, training=True):
        self.trajectory.append(Experience(obs1=obs1, act=act, rew=rew, obs2=None, done=done))
        if done or len(self.trajectory) >= self.max_episode_length:
            self.memory.append(self.trajectory)
            self.trajectory = []

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or current_terminal:
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def __len__(self):
        return sum(len(self.memory[idx]) for idx in range(len(self.memory)))
