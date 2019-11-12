import numpy as np
from collections import namedtuple
import random

Experience = namedtuple('Experience', 'obs1 act rew obs2 done')

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.max_p = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # reach leave
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

        self.max_p = max(self.max_p, p)

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
        self.max_p = max(self.max_p, p)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, size):
        self.tree = SumTree(size)
        self.capacity = size

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def store(self, obs, act, rew, next_obs, done):
        self.tree.add(max(self.tree.max_p, 1e-5), Experience(obs1=obs, act=act, rew=rew, obs2=next_obs, done=done))

    def sample_batch(self, batch_size=32):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.asarray(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        # we first get the data out of the sampled experience
        obs1 = np.stack(tuple(experience.obs1 for experience in batch))
        act = np.stack(tuple(experience.act for experience in batch))
        done = np.stack(tuple(experience.done for experience in batch))
        rew = np.stack(tuple(experience.rew for experience in batch))
        obs2 = np.stack(tuple(experience.obs2 for experience in batch))

        # batch = Experience(*zip(*batch))

        return (obs1, act, done, rew, obs2), idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
