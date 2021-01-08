import random
from collections import namedtuple
import os
import logging
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# class ReplayMemory(object):
#     def __init__(self, config, state_dim):
#         self.memory = []
#         self.memory_size = config.memory_size
#         self.batch_size = config.batch_size
#         self.memory_counter = 0
#         self.dim = state_dim
#
#     def add(self, *args):
#         if len(self.memory) < self.memory_size:
#             self.memory.append(None)
#         self.memory[self.memory_counter] = Transition(*args)
#         self.memory_counter = (self.memory_counter + 1) % self.memory_size
#
#     def sample(self):
#         # sample_index = np.random.choice(self.memory_size, size=self.batch_size)
#         # return self.memory[sample_index, :]
#         return random.sample(self.memory, self.batch_size)
#     # def __len__(self):
#     #     return len(self.memory)
#

class ReplayMemory:
    def __init__(self, config, state_dim):
        self.memory_size = config.memory_size
        self.states = np.empty((self.memory_size, state_dim), dtype=np.float32)
        self.actions = np.empty(self.memory_size, dtype=np.int64)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.done = np.empty(self.memory_size, dtype=np.bool)
        # without initializing entries.
        self.history_length = config.history_length
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, state_dim), dtype=np.float32)
        self.poststates = np.empty((self.batch_size, self.history_length, state_dim), dtype=np.float32)

    def add(self, state, action, reward, done):
        # Return a new array of given shape and type,
        # without initializing entries.

        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.done[self.current] = done
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.done[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        action = self.actions[indexes]
        reward = self.rewards[indexes]
        done = self.done[indexes]
        return np.transpose(self.prestates, (0, 2, 1)), action, reward, np.transpose(self.poststates, (0, 2, 1)), done