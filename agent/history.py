import numpy as np

class History:
    def __init__(self, config, state_dim):
        self.history = np.zeros([config.history_length, state_dim], dtype=np.float32)

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def reset(self):
        self.history *= 0

    def get(self):
        return np.transpose(self.history, (1, 0))