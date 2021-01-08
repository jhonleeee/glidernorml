# -*- coding: utf-8 -*-
import numpy as np
from os import path
import sys
from project_root import DIR
import tensorflow as tf
from replay_memory import ReplayMemory
from history import History
from keras.layers import Dense, Conv1D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential, load_model


class Sarsa(object):
    def __init__(self, config, state_dim, action_dim):
        # indicate the agent
        sys.stderr.write("Using SARSA agent\n")
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = self.build_network()

        self.loss_history = None
        self.memory = ReplayMemory(self.config, self.state_dim)
        self.history = History(self.config, self.state_dim)

        self.learn_step_counter = 0
        self.ep = self.config.ep_max

        if self.config.train:
            self.loss_file = open(path.join(DIR, 'results', 'loss'), 'w', 0)
            self.reward_file = open(path.join(DIR, 'results', 'reward'), 'w', 0)
        else:
            self.load_model(config.model_path)

    def init_history(self, state):
        for _ in range(self.config.history_length):
            self.history.add(state)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_network(self):
        initializer = tf.truncated_normal_initializer(0, 0.02)
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, strides=1, kernel_initializer=initializer, activation='relu',
                         padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.config.learning_rate))
        return model

    def observe(self, state, action, reward, done):
        self.memory.add(state, reward, action, done)

        # fetch batch to train from memory
        if self.learn_step_counter >= self.config.learn_start:
            if self.learn_step_counter % self.config.train_frequency == 0:
                self.q_learning_mini_batch()

        self.learn_step_counter += 1
        self.save_model()

    def get_action(self, s_t):
        # expand one dimension
        s_t = s_t[np.newaxis, :]
        self.ep = max(self.config.ep_min,
                      self.config.ep_max - (self.config.ep_max - self.config.ep_min) * self.learn_step_counter / self.config.explore_steps)
        if np.random.rand() < self.ep:
            action = np.random.randint(0, self.action_dim)
        else:
            values = self.model.predict(s_t)
            action = np.argmax(values)
        return action

    def q_learning_mini_batch(self):
        state, action, reward, next_state, done = self.memory.sample()
        next_action = []
        for i in range(next_state.shape[0]):
            next_action.append(self.get_action(next_state[i]))
        next_action = np.array(next_action)[:, np.newaxis]
        action = action[:, np.newaxis]
        target = self.model.predict(state)
        q_next = self.model.predict(next_state)
        raw = 0
        for (i, j) in zip(action, next_action):
            target[raw, i] = reward[raw] + self.config.discount * q_next[raw, j]
            raw += 1

        self.loss_history = self.model.fit(state, target, epochs=1, verbose=0)
        loss = self.loss_history.history['loss']

        if self.config.train:
            if self.learn_step_counter % self.config.scale == 0:
                self.loss_file.write("%d, %.4f\n" % (self.learn_step_counter, self.loss_history.history['loss'][0]))

    def save_model(self):
        if self.learn_step_counter % self.config.save_model_step == 0:
            n = self.learn_step_counter / self.config.save_model_step
            save_path = path.join(DIR, 'models', 'model-' + str(n) + '.pkl')
            self.model.save(save_path)

    def load_model(self, file_path):
        sys.stderr.write("Test Phase: loading model........\n")
        model_path = path.join(DIR, 'models-11.1', file_path)
        load_model(model_path)
