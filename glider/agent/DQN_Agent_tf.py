# -*- coding: utf-8 -*-
import numpy as np
from os import path
import sys
from project_root import DIR
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_ops import clipped_error
from helpers.helpers import class_vars
from replay_memory import ReplayMemory
from history import History
from tf_ops import conv1d, linear

class DQN(object):
    def __init__(self, config, state_dim, action_dim, sess):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

        self.build_network()

        self.memory = ReplayMemory(self.config, self.state_dim)
        self.history = History(self.config, self.state_dim)

        self.sess = sess
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.learn_step_counter = 0
        self.ep = self.ep_max

        if self.train:
            self.loss_file = open(path.join(DIR, 'results', 'loss'), 'w', 0)
            self.reward_file = open(path.join(DIR, 'results', 'reward'), 'w', 0)
        else:
            self.load_model()
    def init_history(self, state):
        for _ in range(self.history_length):
            self.history.add(state)

    def build_network(self):
        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        self.w = {}
        self.t_w = {}
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        with tf.variable_scope('predict_net'):
            # 建立现实神经网络
            self.state = tf.placeholder(tf.float32, [None, self.state_dim,self.history_length], name='state')

            # e_layer1 = tf.layers.dense(self.state, self.layer1_elmts, tf.nn.relu,
            #                            kernel_initializer=w_initializer,
            #                            bias_initializer=b_initializer, name='e_layer1')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv1d(self.state,
                                                             32, 2, 1, initializer, activation_fn,
                                                             self.cnn_format, name='l1')
            shape = self.l1.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l1, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q_predict, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_dim, name='q_predict')
            # self.q_predict = tf.layers.dense(self.l4, self.action_dim, kernel_initializer=w_initializer,
            #                               bias_initializer=b_initializer, name='q_predict')

            # self.q_predict = tf.reshape(self.q_predict, (-1, self.action_dim))
            self.q_action = tf.argmax(self.q_predict, axis=1)
        with tf.variable_scope('target_net'):
            self.target_state = tf.placeholder(tf.float32, [None, self.state_dim, self.history_length], name='target_state')
            # t_layer1 = tf.layers.dense(self.next_state, self.layer1_elmts, tf.nn.relu,
            #                            kernel_initializer=w_initializer,
            #                            bias_initializer=b_initializer, name='t_layer1')
            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv1d(self.target_state,
                                                                        32, 2, 1, initializer, activation_fn,
                                                                        self.cnn_format, name='target_l1')
            # self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv1d(self.target_l1,
            #                                                             64, [4, 4], [2, 2], initializer, activation_fn,
            #                                                             self.cnn_format, name='target_l2')
            # self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv1d(self.target_l2,
            #                                                             64, [3, 3], [1, 1], initializer, activation_fn,
            #                                                             self.cnn_format, name='target_l3')

            shape = self.target_l1.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l1, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
            self.q_target, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.action_dim, name='target_q')
            # self.q_next = tf.layers.dense(t_layer1, self.action_dim, kernel_initializer=w_initializer,
            #                                 bias_initializer=b_initializer, name='q_next')
            # self.q_next = tf.reshape(self.q_next, (-1, self.action_dim))
            # self.q_next_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            # self.q_next_with_idx = tf.gather_nd(self.target_q, self.q_next_idx)
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.q_target, self.target_q_idx)
        # update parameters of target network
        with tf.variable_scope('optimizer'):
            self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
            # one hot action
            self.action = tf.placeholder(tf.int64, [None], name='action')
            action_one_hot = tf.one_hot(self.action, self.action_dim, 1.0, 0.0, name='action_one_hot')
            acted_q = tf.reduce_sum(self.q_predict * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.target_q - acted_q
            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_predict))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('pred_to_target'):
            # self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict_net')
            # self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            # self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    def observe(self, state, action, reward, done):
        self.history.add(state)
        self.memory.add(state, action, reward, done)

        # fetch batch to train from memory
        if self.learn_step_counter >= self.learn_start:
            if self.learn_step_counter % self.train_frequency == 0:
                self.q_learning_mini_batch()

        # update target network cloning predict net
        if self.learn_step_counter % self.target_q_update_step == 0:
            self.update_target_q_network()

        self.learn_step_counter += 1

    def get_action(self, s_t):
        # expand one dimension
        s_t = s_t[np.newaxis, :]
        self.ep = max(self.ep_min, self.ep_max - (self.ep_max - self.ep_min) * self.learn_step_counter / self.explore_steps)

        if np.random.rand() < self.ep:
            action = np.random.randint(0, self.action_dim)
        else:
            action = np.argmax(self.sess.run(self.q_action, feed_dict={self.state: s_t}))
        return action

    def update_target_q_network(self):
        # with self.sess:
            for name in self.w.keys():
                # self.sess.run([self.t_w_assign_op[name]], feed_dict={self.t_w_input[name]: self.w[name]})
                self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def q_learning_mini_batch(self):
        # Transition = self.memory.sample()
        # state, action, reward, next_state, done = zip(*Transition)
        state, action, reward, next_state, done = self.memory.sample()
        # state = np.array(state)
        # action = np.array(action)
        # reward = np.array(reward)[:, np.newaxis]
        # next_state = np.array(next_state)
        # done = (np.array(done) + 0.)[:, np.newaxis]
        done = np.array(done) + 0.

        q_t_plus_1 = self.sess.run(self.q_target, feed_dict={self.target_state: next_state})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q = reward + (1.0 - done) * self.discount * max_q_t_plus_1

        _, loss = self.sess.run([self.optim, self.loss], feed_dict={
            self.target_q: target_q,
            self.action: action,
            self.state: state})

        # self.ep = max(self.ep_min, self.ep_max - (self.ep_max - self.ep_min) * self.learn_step_counter / self.explore_steps)

        if self.train:
            if self.learn_step_counter % self.scale == 0:
                self.loss_file.write("%d, %.4f\n" % (self.learn_step_counter, loss))

    def save_model(self):
        if self.learn_step_counter % self.save_model_step == 0:
            n = self.learn_step_counter/self.save_model_step
            save_path = path.join(DIR, 'models', 'model-' + str(n) + '.pkl')
            self.saver.save(self.sess, save_path)

    def load_model(self):
        sys.stderr.write("Test Phase: loading model........\n")
        model_path = path.join(DIR, 'models', 'model-xxx.pkl')
        self.saver.restore(self.sess, model_path)

