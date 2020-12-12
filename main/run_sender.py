#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from os import path
from project_root import DIR
import argparse
from env.sender import Sender
from config import AgentConfig, EnvironmentConfig
import agent.norml
import matplotlib.pyplot as plt
from agent.norml.norml_Agent_tf import Norml
FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'RL_PENDULUM_GYM_CONFIG_META',
                    'Configuration for training.')
def run(agent_type, port,norml_type):
    if agent_type == 'torch_dqn':
        from agent.DQN_Agent_torch import DQN_Agent as DQN
        env = Sender(env_config, port)
        agent = DQN(agent_config, env.state_dim, env.action_dim)
        if agent_config.train:
            train_sender(agent, env, agent_config)
        else:
            run_sender(agent, env, agent_config)
    elif agent_config == 'tf_dqn':# tensorflow agent
        import tensorflow as tf
        from agent.DQN_Agent_tf import DQN    
        with tf.Session() as sess:
            env = Sender(env_config, args.port)
            agent = DQN(agent_config, env.state_dim, env.action_dim, sess)
            if agent_config.train:
                train_sender(agent, env, agent_config)
            else:
                run_sender(agent, env, agent_config)
    elif agent_type == 'sarsa':
        from agent.Sarsa_Agent import Sarsa as DQN
        env = Sender(env_config, args.port)
        agent = DQN(agent_config, env.state_dim, env.action_dim)
        if agent_config.train:
            train_sender(agent, env, agent_config)
        else:
            run_sender(agent, env, agent_config)
    elif agent_type == 'norml':
        import tensorflow as tf
        with tf.Session() as sess:
        env = Sender(env_config, args.port)
        agent = Norml(norml_type)
        train_norml(sess,env,agent)

def run_sender(agent, env, agent_config):
    if agent_config.debug:
        perf_file = open(path.join(DIR, 'results', 'perf'), 'w', 0)

    try:
        env.handshake()
        state = env.reset()
        agent.init_history(state)
        while True:
            # predict
            # use history
            agent.history.add(state)
            action = agent.get_action(agent.history.get())
            # act
            next_state, reward, done, tput = env.step(action)
            state = next_state

            if agent_config.debug:
                perf_file.write("state: ")
                for state in np.array(state):
                    perf_file.write('%.4f\t' % state)
                perf_file.write("action: %d\t reward: %.4f\n" % (action, reward))
                perf_file.write("tput: %.4f\t" % tput)
                perf_file.write("\n")
    except KeyboardInterrupt:
        pass
    finally:
        env.cleanup()


def train_sender(agent, env, agent_config):#NOTE:traing agent
    if agent_config.debug:
        episode_file = open(path.join(DIR, 'results', 'episode'), 'w', 0)
    steps = 0
    try:
        env.handshake()
    try:
        env.handshake()
        for episode in range(agent_config.episode):
            ep_steps = 0
            ep_reward = 0
            state = env.reset()#NOTE:reset sender
            agent.init_history(state)#NOTE:fill history with inital env
            while True:
                # predict
                action = agent.get_action(agent.history.get())
                # act
                n_state, reward, done, tput = env.step(action)
                # observe
                agent.observe(n_state, action, reward, done)
                ep_reward += reward
                if agent_config.debug:
                    debug_file = open(path.join(DIR, 'results', 'debug' + str(steps / (10*agent_config.scale))), 'a', 0)
                    debug_file.write("【episode %d, step %d】\n" % (episode, ep_steps))
                    debug_file.write("state: ")
                    for state in np.array(state):
                        debug_file.write('%.4f\t' % state)
                    debug_file.write("action: %d\t reward: %.4f\n" % (action, reward))
                    debug_file.write("tput: %.4f\t" % tput)
                    debug_file.write("\n")

                # step end next episode
                if done:
                    # env.compute_performance()
                    print('episode: ', episode, 'ep_steps: ', ep_steps, 'steps: ', steps, 'ep_reward: ', round(ep_reward, 2))
                    episode_file.write("[episode %d, steps %d] ep_reward: %.4f, tput: %.4f\n" % (episode, ep_steps, ep_reward, tput))
                    break
                ep_steps += 1
                steps += 1
                state = n_state.copy()

    except KeyboardInterrupt:
        pass
    finally:
        env.cleanup()


def draw_loss(episode, history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend('Train', loc='upper left')
    plt.savefig('episode'+str(episode)+'png')

if __name__ == '__main__':
    del argv  # Unused
    config = DotMap(getattr(config_maml, FLAGS.config))
    print('MAML config: %s' % config)
    tf.logging.info('MAML config: %s', config)
    algo = agent.norml.maml_rl.MAMLReinforcementLearning(config)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    agent_config = AgentConfig
    env_config = EnvironmentConfig

    run(agent_type='tf_norml', port=args.port,config)

def train_norml(self,session,env,agentt,num_outer_iterations=1,dont_update_weights=False,ignore_termination=False):
    """Performs one or multiple training steps.

    Per task: rollout train samples, rollout test samples
    Outer: computer outer loss gradient update

    Args:
      session: TF session.
      num_outer_iterations: Number of outer loop steps (gradient steps).
      dont_update_weights: Run the algorithm, but don't update any parameters.
      ignore_termination: Ignore early termination and max iterations condition.

    Returns:
      Objective value after optimization last step.

    Raises:
      ValueError: if the loss is NaN
    """
    inner_tasks = random.sample(agent.mamlrl.task_env_modifiers, agent.mamlrl.tasks_batch_size)#add modifier to task baths
    done = False
    avg_test_reward = np.NaN#test reward
    for step in range(agent.mamlrl.current_step,#from 0,outter loop itr
                      agent.mamlrl.current_step + num_outer_iterations):
      if ignore_termination:
        done = False
      elif agent.mamlrl.current_step >= num_outer_iterations:
        done = True
      elif agent.mamlrl.early_termination is not None:
        done = agent.mamlrl.early_termination(agent.marl.avg_test_reward_log)
      if done:
        break
      agent.mamlrl.current_step = step + 1
      tf.logging.info('iteration: %d', agent.mamlrl.current_step)
      print('iteration: %d' % agent.mamlrl.current_step)
      if not self.fixed_tasks:
        inner_tasks = random.sample(self.task_env_modifiers,
                                    self.tasks_batch_size)

      # If we do rollouts locally, don't parallelize inner train loops
      results = []
      for task_idx in range(self.tasks_batch_size):
        results.append(self.train_inner((session, inner_tasks, task_idx)))

      samples = {}
      avg_train_reward = 0.
      avg_test_reward = 0.

      for task_idx in range(self.tasks_batch_size):
        # training rollouts
        train_rollouts, train_reward, test_rollouts, test_reward = results[
            task_idx]
        avg_train_reward += train_reward
        avg_test_reward += test_reward

        samples[self.inner_train_inputs[task_idx]] = train_rollouts['states']
        samples[self.inner_train_next_inputs[task_idx]] = train_rollouts[
            'next_states']
        samples[self.inner_train_actions[task_idx]] = train_rollouts['actions']
        if not self.learn_advantage_function_inner:
          samples[self.inner_train_advantages[task_idx]] = train_rollouts[
              'advantages']
        samples[self.inner_test_inputs[task_idx]] = test_rollouts['states']
        samples[self.inner_test_actions[task_idx]] = test_rollouts['actions']
        samples[
            self.inner_test_advantages[task_idx]] = test_rollouts['advantages']

      # Normalize advantage for easier parameter tuning
      samples[self.inner_test_advantages[task_idx]] -= np.mean(
          samples[self.inner_test_advantages[task_idx]])
      samples[self.inner_test_advantages[task_idx]] /= np.std(
          samples[self.inner_test_advantages[task_idx]])

      avg_test_reward /= self.tasks_batch_size
      avg_train_reward /= self.tasks_batch_size
      self.avg_test_reward_log.append(avg_test_reward)

      if not dont_update_weights:
        if self.outer_lr_decay:
          samples[self.outer_lr_ph] = self.outer_lr_init * (
              1. - float(step) / self.num_outer_iterations)
        session.run(self.apply_grads_outer, samples)
      print('avg train reward: %f' % avg_train_reward)
      print('avg test reward: %f' % avg_test_reward)
      tf.logging.info('avg train reward: %f', avg_train_reward)
      tf.logging.info('avg test reward: %f', avg_test_reward)
      samples[self.avg_test_reward] = avg_test_reward
      samples[self.avg_train_reward] = avg_train_reward
      eval_summaries = session.run(self.summaries, samples)
      self.writer.add_summary(eval_summaries, self.current_step)

      if self.reporter is not None:
        eval_global_loss = session.run(self.global_loss, samples)
        if np.isnan(eval_global_loss):
          print('Loss is NaN')
          tf.logging.info('Loss is NaN')
          raise ValueError('Loss is NaN')
        else:
          self.reporter(self.current_step, avg_test_reward)

      if step % self.log_every == 0:
        print('Saving (%d) to: %s' % (self.current_step, self.tensorboard_dir))
        tf.logging.info('Saving (%d) to: %s', self.current_step,
                        self.tensorboard_dir)
        self._save_variables(session)

    return done, avg_test_reward
