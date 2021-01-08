#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from os import path
from project_root import DIR
import argparse
from env.sender import Sender
from config import AgentConfig, EnvironmentConfig
import matplotlib.pyplot as plt

def run(agent_type, port):
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


def train_sender(agent, env, agent_config):
    if agent_config.debug:
        episode_file = open(path.join(DIR, 'results', 'episode'), 'w', 0)

    steps = 0
    try:
        env.handshake()
        for episode in range(agent_config.episode):
            ep_steps = 0
            ep_reward = 0
            state = env.reset()
            agent.init_history(state)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    agent_config = AgentConfig
    env_config = EnvironmentConfig

    run(agent_type='tf_dqn', port=args.port)