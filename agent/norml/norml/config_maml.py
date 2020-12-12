# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurations for MAML training (Reinforcement Learning).

See maml_rl.py for usage examples.
An easy task to get started with is: RL_MINITAUR_POINT_CONFIG_CIRCLE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np
import tensorflow.compat.v1 as tf

from norml import networks
from norml import policies
from norml.envs import cartpole_sensor_bias_env
#from norml.envs import halfcheetah_motor_env
from norml.envs import move_point_env


def _early_termination_avg(rewards, num_steps, avg_reward):
  """Early termination based on average reward."""
  flat_reward = np.array(rewards).ravel()
  len_ok = len(flat_reward) >= num_steps
  val_ok = np.mean(flat_reward[-num_steps:]) >= avg_reward
  return len_ok and val_ok


MOVE_POINT_ROTATE_MAML = dict(
    random_seed=random.randint(0, 1000000),
    num_outer_iterations=1000,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=-1,
        trial_length=10),
    task_env_modifiers=[{
        '_action_rotation': i
    } for i in np.linspace(-np.pi, np.pi, 5000)],
    network_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=2,
        dim_output=2,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    input_dims=2,
    pol_log_std_init=-3.,
    output_dims=2,
    reward_disc=0.9,
    learn_offset=False,
    policy=policies.GaussianPolicy,
    tasks_batch_size=10,
    num_inner_rollouts=25,
    outer_optimizer_algo=tf.train.AdamOptimizer,
    advantage_function='returns-values',
    whiten_values=False,
    always_full_rollouts=False,
    inner_lr_init=0.02,
    outer_lr_init=7e-3,
    outer_lr_decay=True,
    first_order=False,
    learn_inner_lr=True,
    learn_inner_lr_tensor=True,
    fixed_tasks=False,
    ppo=True,
    ppo_clip_value=0.2,
    max_num_batch_env=1000,
    max_rollout_len=10,
    log_every=10,
)
MOVE_POINT_ROTATE_MAML_LAF = MOVE_POINT_ROTATE_MAML.copy()
MOVE_POINT_ROTATE_MAML_LAF.update(
    learn_inner_lr=False,
    learn_inner_lr_tensor=False,
    learn_advantage_function_inner=True,
    advantage_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=2 * 2 + 2,
        dim_output=1,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    inner_lr_init=0.7,
    outer_lr_init=6e-4,
    pol_log_std_init=-3.25)

MOVE_POINT_ROTATE_NORML = MOVE_POINT_ROTATE_MAML_LAF.copy()
MOVE_POINT_ROTATE_NORML.update(
    learn_offset=True,
    inner_lr_init=10.,
    outer_lr_init=6e-3,
    pol_log_std_init=-0.75)