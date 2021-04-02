# coding=utf-8
"""A short script for training MAML.
Example to run
python -m norml.train_maml --config MOVE_POINT_ROTATE_MAML
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from dotmap import DotMap
import tensorflow.compat.v1 as tf
import sys
sys.path.append('/home/lilimul/Documents/norml/glidernorml/agent/norml')
from norml import config_maml
from norml import maml_rl


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'RL_PENDULUM_GYM_CONFIG_META',
                    'Configuration for training.')

#--config CC_NORML
def main(argv):
  del argv  # Unused
  config = DotMap(getattr(config_maml, FLAGS.config))#get --config arg from cmd
  print('MAML config: %s' % FLAGS.config)
  tf.logging.info('MAML config: %s', FLAGS.config)
  algo = maml_rl.MAMLReinforcementLearning(config)#without reporter,logDir,notSaveConfig
  sess_config = tf.ConfigProto(allow_soft_placement=True)#GPU/CPU 
  sess_config.gpu_options.allow_growth = True

  with tf.Session(config=sess_config) as sess:
    algo.init_logging(sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    done = False
    while not done:
      done, _ = algo.train(sess, 10)
    algo.stop_logging()


if __name__ == '__main__':
  tf.app.run()
