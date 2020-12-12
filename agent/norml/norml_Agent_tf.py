from agent.norml.maml_rl import MAMLReinforcementLearning as MAMLRL
class Norml(object):
    def __init__(self,config):
        self.mamlrl = MAMLRL(config)
        print('MAML config: %s' % config)
        tf.logging.info('MAML config: %s', config)

    def init_history(self, state):
        
        pass
    def build_network(self):#inner and outter network is build in init process
        pass
    def observe(self, state, action, reward, done):
        pass
    def get_action(self, s_t):
        pass
    def update_target_q_network(self):
        pass
    def q_learning_mini_batch(self):
        pass
    def save_model(self):#saver is constructed in the init proccess
        self.mamlrl._save_variables()

    def load_model(self):
        self.mamlrl.restore()
    def finetune(self):
        pass