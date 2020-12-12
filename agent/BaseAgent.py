import torch

class BaseAgent(object):
    def __init__(self,config,state_dim,action_dim):
        self.model = None
        self.optimizer = None
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_action(self,state):
        pass
    def update(self,state,action,reward,state_):
        pass

    def save_model(self,path):
        torch.save(self.model.state_dict(),path)

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
