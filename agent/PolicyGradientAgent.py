from BaseAgent import BaseAgent
from pytorch_ops import PolicyNet
import torch
import torch.optim as optim
import torch.distributions as D
import numpy as np

class PolicyGradientAgent(BaseAgent):
    def __init__(self,config,state_dim,action_dim):
        super(PolicyGradientAgent,self).__init__(config,state_dim,action_dim)
        self.model = PolicyNet(state_dim,action_dim,dropout_p=0.6)
        self.gamma = config.discount
        # episode policy and reward
        self.policy_history = []
        self.reward_episode = []

        # overall reward history and loss history
        self.reward_history = []
        self.loss_history = []
        self.optimizer = optim.Adam(self.model.parameters(),lr=config.learning_rate)

    def get_action(self, s_t):
        s_t = torch.from_numpy(s_t).float().unsqueeze(0)
        probs = self.model(s_t)
        # construct distribution based on probs
        c = D.Categorical(probs=probs)
        action = c.sample()
        # add log probs of our chosen action to history
        self.policy_history.append(c.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        rewards = []
        loss = []
        # use GAMMA to discount past history rewards
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        # scale reward
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # calculate loss
        for log_prob, reward in zip(self.policy_history, rewards):
            loss.append(-log_prob * reward)

        # update policy network
        self.optimizer.zero_grad()
        policy_loss = torch.cat(loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(policy_loss.item())
        self.reward_history.append(np.sum(self.reward_episode))
        del self.policy_history[:]
        del self.reward_episode[:]