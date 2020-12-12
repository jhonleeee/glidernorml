import sys
from agent.BaseAgent import BaseAgent
from pytorch_ops import DQN
from replay_memory import ReplayMemory
from project_root import DIR
from os import path
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from history import History


class DQN_Agent(BaseAgent):
    def __init__(self, config, state_dim, action_dim):
        super(DQN_Agent, self).__init__(config, state_dim, action_dim)
        self.config = config
        self.debug = config.debug

        # tensor device
        self.device = config.device
        # discount
        self.gamma = config.discount
        # ep-greedy selection startup
        self.ep = 1
        # replay memory capacity
        self.replay_memory_size = config.memory_size
        # sample batch size
        self.batch_size = config.batch_size
        # self.learn_start = config.LEARN_START

        # learning rate
        self.lr = config.learning_rate

        # train network and target network
        self.declare_network()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9, eps=0.01)

        if self.debug:
            self.loss_count = 0
            self.loss_file = open(path.join(DIR, 'results', 'loss'), 'w', 0)

        # load model
        if not config.train:
            self.load_model(config.model_path)
            self.model.eval()
        else:
            self.model.train()
            self.target_model.eval()
        # create replay memory
        self.declare_replay_memory()
        # create history
        self.history = History(self.config, self.state_dim)

        self.learn_step_counter = 0

    def declare_network(self):
        self.model = DQN(self.state_dim, self.config.history_length, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.config.history_length, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def declare_replay_memory(self):
        self.memory = ReplayMemory(self.config, self.state_dim)

    def pre_minibatch(self):
        state, action, reward, next_state, done = self.memory.sample()

        shape = self.state_dim
        state = np.array(state)
        action = np.array(action)[:, np.newaxis]
        reward = np.array(reward)

        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        done = np.array(done) + 0.
        done = torch.tensor(done, device=self.device, dtype=torch.float32)

        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        except:
            non_final_next_states = None

        return state, reward, action, non_final_next_states, done

    def compute_loss(self, batch_vars):
        batch_state, batch_reward, batch_action, non_final_next_states, done = batch_vars

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_state)))

        # estimate
        state_action_values = self.model(batch_state).gather(1, batch_action)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.config.batch_size, device=self.config.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (1 - done) * self.gamma * next_state_values + batch_reward

        # huber loss
        loss = F.smooth_l1_loss(expected_state_action_values.unsqueeze(1), state_action_values)
        return loss

    def update(self, **kwargs):
        batch_vars = self.pre_minibatch()

        loss = self.compute_loss(batch_vars)
        if math.isnan(loss.item()):
            print("loss is nan")

        if self.debug:
            self.loss_file.write("%d, %.4f\n" % (self.loss_count, loss))
            self.loss_count += 1
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def observe(self, state, action, reward, done):
        self.history.add(state)
        self.memory.add(state, action, reward, done)

        # fetch batch to train from memory
        if self.learn_step_counter >= self.config.learn_start:
            if self.learn_step_counter % self.config.train_frequency == 0:
                self.update()

        # update target network cloning predict net
        if self.learn_step_counter % self.config.target_q_update_step == 0:
            self.update_target_q_network()

        self.learn_step_counter += 1
        self.save_model()

    def get_action(self, state):
        if self.config.train:
            # ep-greedy
            if np.random.random() >= self.ep:
                print("use the model, self.ep=%f" % (self.ep))
                with torch.no_grad():
                    x = torch.tensor([state], device=self.device)
                    a = self.model(x).max(1)[1].view(1, 1)
                    return a.item()
            else:
                self.ep -= 0.0001
                self.ep = max(0.01, self.ep)
                return np.random.randint(0, self.action_dim)
        else:
            # only load and use the model dict
            x = torch.tensor([state], device=self.device)
            a = self.model(x).max(1)[1].view(1, 1)
            return a.item()

    def save_model(self):
        if self.learn_step_counter % self.config.save_model_step == 0:
            n = self.learn_step_counter/self.config.save_model_step
            save_path = path.join(DIR, 'models', 'model-' + str(n) + '.pkl')
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, filepath):
        sys.stderr.write("Test Phase: loading model........"+filepath+'\n')
        model_path = path.join(DIR, 'models', filepath)
        self.model.load_state_dict(torch.load(model_path))
        self.target_model.load_state_dict(self.model.state_dict())

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def update_target_q_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def init_history(self, state):
        for _ in range(self.config.history_length):
            self.history.add(state)
