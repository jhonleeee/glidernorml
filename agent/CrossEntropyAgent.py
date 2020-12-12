from BaseAgent import BaseAgent
from pytorch_ops import EntropyNet
from collections import OrderedDict
from operator import itemgetter
import multiprocessing

import torch
import math
import numpy as np

class CrossEntropyAgent(BaseAgent):
    def __init__(self,config,state_dim,action_dim):
        super(CrossEntropyAgent, self).__init(config, state_dim, action_dim)
        self.model = EntropyNet(state_dim,action_dim)
        self.device = config.device
        self.gamma = config.gamma
        self.best_score = 0

        self.workers = [self.create_network(config) for _ in range(config["population_size"])]

        self.n_elite = round(config["population_size"] * config["elitism_rate"])

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.model.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

    def set_weights(self, weights):
        """override current weights with given weight list values"""
        weights = iter(weights)
        for param in self.model.parameters():
            param.data.copy_(torch.from_numpy(next(weights)))

    def get_weights(self):
        """get current weights as list"""
        return [param.detach().numpy() for param in self.model.parameters()]



    def learn(self):
        """run one learning step"""
        worker_results = self._run_worker()
        sorted_results = OrderedDict(sorted(worker_results.items(), key=itemgetter(1)))
        elite_idxs = list(sorted_results.keys())[-self.n_elite:]

        elite_weighs = [self.workers[i].get_weights() for i in elite_idxs]
        self.best_weights = [np.array(weights).mean(axis=0) for weights in zip(*elite_weighs)]
        self.model.set_weights(self.best_weights)
        self.best_score = self.objective_function(self.model, gym.make(self.config["env_name"]),
                                                  self.config["max_steps_in_episodes"], 1.0)
        return self.best_score


    def _run_worker(self):
        """evaluate objective function with for n random neighbors of current best weights. Neighbor evaluation
        uses multiprocesses"""
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for worker_id, worker in enumerate(self.workers):
            new_worker_weights = [w + self.config["sigma"] * np.random.normal(size=w.shape) for w in self.best_weights]

            worker.set_weights(new_worker_weights)
            worker_args = (worker, worker_id, gym.make(self.config["env_name"]), self.config, return_dict)
            p = multiprocessing.Process(target=run_worker, args=worker_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        return dict(return_dict)
