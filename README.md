## Glider: Practical Deep Reinforcement Learning Based Congestion Control

### About the definition of episode

Current episode should be ended when the glider suffers from pathological throughput because the agent is likely to choose actions devious from the rational ones. More detailedly, if the average throughput is consistently lower than a throughput threshold in a predefined time interval, Sender would end the current episode and start a new one. However, the current determination of threshold is too intuitive, that is, we decide it via observing the throughput trace in the debug file. We haven't come up with a good method to determine the threshold.



### Problem Description

There are some problems in current version of code.

* Sender is likely to increase or decrease the cwnd to approach the boundaries (upper bound or lower bound) 
* high delay in emulation experiments



### Done List

* [o] separated the code in Sender into two functional parts: agent and environment.
* [o] found how to train glider at a certain emulation link—more details in mm-test.py.
* [o] designed to train agent episodically.
* [o] Implemented and tested DQN agent, Deep SARSA agent and Cross-Entropy agent.



### TODO List

* [ ] Try to adopt copa's action choice.
* [ ] Continue conducting the emulation experiments.
* [ ] Evaluate the performance using different types of RL agents.
* [ ] Paper survey: find the episode definition.
* [ ] Multiflow environment.

### 设置方法  
在config_maml中，修改输入输出的维度，并且设定task_generator以及task_env_modifiers 训练逻辑

### 训练流程
```
 ./agent/norml/train_maml.py->algo(根据设置选择MAML/NORML/NORML+OFFSET等等一系列算法)->algo.train->maml_rl.py->采样inner_tasks(就是获得batch大小的环境变量，模拟不同的环境变化)-> 
 在外循环下（小于num_outer_iterations）         
    判断终止条件（效果好 early_termination函数 /达到步数）         
    否则在内循环下 for task_idx in range(self.tasks_batch_size)
        对一个环境（task）进行梯度下降train_inner:
            因为可以多次梯度下降，所以使用平行的思想，每次的step会获得多个reward和perform_rollouts(可以获得states，actions，rewards的组合组成的数组，用来计算在这个环境完整训练一个episode的效果)
            先调用_get_batch_env使用task_modifier获得一个不同的环境同时对环境进行 新建并初始化 或者 复用 ，然后reset，接下来action = session.run,batch_env.step(action),在把各种action，state，reward放入rollouts中返回
    当一个taskBatch的不同环境（tasks）训练完成，就对获得的所有rollouts进行整体的参数更新，然后评估是否可以结束，否则继续调用algo.train
```