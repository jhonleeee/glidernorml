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

