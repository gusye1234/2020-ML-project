'''
@Author: your name
@Date: 2020-06-14 06:35:42
@LastEditTime: 2020-06-15 01:02:04
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \ML-project-2020\Lunarlander\play_model.py
'''
import sys
sys.path.append("..")
sys.path.append("../..")
import gym
import time

import random
import torch
import pytrace
import numpy as np
# from dqn_agent import Agent
from dqn_agent_craft import Agent
from collections import deque

env = gym.make('LunarLander-v2')

def random_play(action_size=4):
    state = env.reset()
    score = 0.
    for j in range(1000):
        action = np.random.randint(action_size)
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    return score

random_scores = []
for i in range(1000):
    random_scores.append(random_play())

print("random 1000 mean:", np.mean(random_scores))

agent = Agent(state_size=8, action_size=4, seed=0)
agent.qnetwork_local.load_seq_list(pytrace.load('best_list.pth'))

scores = []
for i in range(1000):
    state = env.reset()
    score = 0
    for j in range(1000):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    scores.append(score)
    # print(f"Score: {score}")
env.close()
print("DQN 1000 mean", np.mean(scores))