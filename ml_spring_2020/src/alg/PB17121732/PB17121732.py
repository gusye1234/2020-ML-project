"""
    For lunarlander.
    To re-trained model, please use `python lunarlander/main.py`
"""
import sys
root_path = './src/alg/PB17121732'
sys.path.append(root_path)
sys.path.append("./src/alg") # to import pytrace

from os.path import join
from copy import deepcopy
from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file


import pytrace 
from pytrace import tracer
from agent import Agent

# ----------------------------------------------------------
# set random seed to re-perform
seed = 0
pytrace.set_seed(seed)

class PB17121732(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17121707','PB17121732','PB18151853'] # 记录队员学号
        # self.config = get_params_from_file('src.alg.PB00000000.rl_configs',params_name='params') # 传入参数
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n
        # ----------------------------------------------------------
        # initialize implemented DQN models and weight
        self.agent  = Agent(self.state_dim, self.action_dim)
        # details about the api of Model in pytrace.nn.Model
        self.agent.qnetwork_local.load_seq_list(
            pytrace.load(
                join(root_path, 'lunarlander/best_list.pth')
                )
            )
        pytrace.prYellow(f"load weight from: {join(root_path, 'lunarlander/best_list.pth')}")
        
        
    def step(self, state):
        action = self.agent.act(state)
        return action

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        print('??')

