"""
    For lunarlander.
    To re-trained model, please use `python riverraid/main.py`
"""
import sys
root_path = './src/alg/PB18151853'
sys.path.append(root_path)
sys.path.append("./src/alg") # import pytrace

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

class PB18151853(RL_alg):
    def __init__(self,ob_space,ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17121707','PB17121732','PB18151853'] # 记录队员学号
        # self.config = get_params_from_file('src.alg.PB00000000.rl_configs',params_name='params') # 传入参数
        
        self.ac_space = ac_space
        self.state_dim = ob_space.shape
        print(self.state_dim)
        self.action_dim = ac_space.n
        # ----------------------------------------------------------
        # initialize implemented DQN models and weight
        self.agent  = Agent()
        # details about the api of Model in pytrace.nn.Model
        self.agent.qnetwork_local.load_seq_list(
            pytrace.load(
                join(root_path, './riverraid/best_list.pth')
                )
            )
        pytrace.prYellow(f"load weights from: {join(root_path, './riverraid/best_list.pth')}")
        
        
    def step(self, state):
        action = self.agent.act(state)
        return action

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        print('??')

