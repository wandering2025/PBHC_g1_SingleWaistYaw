import torch

from torch import nn, Tensor

import time
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict


from hydra.utils import instantiate

from humanoidverse.envs.base_task.base_task import BaseTask
import multiprocessing
class BaseAlgo:
    def __init__(self, env: BaseTask, config, device, ipc_queue: multiprocessing.Queue):
        self.env = env
        self.config = config
        self.device = device
        self.ipc_queue = ipc_queue

    def setup(self):
        return NotImplementedError

    def learn(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError
    
    @property
    def inference_model(self):
        return NotImplementedError

    def env_step(self, actions, extra_info=None):
        obs_dict, rewards, dones, extras = self.env.step(actions, extra_info)
        return obs_dict, rewards, dones, extras

    @torch.no_grad()
    def evaluate_policy(self):
        return NotImplementedError
    
    @torch.no_grad()
    def evaluate_policy_steps(self, Nsteps:int):
        return NotImplementedError

    def save(self, path=None, name="last.ckpt"):
        raise NotImplementedError
