import yaml

import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(),'trainer'))
sys.path.append(os.path.join(os.getcwd(),'env'))
sys.path.append(os.path.join(os.getcwd(),'Agents'))
from env.Environment import Environment
from Agents.DQN import DQN
from Trainer.trainer import Trainer

param_file = 'env/params.yaml'
with open(param_file) as f:
    params = yaml.safe_load(f)
env = Environment(params)
init_obs = env.get_observation()
agent = DQN(params)

trainer = Trainer(params, env, agent, init_obs)
trainer.training_loop()