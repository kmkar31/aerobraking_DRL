import yaml

import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(),'env'))
from env.Environment import Environment

param_file = 'env/params.yaml'
with open(param_file) as f:
    params = yaml.safe_load(f)
env = Environment(params)

for i in range(10):
    r,ob,terminal = env.step(np.random.randint(0,11))
    print(r, ob, terminal)
    if terminal:
        ob = env.reset()
        print(ob)