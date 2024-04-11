import yaml

import numpy as np
import sys, os
import torch

sys.path.append(os.path.join(os.getcwd(),'trainer'))
sys.path.append(os.path.join(os.getcwd(),'env'))
sys.path.append(os.path.join(os.getcwd(),'Agents'))
sys.path.append(os.path.join(os.getcwd(),'data'))

from Trainer.model_trainer import MBTrainer
from Agents.MBAgent import MBAgent

f = open('eval.csv','w+')
sys.stdout = f
param_file = 'env/params.yaml'
with open(param_file) as f:
    params = yaml.safe_load(f)



# Training Phase
#trainer = MBTrainer(params)
#trainer.training_loop()

# Eval

model_path = params["model"]["save_dir"] + '/model.pth'
model_data = torch.load(model_path)
agent = MBAgent(params, model_data["in_obs_mean"], model_data["in_obs_std"], \
                model_data["acs_mean"], model_data["acs_std"], model_data["sd_mean"], model_data["sd_std"])

for (i, model) in enumerate(agent.models):
    model = model_data["Model_" + str(i)]

eval_file = 'data/sars_eval.csv'
data = np.genfromtxt(eval_file, delimiter=',')
data = np.concatenate((data, np.zeros((len(data),1))), axis=1)
        
input_states = data[:,np.array([0,16,1,2,3,4,5,16,16])]
actions = data[:,6]
output_states = data[:,7:16]

log = agent.eval(input_states, actions, output_states)

output = log["Un-normalized Output"]
for i in range(len(output)):
    print(','.join(map(str, output[i,:])))
print(log["Mean Training Loss over Ensembles"])
print(log["Std Deviation of Training Loss over Ensembles"])

f.close()