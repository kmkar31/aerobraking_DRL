import torch
import numpy as np
from neural_network import NeuralNetwork

class Environment():
    def __init__(self, params, model_file) -> None:
        
        self.action_map = self.params["setup"]["actions"]
        self.model = NeuralNetwork()
        
        self.input_dim = self.params["setup"]["observation_dim"] + 1 # Extra 1 for the input action
        self.output_dim = self.params["setup"]["observation_dim"]
        self.layers = self.params["model"]["hidden_layers"]
        self.n_hidden = self.params["model"]["n_hidden"]
        self.lr = self.params["model"]["learning_rate"]
        self.ensemble_size = self.params["model"]["ensemble_size"]
        
        model_data = torch.load(model_file)

        # Model should predict the "normalized" state "difference"
        self.sd_mean = model_data["sd_mean"]
        self.sd_std = model_data["sd_std"]
        self.in_obs_mean = model_data["in_obs_mean"]
        self.in_obs_std = model_data["in_obs_std"]
        self.acs_mean = model_data["acs_mean"]
        self.acs_std = model_data["acs_std"]

        self.models = []
        for i in range(self.ensemble_size):
            model = model_data['model_' + str(i)]
            self.models.append(model)
            
            
    
    def step(self, action_idx):
        
        deltaV = self.action_map[action_idx]
        
        
    def output(self, obs, acs):
        obs_norm = (obs - self.in_obs_mean)/(self.in_obs_std + 1e-12)
        acs_norm = (acs - self.acs_mean)/(self.acs_std + 1e-12)
        input = np.concatenate((obs_norm, np.atleast_2d(acs_norm).T), axis=1)

        input = torch.from_numpy(input).to(torch.float32)
        output = [self.models[i].forward(input).detach().numpy() for i in range(self.ensemble_size)]
        next_obs_diff_norm = np.mean(output, axis=0)
        next_obs = self.sd_std*next_obs_diff_norm + self.sd_mean + obs

        return next_obs_diff_norm, next_obs
        
        
        