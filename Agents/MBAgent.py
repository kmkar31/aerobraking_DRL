import numpy as np
from Network.ANN import NeuralNetwork
import torch
from torch import nn

class MBAgent():
    def __init__(self, params, obs_mean, obs_std, acs_mean, acs_std, sd_mean , sd_std):
        self.params = params

        self.input_dim = self.params["setup"]["observation_dim"] + 1 # Extra 1 for the input action
        self.output_dim = self.params["setup"]["observation_dim"]
        self.layers = self.params["model"]["hidden_layers"]
        self.n_hidden = self.params["model"]["n_hidden"]
        self.lr = self.params["model"]["learning_rate"]
        self.ensemble_size = self.params["model"]["ensemble_size"]

        # Model should predict the "normalized" state "difference"
        self.sd_mean = sd_mean
        self.sd_std = sd_std
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acs_mean = acs_mean
        self.acs_std = acs_std

        self.models = []
        self.optimizers = []
        self.losses = []
        for i in range(self.ensemble_size):
            self.models.append(NeuralNetwork(self.input_dim, self.output_dim, self.n_hidden, self.layers))
            self.optimizers.append(torch.optim.Adam(self.model.parameters(), lr = self.lr))
            self.losses.append(nn.MSELoss())

    
    def output(self, obs, acs, i = None):
        obs_norm = (obs - self.obs_mean)/(self.obs_std + 1e-9)
        acs_norm = (acs - self.acs_mean)/(self.acs_std + 1e-9)
        input = np.concatenate((obs_norm, np.atleast_2d(acs_norm).T), axis=1)

        if i is None:
            next_obs_diff_norm = torch.mean(torch.tensor([self.models[i].forward(torch.from_numpy(input).to(torch.float32)) for i in range(self.ensemble_size)]))
        else:
            next_obs_diff_norm = self.models[i].forward(torch.from_numpy(input).to(torch.float32))
        next_obs = self.std*next_obs_diff_norm + self.mean

        # Update sd mean and std
        return next_obs_diff_norm, next_obs 
    
    def update(self, obs, acs, next_obs):

        target = ((next_obs - obs) - self.sd_mean)/(self.sd_std + 1e-9)
        loss = torch.tensor([])
        for i in range(self.ensemble_size):
            _, prediction = self.output(obs, acs, i)

            self.optimizers[i].zero_grad()
            loss[i] = self.losses[i](prediction, target)
            loss[i].backward()
            self.optimizers[i].step()

        return {
            'Mean Training Loss over Ensembles': torch.mean(loss).detach().numpy(),
            'Std Deviation of Training Loss over Ensembles' : torch.std(loss).detach().numpy()
        }


