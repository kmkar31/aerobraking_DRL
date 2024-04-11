import numpy as np
from Network.ANN import NeuralNetwork
import torch
from torch import nn

class MBAgent():
    def __init__(self, params, in_obs_mean, in_obs_std, acs_mean, acs_std, sd_mean , sd_std):
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
        self.in_obs_mean = in_obs_mean
        self.in_obs_std = in_obs_std
        self.acs_mean = acs_mean
        self.acs_std = acs_std

        self.models = []
        self.optimizers = []
        self.losses = []
        for i in range(self.ensemble_size):
            self.models.append(NeuralNetwork(self.input_dim, self.output_dim, self.n_hidden, self.layers))
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters(), lr = self.lr))
            self.losses.append(nn.MSELoss())

    
    def output(self, obs, acs, i = None):
        obs_norm = (obs - self.in_obs_mean)/(self.in_obs_std + 1e-12)
        acs_norm = (acs - self.acs_mean)/(self.acs_std + 1e-12)
        input = np.concatenate((obs_norm, np.atleast_2d(acs_norm).T), axis=1)

        if i is None:
            next_obs_diff_norm = torch.mean(torch.tensor([self.models[i].forward(torch.from_numpy(input).to(torch.float32)) for i in range(self.ensemble_size)]))
        else:
            next_obs_diff_norm = self.models[i].forward(torch.from_numpy(input).to(torch.float32))
        next_obs = self.sd_std*next_obs_diff_norm.detach().numpy() + self.sd_mean

        # Update sd mean and std
        return next_obs_diff_norm, next_obs 
    
    def update(self, obs, acs, next_obs):

        
        losses = torch.zeros([self.ensemble_size,])
        for i in range(self.ensemble_size):
            prediction,_ = self.output(obs, acs, i)
            target = torch.tensor(((next_obs - obs) - self.sd_mean)/(self.sd_std + 1e-9)).to(torch.float32)
            self.optimizers[i].zero_grad()
            loss = self.losses[i](prediction, target)
            loss.backward()
            self.optimizers[i].step()
            losses[i] = loss
        
        return {
            'Mean Training Loss over Ensembles': float(torch.mean(losses).detach().numpy()),
            'Std Deviation of Training Loss over Ensembles' : float(torch.std(losses).detach().numpy())
        }
    
    def eval(self, obs, acs, next_obs):
        losses = torch.zeros([self.ensemble_size,])
        for i in range(self.ensemble_size):
            prediction, output = self.output(obs, acs, i)
            target = torch.tensor(((next_obs - obs) - self.sd_mean)/(self.sd_std + 1e-9)).to(torch.float32)
            loss = self.losses[i](prediction, target)
            losses[i] = loss
        
        return {
            'Un-normalized Output' : output,
            'Mean Training Loss over Ensembles': float(torch.mean(losses).detach().numpy()),
            'Std Deviation of Training Loss over Ensembles' : float(torch.std(losses).detach().numpy())
        }
        
    def save(self, save_dir):
        save_data = dict()
        for (i,model) in enumerate(self.models):
            save_data["Model_" + str(i)] = model
        
        save_data["sd_mean"] = self.sd_mean
        save_data["sd_std"] = self.sd_std
        save_data["in_obs_mean"] = self.in_obs_mean
        save_data["in_obs_std"] = self.in_obs_std
        save_data["acs_mean"] = self.acs_mean
        save_data["acs_std"] = self.acs_std
        
        torch.save(save_data, save_dir + '/model.pth')


