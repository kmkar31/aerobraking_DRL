import numpy as np
from Agents.MBAgent import MBAgent

class MBTrainer():
    def __init__(self, params):
        self.params = params
        
        self.data_file_path = self.params["model"]["sars_datafile"]
        self.save_directory = self.params["model"]["save_dir"]
        self.num_training_iterations = self.params["model"]["training_iterations"]
        self.batch_size = self.params["model"]["batch_size"]
        
        self.data = np.genfromtxt(self.data_file_path, delimiter=',')
        self.data = np.concatenate((self.data, np.zeros((len(self.data),1))), axis=1)
        self.len_data = len(self.data)
        
        self.input_states = self.data[:,np.array([0,16,1,2,3,4,5,16,16])]
        self.actions = self.data[:,6]
        self.output_states = self.data[:,7:16]
        
        
        self.in_obs_mean = np.mean(self.input_states, axis=0)
        self.in_obs_std = np.std(self.input_states, axis=0)
        self.acs_mean = np.mean(self.actions)
        self.acs_std = np.std(self.actions)
        self.sd_mean = np.mean(self.output_states - self.input_states, axis=0)
        self.sd_std = np.std(self.output_states - self.input_states, axis=0)
        
        #print(self.in_obs_mean, self.in_obs_std, self.acs_mean, self.acs_std, self.out_obs_mean, self.out_obs_std, self.sd_mean, self.sd_std)
        self.agent = MBAgent(params, self.in_obs_mean, self.in_obs_std, self.acs_mean, self.acs_std, self.sd_mean, self.sd_std)
        
    def training_loop(self):
        for iter in range(self.num_training_iterations):
            idxs = np.random.permutation(self.len_data)[:self.batch_size]
            obs = self.input_states[idxs,:]
            acs = self.actions[idxs]
            next_obs = self.output_states[idxs,:]
            
            msg = self.agent.update(obs, acs, next_obs)
            print(msg["Mean Training Loss over Ensembles"], msg["Std Deviation of Training Loss over Ensembles"])
        
        self.agent.save(self.save_directory)
            
        
        
        
        