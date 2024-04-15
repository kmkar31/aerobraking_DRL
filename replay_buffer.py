import numpy as np

class ReplayBuffer():
    def __init__(self, params):
        self.params = params
        
        self.buffer_size = self.params["training"]["buffer_size"]
        
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None
        
    def append(self, obs, acs, rews, next_obs, terminals):
        if self.acs is None:
            self.obs = np.atleast_2d(obs)
            self.acs = np.atleast_1d(acs)
            self.rews = np.atleast_1d(rews)
            self.next_obs = np.atleast_2d(next_obs)
            self.terminals = np.atleast_1d(terminals)
        else:
            self.obs = np.concatenate(np.atleast_2d(obs), axis = 0)
            self.acs = np.concatenate(np.atleast_1d(acs), axis = 0)
            self.rews = np.concatenate(np.atleast_1d(rews), axis = 0)
            self.next_obs = np.concatenate(np.atleast_2d(next_obs), axis = 0)
            self.terminals = np.concatenate(np.atleast_1d(terminals), axis = 0)
        
        if self.__length__() > self.buffer_size:
            self.obs = self.obs[-self.buffer_size:,:]
            self.acs = self.acs[-self.buffer_size:]
            self.rews = self.rews[-self.buffer_size:]
            self.next_obs = self.next_obs[-self.buffer_size:,:]
            self.terminals = self.terminals[-self.buffer_size,:]
            
        
    def __length__(self):
        return len(self.acs)
    
    def sample(self, size):
        indices = np.random.permutation(self.__length__())[:size]
        return self.obs[indices,:], self.acs[indices], self.rews[indices], self.next_obs[indices,:], self.terminals[indices]