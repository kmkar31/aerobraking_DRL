import numpy as np
import csv

class ReplayBuffer():
    def __init__(self):
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None

    def append(self, obs, actions, rewards, next_obs, terminals):
        if self.actions is not None:
            self.observations = np. concatenate((self.observations, np.atleast_2d(obs)), axis=0)
            self.actions = np.concatenate((self.actions, np.atleast_1d(actions)), axis=0)
            self.rewards = np.concatenate((self.rewards, np.atleast_1d(rewards)), axis=0)
            self.next_observations = np.concatenate((self.next_observations, np.atleast_2d(next_obs)), axis=0)
            self.terminals = np.concatenate((self.terminals, np.atleast_1d(terminals)), axis=0)
        else:
            self.observations = np.atleast_2d(obs)
            self.actions = np.atleast_1d(actions)
            self.rewards = np.atleast_1d(rewards)
            self.next_observations = np.atleast_2d(next_obs)
            self.terminals = np.atleast_1d(terminals)

        #print(self.observations, self.actions)

    def sample(self, indices):
        return self.observations[indices, :], self.actions[indices], self.rewards[indices], self.next_observations[indices, :], self.terminals[indices]
    
    def _length(self):
        return len(self.actions)
    
    def save(self):
        np.savetxt('data_1000.csv', np.concatenate((self.observations, np.atleast_2d(self.actions).T, np.atleast_2d(self.rewards).T, self.next_observations, np.atleast_2d(self.terminals).T), axis=1))

