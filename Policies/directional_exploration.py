# Runs an epsilon-greedy algorithm to obtain actions during a rollout
import numpy as np

class DirectionalExploration():
    def __init__(self, params):
        self.params = params
        self.decay_rate = self.params['policies']['decay_rate']

        # Actions dimension is (2k+1) with the first k values representing negative thrust 
        # and the next k values representing positive thrust
        self.action_dim = self.params['setup']['action_dim']

    def get_current_epsilon(self, env_step_number):
        return np.exp(-env_step_number/self.decay_rate)
    

    def get_action(self, env_step_number, ob, Q_s_a):
        action = None
        p = np.random.random()
        if p < self.get_current_epsilon(env_step_number):
            if ob[7] < 0.25:
                action = np.random.randint(self.action_dim//2+1, self.action_dim)
            elif ob[7] > 0.75:
                action = np.random.randint(0,self.action_dim//2)
            else:
                action = np.random.randint(0, self.action_dim)
        else:
            action = np.argmax(Q_s_a).squeeze()

        return action
    
    def get_max_action(self, Q_s_a):
        return np.argmax(Q_s_a).squeeze()