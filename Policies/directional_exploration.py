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
    

    def get_action(self, env_step_number, obs, Q_s_a):
        actions = []
        
        for (i,ob) in enumerate(obs):
            p = np.random.random()
            if p < self.get_current_epsilon(env_step_number):
                if ob[7] < 0.1:
                    actions.append(np.random.randint(self.action_dim//2+1, self.action_dim))
                elif ob[7] > 0.25:
                    actions.append(np.random.randint(0,self.action_dim//2))
                else:
                    actions.append(np.random.randint(0, self.action_dim))
            else:
                actions.append(np.argmax(Q_s_a[i,:]))

        return actions