import numpy as np
from Policies.directional_exploration import DirectionalExploration

class Trainer():
    def __init__(self, params, env, agent, init_state):
        self.params = params

        self.training_start = self.params['training']['train_start_steps']
        self.training_stop = self.params['training']['train_stop_steps']
        self.train_frequency = self.params['training']['train_frequency']
        self.eval_episodes = self.params['training']['eval_episodes']
        self.eval_frequency = self.params['training']['eval_frequency']
        self.buffer_size = self.params['training']['buffer_size']
        self.batch_size = self.params['training']['batch_size']

        self.train = False
        self.train_steps = 0
        self.env = env

        self.replay_buffer = dict(zip(['States', 'Actions', 'Rewards', 'Next_States', 'Terminals'], [[],[],[],[], []]))

        self.dqn_agent = agent
        self.state = init_state
        self.action = None
        self.reward = 0
        self.next_state = None
        self.terminal = None

        self.policy = DirectionalExploration(self.params)


    def training_loop(self):
        while self.train_steps <= self.training_stop:
            self.train_steps += 1

            if self.train_steps >= self.training_start:
                self.train = True

            # Get action according to epsilon-greedy
            self.action = self.policy.get_action(self.train_steps, self.state, self.dqn_agent.output(self.mask(self.state)))
            self.reward, self.next_state, self.terminal = self.env.step(self.state, self.action)

            # Add to replay buffer
            self.replay_buffer['States'].append(self.state)
            self.replay_buffer['Actions'].append(self.action)
            self.replay_buffer['Rewards'].append(self.reward)
            self.replay_buffer['Next_States'].append(self.next_state)
            self.replay_buffer['Terminals'].append(self.terminal)

            self.state = self.next_state if not self.terminal else self.env_reset()

            if self.train:
                if self.train_steps % self.train_frequency == 0:
                    # Sample from replay buffer
                    states, actions, rewards, next_states, terminals = self.sample()
                    log = self.dqn_agent.update(states, actions, rewards, next_states, terminals)
                    self.log(log)
                
                if self.train_steps % self.eval_frequency:
                    self.eval()

    
    def eval(self):
        num_episodes = 0
        while num_episodes < self.eval_episodes:
            pass
        # TODO


    def unmask(self, state):
        '''
            Converts a list State to a dictionary State
        '''
        return self.env.unmask(state)

    def sample(self):
        indices = np.random.permutation(len(self.replay_buffer['States']))[:self.batch_size]
        states = self.replay_buffer['States'][indices]
        actions = self.replay_buffer['Actions'][indices]
        rewards = self.replay_buffer['Rewards'][indices]
        next_states = self.replay_buffer['Next_States'][indices]
        terminals = self.replay_buffer['Terminals'][indices]

        return states, actions, rewards, next_states, terminals
    
    def log(self):
        # TODO
        pass

    def env_reset(self):
        state = self.env.reset()
        return state
            

            

            
