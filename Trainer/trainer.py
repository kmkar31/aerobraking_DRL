import numpy as np
from Policies.directional_exploration import DirectionalExploration
from replay_buffer import ReplayBuffer
from env.Environment import Environment
from concurrent.futures import ProcessPoolExecutor, wait

class Trainer():
    def __init__(self, params, env, agent, init_obs):
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

        self.replay_buffer = []

        self.dqn_agent = agent
        self.obs = init_obs
        self.action = None
        self.reward = 0
        self.next_state = None
        self.terminal = None

        self.policy = DirectionalExploration(self.params)

        self.replay_buffer = ReplayBuffer()


    def training_loop(self):
        while self.train_steps <= self.training_stop:
            self.train_steps += 1

            if self.train_steps >= self.training_start:
                self.train = True
            
            # Get action according to epsilon-greedy
            self.action = self.policy.get_action(self.train_steps, self.obs, self.dqn_agent.output(self.obs))
            
            self.reward, self.next_obs, self.terminal = self.env.step(self.action)
            print(self.reward, self.next_obs, self.terminal)
            # Add to replay buffer
            self.replay_buffer.append(self.obs, self.action, self.reward, self.next_obs, self.terminal)

            self.obs = self.next_obs if not self.terminal else self.env_reset()

            if self.train:
                if self.train_steps % self.train_frequency == 0:
                    # Sample from replay buffer
                    obs, actions, rewards, next_obs, terminals = self.sample()
                    print(obs, actions, rewards, next_obs, terminals)
                    log = self.dqn_agent.update(obs, actions, rewards, next_obs, terminals)
                    self.log(log)
                
                if self.train_steps % self.eval_frequency:
                    self.eval()

    
    def eval(self):
        print("Evaluating Current Policy over 40 Episodes")
        '''
        with ProcessPoolExecutor() as executor:
            tasks = [executor.submit(self.evaluate_episode) for i in range(self.eval_episodes)]
            wait(tasks)
        '''
        
        for episode in range(self.eval_episodes):
            print("Episode, ",episode)
            obs = self.env.reset()
            action = self.policy.get_max_action(self.dqn_agent.output(obs))
            terminal = 0
            while terminal!=1:
                reward, next_obs, terminal = self.env.step(action)
                print(reward)
                reward_list[episode] = reward_list[episode]*self.params["policies"]["discount_rate"]*(1-terminal) + reward
        
        #reward_list = [task.result() for task in tasks]
        print("Mean Reward", np.mean(reward_list))
        print("Std Reward", np.std(reward_list))


    def sample(self):
        indices = np.random.permutation(self.replay_buffer._length())[:self.batch_size]
        #print(indices)
        return self.replay_buffer.sample(indices)
    
    def log(self, msg):
        # TODO
        print(msg)

    def env_reset(self):
        obs = self.env.reset()
        return obs
    
    def evaluate_episode(self):
        env = Environment(self.params)
        obs = self.env.reset()
        action = self.policy.get_max_action(self.dqn_agent.output(obs))
        discounted_reward = 0
        terminal = 0
        while terminal!=1:
            reward, next_obs, terminal = env.step(action)
            discounted_reward = discounted_reward*self.params["policies"]["discount_rate"]*(1-terminal) + reward
        
        return discounted_reward


            

            
