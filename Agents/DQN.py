import numpy as np
import torch
from torch import nn
from Network.ANN import NeuralNetwork


class DQN():
    def __init__(self, params):
        # Dictionary Storing all pertinent parameters
        self.params = params

        # RL Setup
        self.ddqn = self.params['setup']['DDQN']

        # Real-world Parameters to Neural Network Parameters
        self.input_dim = self.params['setup']['observation_dim']
        self.output_dim = self.params['setup']['action_dim']
        self.hidden_dim = self.params['neural_network']['hidden_dim']
        self.n_layers = self.params['neural_network']['hidden_layers']
        self.activation = self.params['neural_network']['activation']
        self.output_activation = self.params['neural_network']['output_activation']
        
        self.gamma = self.params['policies']['discount_rate']
        self.lr = self.params['neural_network']['learning_rate']

        # Q Network Initialization
        self.q_net = NeuralNetwork(self.input_dim, self.output_dim, self.hidden_dim, self.n_layers, \
                                  self.activation, self.output_activation)
        self.q_loss = nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Target Q Network Initialization
        if self.ddqn:
            self.target_q_net = NeuralNetwork(self.input_dim, self.output_dim, self.hidden_dim, self.n_layers, \
                                  self.activation, self.output_activation)
            self.lambda_update = self.params['neural_network']['lambda_target_network']
            self.target_update_freq = self.params['neural_network']['target_network_update_frequency']
            self.num_update_steps = 0


    def update(self, observations, actions, rewards, next_observations, terminals):
        '''
            Conducts a single step update of the Q network
            Every update frequency steps, it also updates the weights of the targte network Q'
        '''
        observations = torch.from_numpy(observations).to(torch.float32)
        actions = torch.from_numpy(actions).to(torch.int64)
        rewards = torch.from_numpy(rewards).to(torch.float32)
        next_observations = torch.from_numpy(next_observations).to(torch.float32)
        terminals = torch.from_numpy(terminals).to(torch.int64)

        # Training
        q_s_a = self.q_net.forward(observations) # Returns over all actions
        q_s_a = torch.gather(q_s_a, 1, actions.unsqueeze(1)).squeeze(1) # Predicted by Q_network
        q_s_next_a = self.q_net.forward(next_observations) # From primary network
        
        if self.ddqn:
            q_s_a_target = self.target_q_net(observations)
            argmax_actions = q_s_a_target.argmax(1)
            q_ddqn = torch.gather(q_s_a_target, 1, argmax_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma*(1-terminals)*q_ddqn
        
        else:
            target = rewards + self.gamma*np.max(q_s_next_a, 0)
        
        loss = self.q_loss(q_s_a, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # Update Target Q-network
        if self.ddqn:
            self.update_target()

        return {
            'Training Loss': loss.detach().numpy(),
        }
    
    def update_target(self):
        for target_param, param in zip(
                self.target_q_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)
    
    def output(self, obs):
        return self.q_net(torch.tensor(obs).to(torch.float32)).detach().numpy()







            
            
        

