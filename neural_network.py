import torch
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation = 'ReLU', output_activation = 'Identity', dropout = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = getattr(nn, activation)()
        self.output_activation = getattr(nn, output_activation)()
        self.dropout = dropout


        # Neural Network Definition
        self.network = nn.Sequential()
        # Input Layer
        self.network.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.network.append(self.activation)
        # Hidden Layers
        for n in range(n_layers):
            self.network.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.network.append(self.activation)
            if self.dropout and n==0:
                self.network.append(nn.Dropout(0.2))
        # Output Layer
        self.network.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.network.append(self.output_activation)
    

    def forward(self, input):
        '''
            Returns Neural Network Output for the given input
            Takes in a torch tensor and returns a torch tensor
        '''
        return self.network(input)
    
    def display(self):
        '''
            Prints the network to console
        '''
        print(self.network)
    
    def output(self, input):
        '''
            Returns Neural Network Output for the given input
            Takes in a numpy array and returns a numpy array
        '''
        input = torch.from_numpy(input).to(torch.float32)
        return self.network(input).detach().numpy()

