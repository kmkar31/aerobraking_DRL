from neural_network import NeuralNetwork
import torch

class DQN():
    def __init__(self, params):
        self.params = params
        
        self.obs_dim = self.params["setup"]["observation_dim"]
        self.acs_dim = self.params["acs_dim"]["action_dim"]
        self.n_hidden = self.params["q_network"]["hidden_dim"]
        self.layers = self.params["q_network"]["hidden_layers"]
        self.activation = self.params["q_network"]["activation"]
        self.gamma = self.params["policies"]["discount_rate"]
        self.lr = self.params["q_network"]["learning_rate"]
        
        self.q_net_A = NeuralNetwork(self.obs_dim, self.acs_dim, self.n_hidden, self.layers, self.activation)
        self.q_net_B = NeuralNetwork(self.obs_dim, self.acs_dim, self.n_hidden, self.layers, self.activation)
        
        self.optimizer = torch.optim.Adam(self.q_net_A.parameters(), lr = self.lr)
        self.loss = torch.nn.MSELoss()
        
    
    def forward(self, input):
        # input is a tensor
        return self.q_net_A.forward(input)
    
    def output(self, obs, acs):
        # obs and acs are numpy arrays
        # Make sure to pass obs as a 2d array
        Q_s = self.forward(torch.from_numpy(obs).to(torch.float32)).detach().numpy()
        return Q_s[:,acs]
    
    def update(self, obs, acs, rews, next_obs, terminals):
        
        obs = torch.from_numpy(obs).to(torch.float32)
        acs = torch.from_numpy(acs).to(torch.float32)
        rews = torch.from_numpy(rews).to(torch.float32)
        next_obs = torch.from_numpy(next_obs).to(torch.float32)
        terminals = torch.from_numpy(terminals).to(torch.float32)
        
        q_A_s = self.forward(obs) # Q(s, all a) at current state from network A
        q_A_s_a = torch.gather(q_A_s, dim=1, index = acs)
        
        a_max = torch.argmax(self.forward(torch.from_numpy(next_obs).to(torch.float32)), dim=1) # best action at next state by neteork A
        q_B_snext = self.q_net_B.forward(torch.from_numpy(next_obs).to(torch.float32))
        q_B_snext_a = torch.gather(q_B_snext, dim=1, index=a_max.squeeze())
        
        target = rews + self.gamma*(1-terminals)*q_B_snext_a
        
        loss = self.loss(q_A_s_a, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.steps()
        
        return {'Training Loss': loss.detach().numpy()}
    
    def update_target(self):
        for B_param, A_param in zip(
                self.q_net_B.parameters(), self.q_net_A.parameters()
        ):
            B_param.data.copy_(A_param.data)
            
    def save(self, filename):
        torch.save(self.q_net_A, self.params["model"]["save_dir"] + '/' + filename)
        
        
         
        