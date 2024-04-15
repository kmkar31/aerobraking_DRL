from replay_buffer import ReplayBuffer

class Trainer():
    def __init__(self, params):
        
        self.params = params
        self.training_stop = self.params["training"]["train_stop_steps"]
        self.training_start = self.params["training"]["train_start_steps"]
        
        self.replay_buffer = ReplayBuffer(self.params)

    
    def training_loop(self):
        
        for train_step in range(self.training_stop):
            if train_step < self.training_start:
                # Only collect data
                