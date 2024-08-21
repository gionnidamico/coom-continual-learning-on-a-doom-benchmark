import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
  
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Use np.concatenate or np.stack based on the dimension of the arrays
        state = torch.stack(state)  # Assuming states are of consistent shape
        next_state = torch.stack(next_state)  # Assuming next_states are of consistent shape
        action = torch.stack(action)  # Assuming actions are scalars or vectors
        reward = torch.stack(reward)  # Assuming rewards are scalars
        done = torch.stack(done)  # Assuming dones are scalars

        # Put tensors on CUDA if available only when used to avoid eating up too much gpu ram
        return state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device), done.to(self.device)
    

    
    def __len__(self):
        return len(self.buffer)