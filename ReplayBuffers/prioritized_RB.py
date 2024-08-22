import random
import numpy as np
import torch


class PrioritizedReplayBuffer:  # modified from the vanilla RB
    def __init__(self, device, max_size=10000, alpha=0.5):
        self.buffer = []
        self.ptr = 0 #pointer
        self.size = 0
        self.max_size = max_size

        #PER parameters
        self.max_priority = 1.0  
        self.alpha = alpha  # The exponent alpha to update
        self.priorities = np.zeros((max_size,), dtype=np.float32) 

        self.device = device
       
    def push(self, state, action, reward, next_state, done):
        #experience = (s, a, r, s_prime, terminated)
        priority = self.max_priority  # initialize with max priority, to update later

        #check if there is space available, otherwise delete lesser priority nodes
        if self.size < self.max_size:
            self.buffer[self.position] = (state, action, reward, next_state, done, priority)
            self.position = (self.position + 1) % self.capacity
        else:
            # Find the index of the minimum priority
            min_priority_index = np.argmin(self.priorities)
            # Replace the minimum priority experience with the new experience
            self.buffer[min_priority_index] = (state, action, reward, next_state, done, priority)

        # Update size counters
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        # Get the whole buffer if full, just the elements in it otherwise
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]

        # Compute sampling probabilities for the experiences in the buffer
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Weighted selection
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        samples = (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),   
            torch.FloatTensor(self.done[indices]).to(self.device),
        )
        return samples, indices #, probabilities[indices]


    def update_priorities(self, indices, errors, offset=0.1):
        priorities = errors + offset
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx) % self.max_size] = priority ** self.alpha #int(idx) % self.max_size is to ensure the indices stay in the scope of the buffer


    def __len__(self):
        return self.size