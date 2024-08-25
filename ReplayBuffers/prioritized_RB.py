import random
import numpy as np
import torch


class PrioritizedReplayBuffer:  # modified from the vanilla RB
    def __init__(self, device, capacity=10000, alpha=0.5):
        self.buffer = [None] * capacity
        self.position = 0 #pointer
        self.size = 0
        self.capacity = capacity    # max size for this replay buffer

        #PER parameters
        self.max_priority = 1.0  
        self.alpha = alpha  # The exponent alpha to update
        self.priorities = np.zeros(capacity, dtype=np.float32) 

        self.device = device
       
    def push(self, state, action, reward, next_state, done):
        # experience = (s, a, r, s_prime, done)

        priority = self.max_priority  # initialize with max priority, to update later

        #check if there is space available, otherwise delete lesser priority nodes
        if self.size < self.capacity:
            #self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority
        else:
            # Find the index of the minimum priority
            min_priority_index = np.argmin(self.priorities)
            # Replace the minimum priority experience with the new experience
            self.buffer[min_priority_index] = (state, action, reward, next_state, done)
            self.priorities[min_priority_index] = priority

        # Update size counters
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        

    def sample(self, batch_size):
        # Get the whole buffer priorities if full, just the elements available otherwise
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]


        # Compute sampling probabilities for the experiences in the buffer
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Weighted selection of indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)#, replace=False)   # sampling without replacement means no duplicate indices

        # gather each component from the buffer
        batch = [self.buffer[i] for i in indices]
        
        # Gather each component from the buffer
        states = torch.stack([item[0] for item in batch]).to(self.device)
        actions = torch.stack([item[1] for item in batch]).to(self.device)
        rewards = torch.stack([item[2] for item in batch]).to(self.device)
        next_states = torch.stack([item[3] for item in batch]).to(self.device)
        dones = torch.stack([item[4] for item in batch]).to(self.device)

        # Return the samples as separate FloatTensors
        return states, actions, rewards, next_states, dones, indices

        # return samples, indices #, probabilities[indices]


    def update_priorities(self, indices, errors, offset=0.1):
        priorities = errors + offset
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx) % self.capacity] = priority ** self.alpha #int(idx) % self.max_size is to ensure the indices stay in the scope of the buffer


    def __len__(self):
        return self.size