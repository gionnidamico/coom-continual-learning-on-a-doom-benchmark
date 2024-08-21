import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# The ValueNetwork estimates the value function 𝑉(𝑠)
class ValueNetwork(nn.Module):
    def __init__(self, num_channels, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # (optional) Weight initialization 
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
        

    def forward(self, state):
        state = state.permute(0, 4, 2, 3, 1)

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x
    
    
# The QNetwork estimates the Q-value 𝑄(𝑠,𝑎) for each action.
class QNetwork(nn.Module):
    def __init__(self, num_channels, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        
         # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_actions)

        # (optional) Weight initialization 
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = state.permute(0, 4, 2, 3, 1)

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.output_layer(x)
        return q_values
    

# The PolicyNetwork outputs a probability distribution over actions.
class PolicyNetwork(nn.Module):
    def __init__(self, num_channels, num_actions, hidden_dim, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        
        # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state, batched=True):
        # if batched:
        #     state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*843]
        # else:
        #     state_flattened = state
        if batched:
            state = state.permute(0, 4, 2, 3, 1)
        else:
            state = state.permute(3, 1, 2, 0).unsqueeze(0)

        x = F.relu(self.conv1(state))   
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    # when training we are in stochastic mode to explore, when training we don't sample from the distribution
    def sample_action(self, state, batched=True, deterministic=True):       # batched is ignored in conv version because there is no flatten() in input to consider
        # if not batched:
        #     state = state.unsqueeze(0)

        probs = self.forward(state, batched)
        if deterministic: # test  mode
            action = torch.argmax(probs, dim=1)
        else:           # train mode
            action = torch.distributions.Categorical(probs).sample()
        
        return action.item()


