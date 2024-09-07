import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# The ValueNetwork estimates the value function 𝑉(𝑠)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*84*3]
        state_flattened = state
        x = F.relu(self.fc1(state_flattened))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return value

# The QNetwork estimates the Q-value 𝑄(𝑠,𝑎) for each action.
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, n_head = 1, init_w=3e-3):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layerl = nn.ModuleList()
        for _ in range(n_head): # n_head is 1 if owl is not used else is = number of tasks
            self.output_layerl.append(nn.Linear(hidden_dim, num_actions))

    def forward(self, state, head = 0):
        #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*84*3]
        state_flattened = state
        x = F.relu(self.fc1(state_flattened))
        x = F.relu(self.fc2(x))
        q_values = self.output_layerl[head](x)
        return q_values

# The PolicyNetwork outputs a probability distribution over actions.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, n_head = 1, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layerl = nn.ModuleList()
        for _ in range(n_head): # n_head is 1 if owl is not used else is = number of tasks
            self.output_layerl.append(nn.Linear(hidden_dim, num_actions))

    def forward(self, state, batched=True, head = 0):
        if batched:
            #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*843]
            state_flattened = state.flatten()
        else:
            state_flattened = state.flatten().unsqueeze(0)

        x = F.relu(self.fc1(state_flattened))
        x = F.relu(self.fc2(x))
        logits = self.output_layerl[head](x)
        probs = F.softmax(logits, dim=-1)
        return probs
    

        # when training we are in stochastic mode to explore, when training we don't sample from the distribution
    def sample_action(self, state, task, batched=True, deterministic=True):       # batched is ignored in conv version because there is no flatten() in input to consider
        # if not batched:
        #     state = state.unsqueeze(0)
        # Flatten the state to fit in fully connected network
        #state_array = np.array(state)
        #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*84*3]
        if not batched:
            state = state.flatten()

        probs = self.forward(state, batched)
        if deterministic: # test  mode
            action = torch.argmax(probs, dim=1)
        else:           # train mode
            action = torch.distributions.Categorical(probs).sample()
        
        return action.item()


