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
    def __init__(self, state_dim, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*84*3]
        state_flattened = state
        x = F.relu(self.fc1(state_flattened))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

# The PolicyNetwork outputs a probability distribution over actions.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state, batched=True):
        if batched:
            #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*843]
            state_flattened = state
        else:
            state_flattened = state

        x = F.relu(self.fc1(state_flattened))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample_action(self, state, batched=True):
        
        # Flatten the state to fit in fully connected network
        #state_array = np.array(state)
        #state_flattened = state.view(state.size(0), -1)  # [batch_size, 4*84*84*3]
        if not batched:
            state = state.flatten()

        probs = self.forward(state, batched)
        action = torch.distributions.Categorical(probs).sample()
        return action.item()



'''
# the SAC Algorithm

# Initialize Networks and Optimizers
state_dim = 4*84*84*3  # Example state dimension
num_actions = 12  # Example number of actions
hidden_dim = 256

value_net = ValueNetwork(state_dim, hidden_dim)
q_net1 = QNetwork(state_dim, num_actions, hidden_dim)
q_net2 = QNetwork(state_dim, num_actions, hidden_dim)
policy_net = PolicyNetwork(state_dim, num_actions, hidden_dim)

value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)
q_optimizer1 = optim.Adam(q_net1.parameters(), lr=3e-4)
q_optimizer2 = optim.Adam(q_net2.parameters(), lr=3e-4)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)


# Define the Loss Functions
def value_loss(state, q_net1, q_net2, value_net):
    with torch.no_grad():
        q1 = q_net1(state)
        q2 = q_net2(state)
        q_min = torch.min(q1, q2)
        next_value = q_min - torch.logsumexp(q_min, dim=-1, keepdim=True)
    value = value_net(state)
    return F.mse_loss(value, next_value)

def q_loss(state, action, reward, next_state, done, q_net, target_value_net):
    with torch.no_grad():
        next_value = target_value_net(next_state)
        target_q = reward + (1 - done) * next_value
    q_values = q_net(state)
    q_value = q_values.gather(1, action.long())
    return F.mse_loss(q_value, target_q)

def policy_loss(state, q_net, policy_net):
    probs = policy_net(state)
    q_values = q_net(state)
    log_probs = torch.log(probs + 1e-8)
    policy_loss = (probs * (log_probs - q_values.detach())).sum(dim=-1).mean()
    return policy_loss


# Training Step
def update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer):
    
 
    # Update Value Network
    value_optimizer.zero_grad()
    v_loss = value_loss(state, q_net1, q_net2, value_net)
    v_loss.backward()
    value_optimizer.step()

    # Update Q Networks
    q_optimizer1.zero_grad()
    q_optimizer2.zero_grad()
    q_loss1 = q_loss(state, action, reward, next_state, done, q_net1, value_net)
    q_loss2 = q_loss(state, action, reward, next_state, done, q_net2, value_net)
    q_loss1.backward()
    q_loss2.backward()
    q_optimizer1.step()
    q_optimizer2.step()

    # Update Policy Network
    policy_optimizer.zero_grad()
    p_loss = policy_loss(state, q_net1, policy_net)
    p_loss.backward()
    policy_optimizer.step()

    
'''


'''
# Step 4: Create and Train in an Environment
# Let's use a simple custom environment to demonstrate training.
import gym
import numpy as np

env = gym.make('CartPole-v1')

num_episodes = 500
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)

    episode_reward = 0
    done = False
    while not done:
        action = policy_net.sample_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward]).unsqueeze(1)
        done = torch.FloatTensor([done]).unsqueeze(1)

        update(state, torch.LongTensor([[action]]), reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)

        state = next_state
        episode_reward += reward.item()

    print(f"Episode {episode}, Reward: {episode_reward}")

'''