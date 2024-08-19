import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# Function to compute entropy
def compute_entropy(logits):
    prob = torch.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(prob * log_prob, dim=-1).mean()
    return entropy

# Function to select action based on the policy
def select_action(policy_net, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    logits = policy_net(state)
    prob = torch.softmax(logits, dim=-1)
    action = torch.multinomial(prob, 1).item()
    return action

# Training parameters
RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
env = make_env(scenario=Scenario.CHAINSAW, resolution=RESOLUTION, render=RENDER)
state_dim = env.observation_space.shape[0] *  env.observation_space.shape[1] *  env.observation_space.shape[2] * env.observation_space.shape[3]
action_dim = env.action_space.n

q_net1 = QNetwork(state_dim, action_dim)
q_net2 = QNetwork(state_dim, action_dim)
policy_net = PolicyNetwork(state_dim, action_dim)

optimizer_q1 = optim.Adam(q_net1.parameters(), lr=3e-4)
optimizer_q2 = optim.Adam(q_net2.parameters(), lr=3e-4)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-4)

replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
alpha = 0.2  # Entropy coefficient

# Training loop
max_episodes = 1

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = select_action(policy_net, state)
        next_state, reward, done, _ = env.step(action)

         # corrects 'LazyFrames object' error
        state_array = np.array(state)
        state_flattened = state_array.flatten()
        next_state_array = np.array(next_state)
        next_state_flattened = next_state_array.flatten()
        
        replay_buffer.append((state_flattened, action, reward, next_state_flattened, done))
        state = next_state
        episode_reward += reward

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch).unsqueeze(1)
            reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_batch)
            done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

            with torch.no_grad():
                next_logits = policy_net(next_state_batch)
                next_prob = torch.softmax(next_logits, dim=-1)
                next_q1 = q_net1(next_state_batch)
                next_q2 = q_net2(next_state_batch)
                next_q = torch.min(next_q1, next_q2)
                next_value = (next_prob * next_q).sum(dim=-1, keepdim=True)
                next_value -= alpha * compute_entropy(next_logits).unsqueeze(1)
                target_q = reward_batch + (1 - done_batch) * gamma * next_value

            q1 = q_net1(state_batch).gather(1, action_batch)
            q2 = q_net2(state_batch).gather(1, action_batch)
            q1_loss = nn.functional.mse_loss(q1, target_q)
            q2_loss = nn.functional.mse_loss(q2, target_q)

            optimizer_q1.zero_grad()
            q1_loss.backward()
            optimizer_q1.step()

            optimizer_q2.zero_grad()
            q2_loss.backward()
            optimizer_q2.step()

            logits = policy_net(state_batch)
            log_prob = torch.log_softmax(logits, dim=-1)
            q1 = q_net1(state_batch)
            q2 = q_net2(state_batch)
            q = torch.min(q1, q2)
            policy_loss = (log_prob * (alpha * log_prob - q)).sum(dim=-1).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

print(f"Episode {episode}: Total Reward = {episode_reward}")

env.close()
