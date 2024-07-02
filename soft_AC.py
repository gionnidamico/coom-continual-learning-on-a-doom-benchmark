import math
import random

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from COOM.env.builder import make_env
from COOM.utils.config import Scenario



# use gpu
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)


# replay buffer implementation
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    

# actor network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        

# critic network      
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs , hidden_size)  #+ num_actions
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1) 
        
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        q_values = self.output_layer(x)
        return q_values
       
        

# soft AC agent network        
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        # self.log_std_min = log_std_min
        # self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, num_actions) ###usare init 
        
        # self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        # self.log_std_linear = nn.Linear(hidden_size, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)  # Output probabilities for each action

    
    def evaluate(self, state, epsilon=1e-6):
        state = torch.Tensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob  #.transpose(0, 1)     #action.item()

         
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        highest_prob_action = torch.argmax(probs, dim=1)
        return highest_prob_action.item()


        
    



#???
def soft_q_update(batch_size,gamma=0.99,soft_tau=1e-2,):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob = policy_net.evaluate(state)   #, epsilon, mean, log_std
    

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
    #print("Q Loss")
    #print(q_value_loss1)
    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    
    
    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    #print("V Loss")
    #print(value_loss)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # Update the target Value function parameters
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    return q_value_loss1.item() + value_loss.item() + policy_loss.item()





# TRAINING (?)



RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
env = make_env(scenario=Scenario.FLOOR_IS_LAVA, resolution=RESOLUTION, render=RENDER)

state_dim = env.observation_space.shape[0] *  env.observation_space.shape[1] *  env.observation_space.shape[2] * env.observation_space.shape[3]
action_dim = env.action_space.n
print(state_dim)
print(action_dim)



hidden_dim = 256

value_net = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()
lr  = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=lr)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)


replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


max_steps   = 500
episode = 0
episodes = 2
rewards     = []
losses = []
batch_size  = 1
window = 50



from statistics import mean


for episode in range(episodes):
    state, _ = env.reset()
    #state = state.flatten()

    state_array = np.array(state)
    state_flattened = state_array.flatten()

    episode_reward = 0
    losses_ep = []
    for step in range(max_steps):
        action = policy_net.get_action(state_flattened)
        next_state, reward, terminated, truncated, _= env.step(np.argmax(action))

        #next_state = next_state.flatten()

        #if truncated: print("!!! truncated")
        done = terminated

        # corrects 'LazyFrames object' error
        state_array = np.array(state)
        state_flattened = state_array.flatten()
        next_state_array = np.array(next_state)
        next_state_flattened = next_state_array.flatten()
        # action_array = np.array(action)
        # action_flattened = action_array.flatten()


        replay_buffer.push(state_flattened, action, reward, next_state_flattened, terminated)
        if len(replay_buffer) > batch_size:
            loss = soft_q_update(batch_size)
            losses_ep.append(loss)
        
        #print(reward)
        state = next_state
        episode_reward += reward
       
        
        if done:
            break

    
        
    episode += 1
    

    rewards.append(episode_reward)
    losses.append(mean(losses_ep))
    
    print("\rEpisode {:d}:  Total Reward = {:.2f}   Loss = {:.2f}\t\t".format(
                            episode, episode_reward, loss), end="")     # loss is the last loss of the episode






# test + render

from COOM.env.builder import make_env
from COOM.utils.config import Scenario


RESOLUTION = '1920x1080'
RENDER = True       # If render is true, resolution is always 1920x1080 to match my screen

done = False
tot_rew = 0
env = make_env(scenario=Scenario.FLOOR_IS_LAVA, resolution=RESOLUTION, render=RENDER)

# Initialize and use the environment
state, _ = env.reset()
state_array = np.array(state)
state_flattened = state_array.flatten()

for steps in range(1000):
    action = policy_net.get_action(state_flattened)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated    
    
    state_array = np.array(state)
    state_flattened = state_array.flatten()
    next_state_array = np.array(next_state)
    next_state_flattened = next_state_array.flatten()
    
    tot_rew += reward
    if done:
        break
    state_flattened = next_state_flattened


print("Tot reward in episode: ", tot_rew)
