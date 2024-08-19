from statistics import mean
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence


# use gpu
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

# train config
RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = Sequence.CO8             # SET TO NONE TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = Scenario.RUN_AND_GUN


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
#######################
class ValueNetwork(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_dim, num_actions, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_actions)
        
        # Weight initialization
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

        

# critic network 
###########################     
class SoftQNetwork(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_size, num_actions, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_actions)
        
        # Weight initialization
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.output_layer(x)
        return q_values
       
        

# soft AC agent network    
######################    
class PolicyNetwork(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_size, num_actions, device, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.device = device
        
        self.target_entropy = -np.prod(self.num_actions)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=3e-4)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Convolutional layers with specified strides
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 2), stride=(2, 2, 1), padding=(1, 1, 0))

        # Fully connected layers
        self.flatten_dim = 64 * 10 * 10 * 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.output_layer_mean = nn.Linear(hidden_size, num_actions)
        self.output_layer_log_std = nn.Linear(hidden_size, num_actions)

        # Weight initialization
        self.output_layer_mean.weight.data.uniform_(-init_w, init_w)
        self.output_layer_mean.bias.data.uniform_(-init_w, init_w)
        self.output_layer_log_std.weight.data.uniform_(-init_w, init_w)
        self.output_layer_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, self.log_std_min, self.log_std_max)

        return x_mean, x_log_std

    
    def evaluate(self, state, epsilon=1e-6):

        mean, log_std = self.forward(state)
        # Get a Normal distribution with those values
        pi_s = Normal(mean, log_std.exp())
        # reparameterization trick.
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log(
        (1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
    
        return action, log_prob 


       
         
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        probs = self.forward(state)[0]  # get mean action
        highest_prob_action = torch.argmax(probs, dim=1)
        return highest_prob_action.item()
       

        
    




def soft_q_update(batch_size,gamma=0.99,soft_tau=1e-2,):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # current qsa_a and qsa_b
    predicted_q_value1 = soft_q_net1(state, action)     # action not used
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    current_actions, logpi_s = policy_net.evaluate(state)   #, epsilon, mean, log_std
    target_alpha = (logpi_s + policy_net.target_entropy).detach()
    alpha_loss = -(policy_net.logalpha * target_alpha).mean()

    # loss of alpha, and here we step alpha’s optimizer.
    policy_net.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    policy_net.alpha_optimizer.step()

    alpha = policy_net.logalpha.exp()

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())   
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    
    
    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, current_actions), soft_q_net2(state, current_actions))

    target_value_func = predicted_new_q_value - alpha*logpi_s

    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Training Policy Function
    policy_loss = (alpha*logpi_s - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # Update the target Value function parameters           # TO DO: CHECK FOR CONSISTENCY IN VALUE NETWORKS
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    return q_value_loss1.item() + value_loss.item() + policy_loss.item()



if __name__ == "__main__":
    # This code will only run when the script is executed directly




    # TRAINING (?)


    # NEURAL NETWORKS INIT + ALL
           
    # STATE AND ACTION DEFINITION (hard-coded, always the same between scenarios)
    state_dim = (4, 84, 84, 3)   # 84x84x3chx4img
    action_dim = 12

    hidden_dim = 256

    value_net = ValueNetwork(num_frames=4, num_actions=action_dim, num_channels=3, hidden_dim=hidden_dim).to(device)
    target_value_net = ValueNetwork(num_frames=4, num_actions=action_dim, num_channels=3, hidden_dim=hidden_dim).to(device)

    soft_q_net1 = SoftQNetwork(num_frames=4, num_actions=action_dim, num_channels=3, hidden_size=hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(num_frames=4, num_actions=action_dim, num_channels=3, hidden_size=hidden_dim).to(device)

    policy_net = PolicyNetwork(num_frames=4, num_actions=action_dim, num_channels=3, hidden_size=hidden_dim, device=device).to(device) # vedi spostamento su device

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


    max_steps   = 10000       # change
    episode = 0
    episodes = 3
    rewards     = []
    losses = []
    batch_size  = 16
    #window = 50
    (3, ...)
 

    # choose between 'SINGLE' and 'SEQUENCE'
    if SEQUENCE is None:    # train on single scenario
        env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)


        print("\nTraining STARTING...")

        for episode in range(episodes):
            state, _ = env.reset()
            state = state.permute(0, 1, 4, 2, 3)

            episode_reward = 0
            losses_ep = []
            for step in range(max_steps):

                if len(replay_buffer) <= batch_size:
                    action = policy_net.get_action(torch.FloatTensor(state).unsqueeze(0)) #to(device) ?

                next_state, reward, done, truncated, _= env.step(action)
                next_state = torch.FloatTensor(np.array(next_state))
                next_state = next_state.permute(3, 0, 1, 2)

                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) > batch_size:
                    loss = soft_q_update(batch_size)
                    losses_ep.append(loss)
                
                #print(reward)
                state = next_state
                episode_reward += reward
            
                if done or truncated:  # go to next episode
                    break        
                
            

            rewards.append(episode_reward)
            losses.append(mean(losses_ep))
            
            print("\nEpisode {:d}:  Total Reward = {:.2f}   Loss = {:.2f}".format(episode+1, episode_reward, loss), end="")     # loss is the last loss of the episode
            
            
            episode += 1


        print("\nTraining COMPLETED.")



    else: # train on a sequence

        done = False
        tot_rew = 0
        cl_env = ContinualLearningEnv(SEQUENCE)
        for env in cl_env.tasks:
            # Initialize and use the environment
            state, _ = env.reset()
            state = torch.FloatTensor(np.array(state))
            state = state.permute(3, 1, 2, 0)

            print("\nTraining STARTING...")

            for episode in range(episodes):
                state, _ = env.reset()
                state = torch.FloatTensor(np.array(state))
                state = state.permute(3, 1, 2, 0)

                episode_reward = 0
                losses_ep = []
                for step in range(max_steps):

                    if len(replay_buffer) <= batch_size:
                        action = policy_net.get_action(state.unsqueeze(0)) #to(device) ?

                    next_state, reward, done, truncated, _= env.step(action)

                    next_state = torch.FloatTensor(np.array(next_state))
                    next_state = next_state.permute(3, 1, 2, 0)

                    replay_buffer.push(state, action, reward, next_state, done)
                    if len(replay_buffer) > batch_size:
                        loss = soft_q_update(batch_size)
                        losses_ep.append(loss)
                    
                    #print(reward)
                    state = next_state
                    episode_reward += reward
                
                    if done or truncated:  # go to next episode
                        break        
                    
                

                rewards.append(episode_reward)
                losses.append(mean(losses_ep))
                
                print("\nEpisode {:d}:  Total Reward = {:.2f}   Loss = {:.2f}".format(episode+1, episode_reward, loss), end="")     # loss is the last loss of the episode
                
                
                episode += 1


            print("\nTraining COMPLETED.")



    # Save the trained model to a file
    with open('model_conv.pkl', 'wb') as file:
        pickle.dump(policy_net, file)





