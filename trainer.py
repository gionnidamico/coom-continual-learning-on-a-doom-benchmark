from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from ReplayBuffers.replay_buffer import ReplayBuffer

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

import pickle

# use gpu
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# SAC type
MODEL = 'conv'

SAVE_PATH = 'models/'

# train config
RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = 'Single' #Sequence.CO8             # SET TO 'Single' TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = Scenario.RUN_AND_GUN

# the SAC Algorithm

# Initialize Networks and Optimizers
hidden_dim = 256
batch_size = 32

replay_buffer = ReplayBuffer(capacity=1000000)

if MODEL == 'fc':
    from SAC.sac_fc import ValueNetwork, QNetwork, PolicyNetwork
    state_dim = 4*84*84*3  # Example state dimension
    num_actions = 12  # Example number of actions
elif MODEL == 'conv':
    from SAC.sac_conv import ValueNetwork, QNetwork, PolicyNetwork
    num_channels = 3
    state_dim = num_channels  # Example state dimension
    num_actions = 12  # Example number of actions

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
    # Reduce next_value to match value's shape
    next_value = next_value.mean(dim=-1, keepdim=True)
    #print(f"values:{value.shape}, {next_value.shape}")
    return F.mse_loss(value, next_value)

def q_loss(state, action, reward, next_state, done, q_net, target_value_net):
    with torch.no_grad():
        next_value = target_value_net(next_state)
        target_q = reward + (1 - done) * next_value
    q_values = q_net(state)
    q_value = q_values.gather(1, action)   # get the Nth q_value that corresponds to the action taken (second dimension of action tensor)
    #print(f"q_values shape: {q_value.shape}, {target_q.shape}")
    return F.mse_loss(q_value, target_q)   

def policy_loss(state, q_net, policy_net):
    probs = policy_net(state)
    q_values = q_net(state)
    log_probs = torch.log(probs + 1e-8)
    policy_loss = (probs * (log_probs - q_values.detach())).sum(dim=-1).mean()
    return policy_loss


# Training Step
def update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer):
    
    # Flatten the state to fit in fully connected network
    #state_array = np.array(state)
    # state_flattened = state.flatten()  # [batch_size, 4*84*84*3]
    # print(f'Tensor here is {state_flattened.shape}')

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





# Step 4: Create and Train in an Environment

num_episodes = 3
gamma = 0.99

print("\nTraining STARTING...")

# choose between 'SINGLE' and 'SEQUENCE'
if SEQUENCE == 'Single':    # train on single scenario
    env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)

for episode in range(num_episodes):
    state, _ = env.reset()
    #state = np.array(state)
    i=0
    episode_reward = 0
    done = False
    while not done:
        # i+=1
        # print(i)
        state = torch.FloatTensor(np.array(state))#.unsqueeze(0)   # [1, 4, 84, 84, 3]
        action = policy_net.sample_action(state, batched=False)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(np.array(next_state))#.unsqueeze(0)
        reward = torch.FloatTensor([reward])#.unsqueeze(1)
        done = torch.FloatTensor([done])#.unsqueeze(1)
        action = torch.LongTensor([action])#.unsqueeze(0)

        #print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
         # Within your training loop, after taking a step in the environment, add this line:
        replay_buffer.push(state, action, reward, next_state, done)

        # Existing code for taking a step
        # update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)

        # Replace the above line with logic that checks if enough samples are in the buffer before updating
        if len(replay_buffer) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)    ### single batch
            #print(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)
            #update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)

        state = next_state
        episode_reward += reward.item()

    print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Save the trained model in a file
    with open(f'{SAVE_PATH}model_{MODEL}.pkl', 'wb') as file:
        pickle.dump(policy_net, file)
