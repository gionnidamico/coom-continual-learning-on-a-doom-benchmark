from statistics import mean
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from ReplayBuffers.replay_buffer import ReplayBuffer
from Regularizators.ewc import ewc

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

import pickle

# use gpu
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# SAC type
MODEL = 'owl conv'

SAVE_PATH = 'models/'

# train config
RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = Sequence.CO8 #Sequence.CO8             # SET TO 'Single' TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = Scenario.RUN_AND_GUN

REGULARIZATION = 'ewc'

# train params
num_episodes = 10
gamma = 0.99

# the SAC Algorithm

# Initialize Networks and Optimizers
hidden_dim = 128
batch_size = 32
num_heads = 1
num_actions = 12  # Example number of actions


if 'fc' in MODEL:
    from SAC.sac_fc import ValueNetwork, QNetwork, PolicyNetwork
    state_dim = 4*84*84*3  # Example state dimension
elif 'conv' in MODEL:
    from SAC.sac_conv import ValueNetwork, QNetwork, PolicyNetwork
    num_channels = 3
    state_dim = num_channels  # Example state dimension
if 'owl' in MODEL:
    from SAC.sac_conv import ValueNetwork, QNetwork, PolicyNetwork
    num_heads = 8


replay_buffers = [ReplayBuffer(capacity=1000, device=device) for _ in range(num_heads)]

value_net = ValueNetwork(state_dim, hidden_dim).to(device)
q_net1 = QNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)
q_net2 = QNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)
policy_net = PolicyNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)

value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
q_optimizer1 = optim.Adam(q_net1.parameters(), lr=1e-4)
q_optimizer2 = optim.Adam(q_net2.parameters(), lr=1e-4)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)


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

def q_loss(state, action, reward, next_state, done, q_net, target_value_net, reg, task):
    with torch.no_grad():
        next_value = target_value_net(next_state)
        target_q = reward + (1 - done) * next_value
    q_values = q_net(state, head = task)
    q_value = q_values.gather(1, action)   # get the Nth q_value that corresponds to the action taken (second dimension of action tensor)
    #print(f"q_values shape: {q_value.shape}, {target_q.shape}")

    reg_weights = 0 #if no reg then no weight
    if reg:
        reg_weights = reg.get_weight_q(q_value, q_net)
    '''
    print("q")
    print("loss:",F.mse_loss(q_value, target_q))
    print("reg:",reg_weights)
    '''
    return F.mse_loss(q_value, target_q) + reg_weights

def policy_loss(state, q_net, policy_net, reg, task):
    state = state.detach().requires_grad_()
    probs = policy_net(state, head = task)
    q_values = q_net(state, head = task)
    log_probs = torch.log(probs + 1e-8)
    policy_loss = (probs * (log_probs - q_values.detach())).sum(dim=-1).mean()

    reg_weights = 0 #if no reg then no weight
    if reg:
        reg_weights = reg.get_weight_policy(probs, policy_net)
    '''
    print("policy")
    print("loss:",policy_loss)
    print("reg:",reg_weights)
    '''
    return policy_loss + reg_weights


# Training Step
def update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer, task):
    
    # Flatten the state to fit in fully connected network
    #state_array = np.array(state)
    # state_flattened = state.flatten()  # [batch_size, 4*84*84*3]
    # print(f'Tensor here is {state_flattened.shape}')

    # Update Value Network
    reg = None
    if REGULARIZATION == 'ewc' :
        reg = ewc()
    

    value_optimizer.zero_grad()
    v_loss = value_loss(state, q_net1, q_net2, value_net)
    v_loss.backward()
    value_optimizer.step()

    # Update Q Networks
    q_optimizer1.zero_grad()
    q_optimizer2.zero_grad()
    q_loss1 = q_loss(state, action, reward, next_state, done, q_net1, value_net, reg, task)
    q_loss2 = q_loss(state, action, reward, next_state, done, q_net2, value_net, reg, task)
    q_loss1.backward()
    q_loss2.backward()
    q_optimizer1.step()
    q_optimizer2.step()

    # Update Policy Network
    policy_optimizer.zero_grad()
    p_loss = policy_loss(state, q_net1, policy_net, reg, task)
    p_loss.backward()
    policy_optimizer.step()





# Create and Train in an Environment

# Train on a scenario until the end
def train_on_scenario(env, task = 0):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    state = torch.FloatTensor(np.array(state))#.to(device)   # [1, 4, 84, 84, 3]

    while not done:
        # state not included here because it's already done before for the first iteration, and will be done for nexe_state for each iteration after
        action = policy_net.sample_action(state.to(device), task, batched=False, deterministic=False)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(np.array(next_state))#.to(device)
        reward = torch.FloatTensor([reward])#.to(device)
        done = torch.FloatTensor([done])#.to(device)
        action = torch.LongTensor([action])#.to(device)

        if MODEL == 'fc':
            state = state.view(-1)
            next_state = next_state.view(-1)

        #print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        replay_buffers[task].push(state, action, reward, next_state, done)

        # Replace the above line with logic that checks if enough samples are in the buffer before updating
        if len(replay_buffers[task]) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffers[task].sample(batch_size)    ### single batch
            #print(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer, task)
            #update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)
            
            #del state_batch, action_batch, reward_batch, next_state_batch, done_batch   # free up gpu space 
        state = next_state
        episode_reward += reward.item()


    return episode_reward



###############################################################################
print("\nTraining STARTING...")


# choose between 'SINGLE' and 'SEQUENCE'


episodes_reward = []

if SEQUENCE == 'Single':    # train on single scenario
    env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)
    for episode in range(num_episodes):
        episode_reward = train_on_scenario(env, task = 0)
        episodes_reward.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}")


else:
    done = False
    tot_reward = 0
    cl_env = ContinualLearningEnv(SEQUENCE)
    for episode in range(num_episodes):
        task = 0
        for env in cl_env.tasks:
            for _ in range(2):
                episode_reward = train_on_scenario(env, task)
                episodes_reward.append(episode_reward)
                tot_reward += episode_reward
                '''
                if 'owl' in MODEL: # this is to empty the buffer when switching head 
                    # we could create RP for each task not to lose experience (maybe we could use the priority for that)
                    replay_buffer = ReplayBuffer(capacity=1000000, device=device)
                '''
            if 'owl' in MODEL:
                task += 1
            print(f"Episode {episode+1}, Reward: {episode_reward}")




# Save the trained model in a file
with open(f'{SAVE_PATH}model_{MODEL}.pkl', 'wb') as file:
    pickle.dump(policy_net, file)
