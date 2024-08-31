from statistics import mean
import torch
import torch.optim as optim
import torch.nn.functional as F

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

import numpy as np
import pickle
import argparse
from datetime import datetime
import os
#from tqdm import tqdm

############################################################################
'TRAINING CONFIGURATION'
############################################################################

# use gpu
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser(description="SAC Training Configuration")

# Model and Save Path
parser.add_argument('--model', type=str, default='conv', help="Model type")
parser.add_argument('--path', type=str, default='models/', help="Path to save the trained models")

# Environment Config
parser.add_argument('--resolution', type=str, default='160X120', choices=['160X120', '1920X1080'], help="Screen resolution")
parser.add_argument('--render', type=bool, default=False, choices=[True, False], help="Render the environment")
parser.add_argument('--sequence', type=str, default='CO8', choices=['None','CO8', 'COC'], help="Sequence type")
parser.add_argument('--scenario', type=str, default='PITFALL', choices=['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING'], help="Scenario to run")
parser.add_argument('--repeat_scenarios', type=int, default=1, help="Repeat N times each scenario in the sequence for each episode")
parser.add_argument('--skip', type=str, default='None', choices=['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING', 'None'], help="Scenario to skip, if any")

# Regularization and PER
parser.add_argument('--reg', type=str, default='None', choices=['None','ewc', 'mas'], help="Type of regularization to use")
parser.add_argument('--use_per', type=bool, default=False, choices=[True, False], help="Use Prioritized Experience Replay (PER) Buffer")
parser.add_argument('--use_multy_per', type=bool, default=True, choices=[True, False], help="Use one Experience Replay Buffer per head")
parser.add_argument('--size_rb', type=int, default=1000, help="size of Experience Replay Buffer")

# Training Parameters
parser.add_argument('--episodes', type=int, default=3, help="Number of episodes for training")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
parser.add_argument('--tau', type=float, default=0.005, help="soft update parameter")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for networks and optimizers")

args = parser.parse_args()
scenarios = ['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING']

# Clear cache
torch.cuda.empty_cache()

# SAC type and path
MODEL_NAME = args.model
SAVE_PATH = args.path + '/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_'+args.model+'_'+args.reg + '/' # creates a folder for each model trained
os.makedirs(SAVE_PATH, exist_ok=True)

# train config
RESOLUTION = args.resolution
RENDER = args.render       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = None if args.sequence=='None' else eval(f'Sequence.{args.sequence}')  #'Single' #Sequence.CO8             # SET TO 'Single' TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = eval(f'Scenario.{args.scenario}')
REPETITION_PER_SCENARIO = args.repeat_scenarios
INDEX_TO_SKIP = -1 if args.skip=='None' else scenarios.index(args.skip)

# train params
EPISODES = args.episodes
REGULARIZATION = None if args.reg=='None' else args.reg
USE_PER = args.use_per # if false, use the vanilla Replay Buffer instead
GAMMA = args.gamma
TAU = args.tau
LEARNING_RATE = args.lr

# Plots loss/rewards over time and saves the plot as a PNG file 
import matplotlib.pyplot as plt
def plot_and_save(time_serie, title, label_name, path):
    print(f'{label_name}: {time_serie}')
    plt.figure(figsize=(10, 6))
    plt.plot(time_serie, label=label_name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(label_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


################################################################################
'SAC Algorithm'
################################################################################

# Initialize Networks and Optimizers
hidden_dim = 256
batch_size = 32
num_heads = 1
num_actions = 12  # Example number of actions

# Choose the model to use
if 'fc' in MODEL_NAME:
    from SAC.sac_fc import ValueNetwork, QNetwork, PolicyNetwork
    state_dim = 4*84*84*3  # Example state dimension
elif 'conv' in MODEL_NAME:
    from SAC.sac_conv import ValueNetwork, QNetwork, PolicyNetwork
    num_channels = 3
    state_dim = num_channels  # Example state dimension
if 'owl' in MODEL_NAME:
    num_heads = 8

# Choose the regularizator if one is chosen
reg = None
if REGULARIZATION == 'ewc':
    from Regularizators.ewc import ewc
    reg = ewc()
elif REGULARIZATION == 'mas':
    from Regularizators.mas import mas
    reg = mas()

if args.use_multy_per:
    num_PER = num_heads
else:
    num_PER = 1

# Choose the replay buffer to use
if USE_PER:
    from ReplayBuffers.prioritized_RB import PrioritizedReplayBuffer
    replay_buffers = [PrioritizedReplayBuffer(capacity=args.size_rb, device=device) for _ in range(num_PER)]
else:
    from ReplayBuffers.replay_buffer import ReplayBuffer
    replay_buffers = [ReplayBuffer(capacity=args.size_rb, device=device) for _ in range(num_PER)]

# Define the networks to use and the optimizers
value_net = ValueNetwork(state_dim, hidden_dim).to(device)
q_net1 = QNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)
q_net2 = QNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)
policy_net = PolicyNetwork(state_dim, num_actions, hidden_dim, n_head = num_heads).to(device)

value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
q_optimizer1 = optim.Adam(q_net1.parameters(), lr=LEARNING_RATE)
q_optimizer2 = optim.Adam(q_net2.parameters(), lr=LEARNING_RATE)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

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

def q_loss(state, action, reward, next_state, done, gamma, q_net, target_value_net, reg, task):
    with torch.no_grad():
        next_value = target_value_net(next_state)
        target_q = reward + gamma * (1 - done) * next_value
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

    return policy_loss + reg_weights


# Training Step
def update(state, action, reward, next_state, done, gamma, tau, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer, task, reg=None):
    
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
    q_loss1 = q_loss(state, action, reward, next_state, done, gamma, q_net1, value_net, reg, task)
    q_loss2 = q_loss(state, action, reward, next_state, done, gamma, q_net2, value_net, reg, task)
    q_loss1.backward()
    q_loss2.backward()
    q_optimizer1.step()
    q_optimizer2.step()

    # Update Policy Network
    policy_optimizer.zero_grad()
    p_loss = policy_loss(state, q_net1, policy_net, reg, task)
    p_loss.backward()
    policy_optimizer.step()

    for target_param, current_param in zip(q_net2.parameters(), q_net1.parameters()):
        target_param.data.copy_(tau * current_param.data + (1 - tau) * target_param.data)

    return p_loss.detach().cpu()





# Create and Train in an Environment

# Train on a scenario until the end
def train_on_scenario(env, task = 0):
    
    episode_reward = 0
    episode_policyloss = 0
    policy_loss = 0

    state, _ = env.reset()
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

        if MODEL_NAME == 'fc':   # flatten the states
            state = state.view(-1)
            next_state = next_state.view(-1)

        #print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        replay_buffers[task].push(state, action, reward, next_state, done)

        # Replace the above line with logic that checks if enough samples are in the buffer before updating
        if len(replay_buffers[task]) > batch_size:
            if USE_PER:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices = replay_buffers[task].sample(batch_size)    ### single batch
            else:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffers[task].sample(batch_size)    ### single batch
            #print(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            
            policy_loss = update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, GAMMA, TAU, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer, task, reg)
            #update(state, action, reward, next_state, done, value_net, q_net1, q_net2, policy_net, value_optimizer, q_optimizer1, q_optimizer2, policy_optimizer)
            
            if USE_PER:
               with torch.no_grad():
                 next_value = value_net(next_state_batch)        # target value network
                 target_q = reward_batch + (1 - done_batch) * next_value
                 q_values = q_net1(state_batch, head = task)
                 q_value = q_values.gather(1, action_batch)  
               td_errors = torch.abs(target_q - q_value).cpu().numpy()#.squeeze()     # temporal difference error between target value and current q value
               replay_buffers[task].update_priorities(td_errors, indices)
            
            
            #del state_batch, action_batch, reward_batch, next_state_batch, done_batch   # free up gpu space 
        state = next_state
        episode_reward += reward.item()
        episode_policyloss += policy_loss


    return episode_reward, episode_policyloss



###############################################################################
print("\nTraining STARTING...")
###############################################################################
# choose between 'SINGLE' and 'SEQUENCE'

episodes_reward = []
episodes_policyloss = []


if SEQUENCE == None:    # train on single scenario
    env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)
    for episode in range(EPISODES):
        episode_reward, episode_policyloss = train_on_scenario(env, task = 0)
        episodes_reward.append(episode_reward)
        episodes_policyloss.append(episode_policyloss if type(episode_policyloss) == int else episode_policyloss.cpu().detach())
        print(f"Episode {episode+1}, Reward: {episode_reward}")


else:                   # else train on a sequence
    done = False
    tot_reward = 0
    cl_env = ContinualLearningEnv(SEQUENCE)
    for episode in range(EPISODES):
        count_env = 0
        task = 0
        for env in cl_env.tasks:

            # if you want to skip a scenario... 
            if count_env == INDEX_TO_SKIP:
                count_env += 1
                if 'owl' in MODEL_NAME:     # make sure to update the task head if 'owl'
                    task += 1
                continue
            else:
                count_env += 1

            # ...otherwise continue as usual
            for _ in range(REPETITION_PER_SCENARIO): #   repeat EACH scenario in a 
                episode_reward, episode_policyloss = train_on_scenario(env, task)   # task is a counter only useful for owl to select the correct replay buffer 
                episodes_reward.append(episode_reward)
                episodes_policyloss.append(episode_policyloss)
                tot_reward += episode_reward
                '''
                if 'owl' in MODEL: # this is to empty the buffer when switching head 
                    # we could create RP for each task not to lose experience (maybe we could use the priority for that)
                    replay_buffer = ReplayBuffer(capacity=1000000, device=device)
                '''
            if 'owl' in MODEL_NAME:
                task += 1
        print(f"Episode {episode+1}, Cumulative Reward: {episode_reward}")





# Save the trained model in a file, with plots and parameters
with open(f'{SAVE_PATH}/model.pkl', 'wb') as file:
    pickle.dump(policy_net, file)

plot_and_save(episodes_reward, f'Training of {MODEL_NAME} - Reward', 'Reward', f'{SAVE_PATH}reward.png')
plot_and_save(np.array(episodes_policyloss), f'Training of {MODEL_NAME} - Policy Loss', 'Reward', f'{SAVE_PATH}policyloss.png')


# save parameters on file
with open(f'{SAVE_PATH}params.txt', 'w') as file:
    for key, value in vars(args).items():
        file.write(f'{key}: {value}\n')
