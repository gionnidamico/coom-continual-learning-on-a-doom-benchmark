# test + render
import numpy as np
import torch
import pickle

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

import os
import argparse

from bandit import Bandit

parser = argparse.ArgumentParser(description="SAC Training Configuration")

# Environment Config
parser.add_argument('--resolution', type=str, default='160X120', choices=['160X120', '1920X1080'], help="Screen resolution")
parser.add_argument('--render', type=bool, default=False, choices=[True, False], help="Render the environment")
parser.add_argument('--sequence', type=str, default='CO8', choices=['None', 'CO4', 'CO8', 'COC'], help="Sequence type")
parser.add_argument('--scenario', type=str, default='PITFALL', choices=['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING'], help="Scenario to run")

# Model and Save Path
parser.add_argument('--model', type=str, default='conv', help="Model type")
parser.add_argument('--path', type=str, default='models/', help="Path to save the trained models")

args = parser.parse_args()

# Test mode
RESOLUTION = args.resolution
RENDER = args.render       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = None if args.sequence=='None' else eval(f'Sequence.{args.sequence}')  #'Single' #Sequence.CO8             # SET TO 'Single' TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = eval(f'Scenario.{args.scenario}')

# SAC type
MODEL = args.model
SAVE_PATH = os.path.join(args.path, '/'+args.model + args.sequence)  # creates a folder for each model trained

num_heads = 1

if 'owl' in MODEL:
    from SAC.sac_conv import PolicyNetwork
    num_heads = 8
    gamma = 0.99
    bandit = Bandit(12, 8) #pass all parameters needed 

if 'fc' in MODEL:
    from SAC.sac_fc import PolicyNetwork
elif 'conv' in MODEL:
    from SAC.sac_conv import PolicyNetwork

with open(f'{SAVE_PATH}model.pkl', 'rb') as file:
    model = pickle.load(file)
    model.eval()

# use gpu
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)


if SEQUENCE == 'Single':    # train on single scenario
    env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)


    state, _ = env.reset()
    #state = np.array(state)
    episode_reward = 0
    task = 0
    done = False
    state = torch.FloatTensor(np.array(state)).to(device)   # [1, 4, 84, 84, 3]
    while not done:
        if 'owl' in MODEL:
            task = bandit.get_head()
        action = model.sample_action(state, batched=False, deterministic=True, task=task) 
        action_probs = model(state, batched=False, head=task)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)#.unsqueeze(0)
        reward = torch.FloatTensor([reward]).to(device)#.unsqueeze(1)
        done = torch.FloatTensor([done]).to(device)#.unsqueeze(1)
        action = torch.LongTensor([action]).to(device)#.unsqueeze(0)

        if MODEL == 'fc':
            state = state.view(-1)
            next_state = next_state.view(-1)

        #print(action.item())
        state = next_state
        episode_reward += reward.item()

        if 'owl' in MODEL:
            next_actions = model.sample_action(state, batched=False, deterministic=True, task=task) 
            next_actions_probs = model(state, batched=False, head=task)
            bandit.update(next_actions, next_actions_probs, action, action_probs, reward, done, gamma, device)

    print(f"{SCENARIO} - Reward: {episode_reward}")

'''
elif SEQUENCE == 'CO8':

    done = False
    tot_rew = 0
    cl_env = ContinualLearningEnv(Sequence.CO8)
    cl_env.render()
    for env in cl_env.tasks:
        # Initialize and use the environment
        state, _ = env.reset()
        state_array = np.array(state)
        state_flattened = state_array.flatten()

        for steps in range(1000):
            action = model.get_action(torch.FloatTensor(state_flattened).unsqueeze(0)) #to(device) ?
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated    
            
            state_array = np.array(state)
            state_flattened = state_array.flatten()
            next_state_array = np.array(next_state)
            next_state_flattened = next_state_array.flatten()
            
            tot_rew += reward
            #print(action)
            if done:
                break
            state_flattened = next_state_flattened

print("Total reward: ", tot_rew)
'''





