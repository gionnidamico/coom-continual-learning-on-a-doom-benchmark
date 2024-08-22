# test + render
import numpy as np
import torch
import pickle

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

from bandit import Bandit

# Test mode
RESOLUTION = '1920x1080'
RENDER = True       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE =  'Single'        #'Single', 'CO8' : define on which sequence you would like to test 
SCENARIO = Scenario.RUN_AND_GUN

# SAC type
MODEL = 'owl conv'

SAVE_PATH = 'models/'

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

with open(f'{SAVE_PATH}model_{MODEL}.pkl', 'rb') as file:
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
    if 'owl' in MODEL:
        task = bandit.get_head()
    while not done:
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

        print(action.item())

        state = next_state
        episode_reward += reward.item()

        if 'owl' in MODEL:
            next_actions = model.sample_action(state, batched=False, deterministic=True, task=task) 
            next_actions_probs = model(state, batched=False, head=task)
            task = bandit.update(next_actions, next_actions_probs, action, action_probs, reward, done, gamma, device)

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





