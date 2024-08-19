# test + render
import numpy as np
import torch
import pickle

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

from soft_AC_conv_vnlin import PolicyNetwork # for pickle load

RESOLUTION = '1920x1080'
RENDER = True       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE =  'Single'        #'Single', 'CO8' : define on which sequence you would like to test 


with open('model_conv.pkl', 'rb') as file:
    model = pickle.load(file)



if SEQUENCE == 'Single':
    done = False
    tot_rew = 0
    env = make_env(scenario=Scenario.FLOOR_IS_LAVA, resolution=RESOLUTION, render=RENDER)



    # Initialize and use the environment
    state, _ = env.reset()
    state = torch.FloatTensor(np.array(state))
    state = state.permute(3, 1, 2, 0)

    for steps in range(1000):
        action = model.get_action(torch.FloatTensor(state).unsqueeze(0)) #to(device) ?
        print(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated    
        
        next_state = torch.FloatTensor(np.array(next_state))
        next_state = next_state.permute(3, 1, 2, 0)
        
        tot_rew += reward
        #print(action)
        if done:
            break
        state = next_state


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


