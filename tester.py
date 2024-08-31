# test + render
import numpy as np
import torch
import pickle

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

import cv2
import os
import argparse

from bandit import Bandit

parser = argparse.ArgumentParser(description="SAC Training Configuration")

# Environment Config
parser.add_argument('--resolution', type=str, default='1920X1080', choices=['160X120', '1920X1080'], help="Screen resolution")
parser.add_argument('--render', type=bool, default=False, choices=[True, False], help="Render the environment")
parser.add_argument('--sequence', type=str, default='None', choices=['None', 'CO4', 'CO8', 'COC'], help="Sequence type")
parser.add_argument('--scenario', type=str, default='PITFALL', choices=['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING'], help="Scenario to run")
parser.add_argument('--save_log', type=bool, default=False, help="Save or not reward and videos in the model folder")

# Model and Save Path
parser.add_argument('--model', type=str, default='conv', help="Model type in the form of $DATE_$TIME_$MODELNAME")
parser.add_argument('--path', type=str, default='models/', help="Path where are trained models")

args = parser.parse_args()

# Test mode
RESOLUTION = args.resolution
RENDER = args.render       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = None if args.sequence=='None' else eval(f'Sequence.{args.sequence}')  #'Single' #Sequence.CO8             # SET TO 'Single' TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = eval(f'Scenario.{args.scenario}')
SAVE = args.save_log

# SAC type
MODEL = args.model
MODEL_PATH = os.path.join(args.path, args.model)  # creates a folder for each model trained
print(f'Loading from {MODEL_PATH}...')

# create test folder if not exists
if not os.path.exists(MODEL_PATH+'/test_results'):
    os.makedirs(MODEL_PATH+'/test_results')

num_heads = 1

if 'owl' in MODEL:
    num_heads = 8
    gamma = 0.99
    bandit = Bandit(12, 8) #pass all parameters needed 

if 'fc' in MODEL:
    from SAC.sac_fc import PolicyNetwork
elif 'conv' in MODEL:
    from SAC.sac_conv import PolicyNetwork

with open(f'{MODEL_PATH}/model.pkl', 'rb') as file:
    model = pickle.load(file)
    model.eval()

# use gpu
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# core function to test the model on one single scenario
def test_on_scenario(env):
    frame_list = [] # list of images to convert to video of the simulation

    state, _ = env.reset()
    #state = np.array(state)
    episode_reward = 0
    task = 0
    done = False
    state = torch.FloatTensor(np.array(state)).to(device)   # [1, 4, 84, 84, 3]
    while not done:
        img = env.render()
        #img = np.transpose(img.screen_buffer, [1, 2, 0]) #if state else np.uint8(np.zeros(self.game_res))
        frame_list.append(img[0])

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

    return episode_reward, frame_list




def create_and_save_video(image_list, output_path, scenario_index:int=None, fps=10):
    """
    Creates a video from a list of images stored as NumPy arrays.

    :param image_array_list: List of images as NumPy arrays.
    :param output_path: File path where the video will be saved.
    :param fps: Frames per second for the video. Default is 1.
    """
    scenario_list = ['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING']
    env_name = SCENARIO.name if scenario_index is None else scenario_list[scenario_index]
    video_name = env_name+'.avi' if scenario_index is None else SEQUENCE.name+'_'+env_name+'.avi'

    # Convert the first image to a NumPy array to get dimensions
    first_image = image_list[0]
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(output_path+'/'+video_name, fourcc, fps, (width, height))

    # Loop through the list of images and write them to the video
    for image in image_list:
        # Ensure the image is in the right format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    # Release the video writer
    video.release()
 
    print(f"\nVideo {video_name} saved at {output_path}")


# save rewards in a txt file (append to it if it already exists)
def save_reward(reward, output_path, file_name, scenario_index:int=None):   # when scenario_index is None, we are referring to a Single scenario and not a Sequence
    scenario_list = ['PITFALL', 'ARMS_DEALER', 'FLOOR_IS_LAVA', 'HIDE_AND_SEEK', 'CHAINSAW', 'RAISE_THE_ROOF','RUN_AND_GUN','HEALTH_GATHERING']
    env_name = SCENARIO if scenario_index is None else scenario_list[scenario_index]

    print(f'{env_name}: Reward {reward}')
    # Open the file in append mode ('a')
    with open(output_path+'/'+file_name+'.txt', 'a') as file:
        file.write(f'{env_name}: Reward {reward}\n')





if not(SEQUENCE):    # test on a single scenario
    env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)
    episode_reward, frame_list = test_on_scenario(env)
    print(f"{SCENARIO} - Reward: {episode_reward}")
    create_and_save_video(frame_list, MODEL_PATH+'/test_results')
    save_reward(episode_reward, MODEL_PATH+'/test_results', SCENARIO.name)



else:    # test on a Sequence
    done = False
    rewards_log = []
    count_env = 0

    cl_env = ContinualLearningEnv(sequence=SEQUENCE, resolution=RESOLUTION, render=RENDER) #, wrapper_config={'record':True, 'record_dir':MODEL_PATH}
    for env in cl_env.tasks:
        #env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: True) # Wrap the environment to record video
        episode_reward, frame_list = test_on_scenario(env)
        rewards_log.append(episode_reward)
        create_and_save_video(frame_list, MODEL_PATH+'/test_results', scenario_index=count_env)
        save_reward(episode_reward, MODEL_PATH+'/test_results', SEQUENCE.name, scenario_index=count_env)
        count_env += 1
        
    
    total_reward = sum(rewards_log)
    print(f"{SEQUENCE} - Cumulative Reward: {total_reward}")


