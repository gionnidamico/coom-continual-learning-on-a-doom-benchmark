from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from replay_buffer import ReplayBuffer

from COOM.env.builder import make_env
from COOM.utils.config import Scenario

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence
import pickle

# use gpu
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

# train config
RESOLUTION = '160X120'
RENDER = False       # If render is true, resolution is always 1920x1080 to match my screen
SEQUENCE = Sequence.CO8             # SET TO NONE TO RUN THE SINGLE SCENARIO (set it on next line)
SCENARIO = Scenario.RUN_AND_GUN

class trainer_sac():
    def __init__(self, mode = 'fc', regularization = 'none'):
        super()
        if mode == 'fc':
            from SAC.soft_AC_fc import ValueNetwork, SoftQNetwork, PolicyNetwork
        elif mode == 'conv':
            from SAC.soft_AC_conv import ValueNetwork, SoftQNetwork, PolicyNetwork
        elif mode == 'conv_vnlin':
            from SAC.soft_AC_conv_vnlin import ValueNetwork, SoftQNetwork, PolicyNetwork
        # NEURAL NETWORKS INIT + ALL
            
        # STATE AND ACTION DEFINITION (hard-coded, always the same between scenarios)
        state_dim = 84672
        action_dim = 12

        hidden_dim = 256

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)

        self.mode = mode
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
            

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        lr  = 3e-4

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)


        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.regularization = regularization

    def soft_q_update(self, batch_size, replay_buffer, gamma=0.99, soft_tau=1e-2):
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        #print(state.shape)

        # current qsa_a and qsa_b
        predicted_q_value1 = self.soft_q_net1(state, action)     # action not used
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value    = self.value_net(state)
        current_actions, logpi_s = self.policy_net.evaluate(state)   #, epsilon, mean, log_std
        target_alpha = (logpi_s + self.policy_net.target_entropy).detach()
        alpha_loss = -(self.policy_net.logalpha * target_alpha).mean()

        # loss of alpha, and here we step alpha’s optimizer.
        self.policy_net.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_net.alpha_optimizer.step()

        alpha = self.policy_net.logalpha.exp()

        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())   
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        #print("Q Loss")
        #print(q_value_loss1)
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()    
        
        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, current_actions), self.soft_q_net2(state, current_actions))
        #print(predicted_new_q_value.shape, log_prob.shape)
        target_value_func = predicted_new_q_value - alpha*logpi_s

        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
        #print("V Loss")
        #print(value_loss)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Training Policy Function
        policy_loss = (alpha*logpi_s - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update the target Value function parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return q_value_loss1.item() + value_loss.item() + policy_loss.item()


    def train(self):

        max_steps   = 10000       # change
        episode = 0
        episodes = 3
        rewards     = []
        losses = []
        batch_size  = 16
        #window = 50


    

        # choose between 'SINGLE' and 'SEQUENCE'
        if SEQUENCE is None:    # train on single scenario
            env = make_env(scenario=SCENARIO, resolution=RESOLUTION, render=RENDER)

            print("\nTraining STARTING...")

            for episode in range(episodes):
                state, _ = env.reset()
                if self.mode == 'fc':
                    state_array = np.array(state)
                    state_flattened = state_array.flatten()
                else:
                    state = state.permute(0, 1, 4, 2, 3)

                episode_reward = 0
                losses_ep = []
                for step in range(max_steps):

                    if len(self.replay_buffer) <= batch_size:
                        action = self.policy_net.get_action(torch.FloatTensor(state_flattened).unsqueeze(0)) #to(device) ?

                    next_state, reward, done, truncated, _= env.step(action)

                    if self.mode == 'fc':
                        # corrects 'LazyFrames object' error
                        state_array = np.array(state)
                        state_flattened = state_array.flatten()
                        next_state_array = np.array(next_state)
                        next_state_flattened = next_state_array.flatten()
                        # action_array = np.array(action)
                        # action_flattened = action_array.flatten()
                    else:
                        next_state = torch.FloatTensor(np.array(next_state))
                        next_state = next_state.permute(3, 0, 1, 2)

                    self.replay_buffer.push(state_flattened, action, reward, next_state_flattened, done)
                    if len(self.replay_buffer) > batch_size:
                        loss = self.soft_q_update(batch_size, self.replay_buffer)
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
                if self.mode == 'fc':
                    state_array = np.array(state)
                    state_flattened = state_array.flatten()
                else:
                    state = state.permute(0, 1, 4, 2, 3)

                


                print("\nTraining STARTING...")

                for episode in range(episodes):
                    state, _ = env.reset()
                    #state = state.flatten()

                    state_array = np.array(state)
                    state_flattened = state_array.flatten()

                    episode_reward = 0
                    losses_ep = []
                    for step in range(max_steps):

                        if len(self.replay_buffer) <= batch_size:
                            action = self.policy_net.get_action(torch.FloatTensor(state_flattened).unsqueeze(0)) #to(device) ?

                        next_state, reward, done, truncated, _= env.step(action)

                        if self.mode == 'fc':
                            # corrects 'LazyFrames object' error
                            state_array = np.array(state)
                            state_flattened = state_array.flatten()
                            next_state_array = np.array(next_state)
                            next_state_flattened = next_state_array.flatten()
                            # action_array = np.array(action)
                            # action_flattened = action_array.flatten()
                        else:
                            next_state = torch.FloatTensor(np.array(next_state))
                            next_state = next_state.permute(3, 0, 1, 2)

                        self.replay_buffer.push(state_flattened, action, reward, next_state_flattened, done)
                        if len(self.replay_buffer) > batch_size:
                            loss = self.soft_q_update(batch_size, self.replay_buffer)
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
        with open('model.pkl', 'wb') as file:
            pickle.dump(self.policy_net, file)


if __name__ == "__main__":
    trainer = trainer_sac()
    trainer.train()