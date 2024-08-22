from typing import List

import numpy as np
import torch

class ExpWeights(object):

    def __init__(self,
                 arms: List[int],
                 lr: float,
                 window: int = 20,  # we don't use this yet..
                 epsilon: float = 0,  # set this above zero for
                 decay: float = 1,
                 greedy: bool = False):

        self.arms = arms
        self.l = {i: 0 for i in range(len(self.arms))}
        self.p = [0.5] * len(self.arms)
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.greedy = greedy

        self.choices = [self.arm]
        self.data = []

    def sample(self):
        if np.random.uniform() > self.epsilon:
            self.p = [np.exp(x) for x in self.l.values()]
            self.p /= np.sum(self.p)  # normalize to make it a distribution
            try:
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
            except ValueError:
                print("loss too large scaling")
                decay = self.lr * 0.1
                self.p = [np.exp(x * decay) for x in self.l.values()]
                self.p /= np.sum(self.p)  # normalize to make it a distribution
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return self.value

    def update_dists(self, feedback):
        for i in range(len(self.arms)):
            if self.greedy:
                self.l[i] *= self.decay
                self.l[i] += self.lr * feedback[i]
            else:
                self.l[i] *= self.decay
                self.l[i] += self.lr * (feedback[i] / max(np.exp(self.l[i]), 0.0001))


class Bandit():

    def __init__(self, num_actions, num_tasks):
        super()
        self.num_actions = num_actions
        self.num_tasks = num_tasks

         # Bandit params
        lr = 0.90
        decay = 0.90
        epsilon = 0.0
        bandit_step = 1
        greedy_bandit = True
        bandit_loss = 'mse'

        self.bandit = ExpWeights(arms=list(range(num_tasks)), lr=lr, decay=decay, greedy=greedy_bandit, epsilon=epsilon)

    def get_head(self):
        idx = self.bandit.sample()

        return idx
    
    def update(self, next_actions, next_actions_probs, action, action_probs, reward, done, gamma, device):
        '''
        Use Equation 2 to update pt
        φ with lt
        it = Gφt (θt+1)
        '''
        scores = []
        q_target = next_actions
        value_target = reward + (1.0 - done) * gamma * q_target
        for _ in range(self.num_tasks):
            state_action_values = action_probs.gather(1, torch.Tensor(np.array([action.cpu()])).long().view(1, -1).to(device))
            mus_ = state_action_values.detach().cpu().numpy()
            mse = np.sqrt(np.mean((mus_ - value_target.cpu().numpy()) ** 2))
            scores.append(min(1/mse, 50))

        self.bandit.update_dists(scores)
        