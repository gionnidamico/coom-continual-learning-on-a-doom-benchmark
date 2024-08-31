import torch

class L2:
    def __init__(self, lambda_q=1e-3, lambda_policy=1e-3, lambda_value=1e-3):
        self.lambda_q = lambda_q
        self.lambda_policy = lambda_policy
        self.lambda_value = lambda_value

    def get_weight_q(self, q_value, q_network):         # qvalue is not used, but it's here to align code to other regularizers in trainer
        l2_loss_q = 0.0
        for param in q_network.parameters():
            l2_loss_q += torch.norm(param, 2)
        return self.lambda_q * l2_loss_q

    def get_weight_policy(self, probs, policy_network): # same for probs
        l2_loss_policy = 0.0
        for param in policy_network.parameters():
            l2_loss_policy += torch.norm(param, 2)
        return self.lambda_policy * l2_loss_policy

    def get_weight_value(self, value_network):
        l2_loss_value = 0.0
        for param in value_network.parameters():
            l2_loss_value += torch.norm(param, 2)
        return self.lambda_value * l2_loss_value
