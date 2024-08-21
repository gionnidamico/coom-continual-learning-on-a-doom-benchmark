import torch

class ewc():
    def __init__(self, lambdaq=1.0, lambdap=1.0) -> None:
        self.lambdaq = lambdaq
        self.lambdap = lambdap

    def get_weight_q(self, q, softq):
        jacobianqs = []
        reg_weights = []
        for param in softq.parameters():
            # Compute the gradient of the output with respect to the parameter
            jacobianq = torch.autograd.grad(outputs=q, inputs=param,
                                        grad_outputs=torch.ones_like(q),
                                        create_graph=True, retain_graph=True)[0]
            jacobianqs.append(jacobianq.flatten()**2)
        for q_g in jacobianqs:
            fisher = q_g**2
            reg_weights.append(self.lambdaq * torch.mean(fisher, dim=0))

        return torch.sum(torch.stack(reg_weights, dim = 0))

    def get_weight_policy(self, probs, policynet):
        jacobianpolicys = []
        reg_weights = []
        for param in policynet.parameters():
            # Compute the gradient of the output with respect to the parameter
            jacobianpolicy = torch.autograd.grad(outputs=probs, inputs=param,
                                        grad_outputs=torch.ones_like(probs),
                                        create_graph=True, retain_graph=True)[0]
            jacobianpolicys.append(jacobianpolicy.flatten()**2)

        for gs in jacobianpolicys:
            fisher = torch.sum(gs**2, 0)

            fisher = torch.clamp(fisher, min=1e-5)
            reg_weights.append(self.lambdaq * torch.mean(gs, dim=0))

        return torch.sum(torch.stack(reg_weights, dim = 0))