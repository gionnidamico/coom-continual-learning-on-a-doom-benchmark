import torch

class mas():
    def __init__(self, lambdaq=1.0, lambdap=1.0) -> None:
        self.lambdaq = lambdaq
        self.lambdap = lambdap

    def get_weight_q(self, q, softq):
        jacobianqs = []
        reg_weights = []
        for param in softq.parameters():
            # Compute the gradient of the output with respect to the parameter
            jacobianq = torch.autograd.grad(outputs=torch.pow(q,2), inputs=param,
                                        grad_outputs=torch.ones_like(q),
                                        create_graph=True, retain_graph=True, allow_unused=True)[0]
            if jacobianq == None:
                continue
            jacobianqs.append(jacobianq.flatten()**2)
        for q_g in jacobianqs:
            reg_weights.append(self.lambdaq * torch.mean(torch.abs(q_g), dim=0))
        
        del jacobianqs

        return torch.sum(torch.stack(reg_weights, dim = 0))

    def get_weight_policy(self, probs, policynet):
        jacobianpolicys = []
        reg_weights = []
        for param in policynet.parameters():
            # Compute the gradient of the output with respect to the parameter
            jacobianpolicy = torch.autograd.grad(outputs=torch.pow(probs,2), inputs=param,
                                        grad_outputs=torch.ones_like(probs),
                                        create_graph=True, retain_graph=True, allow_unused=True)[0]
            if jacobianpolicy == None:
                continue
            jacobianpolicys.append(jacobianpolicy.flatten()**2)

        for gs in jacobianpolicys:
            reg_weights.append(self.lambdaq * torch.mean(torch.abs(gs), dim=0))

        del jacobianpolicys

        return torch.sum(torch.stack(reg_weights, dim = 0))