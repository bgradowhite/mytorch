"""
Constructs functions that show up in common nns, including: 
-- ReLU
-- Sigmoid
-- One Hot Vector
-- Softmax
-- Log Softmax
-- Cross Entropy Loss
"""

import torch


def relu(input):
    return input*(input>0)

def sigmoid(input):
    return 1/(1+torch.exp(-input))

def one_hot(input, n_classes):
    return torch.zeros(*input.shape, n_classes, device=input.device).scatter_(-1,input.unsqueeze(-1),1)

def log_softmax(input,dim=-1):
   max_val = input.max(dim=dim, keepdim=True).values
   stab_inp = input-max_val
   return stab_inp - torch.log(torch.sum(torch.exp(stab_inp),dim=dim, keepdim=True))
    
    
def cross_entropy(pred, target, reduction='mean'): #takes in label for target (not one-hot y)
    oh_t = one_hot(target,pred.shape[-1])
    loss = torch.sum(-log_softmax(pred, dim=-1)*oh_t, dim=-1)
    if reduction=='mean':
        return torch.mean(loss)
    elif reduction=='sum':
        return torch.sum(loss)
    
def softmax(input,dim=-1):
    max_val = input.max(dim=dim, keepdim=True).values
    stab_inp = input-max_val
    return torch.exp(stab_inp)/torch.sum(torch.exp(stab_inp),dim=dim, keepdim=True)





