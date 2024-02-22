
from torch import nn as nn
import torch
import math
import torch.nn.functional as F
from .embedding import LayerNorm

def SAGP(mean1, mean2, cov1, cov2):
    """Self-Adaptive Gaussian Production"""

    cov1 = torch.clamp(cov1, min=1e-24)
    cov2 = torch.clamp(cov2, min=1e-24)
    mean = ( cov1 * mean2 + cov2 * mean1 ) / (cov1 + cov2 )
    cov = 2 * (cov1*cov2)/(cov1 + cov2 ) 

    return mean, cov
    
def TriSAGP(mean1, mean2, mean3, cov1, cov2 , cov3):
    """Tri-Self-Adaptive Gaussian Production"""
    
    cov1 = torch.clamp(cov1, min=1e-24)
    cov2 = torch.clamp(cov2, min=1e-24)
    cov3 = torch.clamp(cov3, min=1e-24)
    cov = 1. / ( 1. / (cov1) + 1. / (cov2) + 1. / (cov3) )
    mean = cov * ( mean1 / (cov1) + mean2 / (cov2) + mean3 / (cov3) )
    return mean, cov

def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24)) 
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret


class PFF(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PFF, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.apply(self._init_weights)

    def forward(self, x):
        return self.LayerNorm(self.w_2(self.dropout(self.activation(self.w_1(x)))))
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class BSFFL(nn.Module):
    """
    Behavior specific feedforward network.
    """
    def __init__(self, d_model, d_ff, n_b, dropout=0.1):
        super().__init__()
        self.n_b = n_b
        self.pff = nn.ModuleList([PFF(d_model=d_model, d_ff=d_ff, dropout=dropout) for i in range(n_b)])

    def multi_behavior_pff(self, x, b_seq):
        outputs = [torch.zeros_like(x)]
        for i in range(self.n_b):
            outputs.append(self.pff[i](x))
        return torch.einsum('nBTh, BTn -> BTh', torch.stack(outputs, dim=0), F.one_hot(b_seq, num_classes=self.n_b+1).float())
    
    def forward(self, x, b_seq=None):
        return self.multi_behavior_pff(x, b_seq)
