import torch 
from torch import nn 

class Concater(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return torch.cat(tuple(features.values()), dim=-1)