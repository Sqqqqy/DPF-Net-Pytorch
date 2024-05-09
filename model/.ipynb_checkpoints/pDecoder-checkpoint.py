import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.sin(30 * inputs)

class sineLinear(nn.Module):
    def __init__(self, in_channle, out_channel):
        super().__init__()
        self.linear = nn.Linear(in_channle, out_channel)
        self.activation = Sine()
    def forward(self, inputs):
        return self.activation(self.linear(inputs))
    
class sineLinearMLP(nn.Module):
    def __init__(self, in_channel=25, hidden_dims=[12, 6, 12, 25]):
        super().__init__()
        self.layers = nn.Sequential()
        self.hidden_dims = hidden_dims
        last_channel = in_channel
        for hidden_dim in self.hidden_dims:
            self.layers.append(sineLinear(last_channel, hidden_dim))
            last_channel = hidden_dim
            
        self.layers.append(nn.Linear(last_channel, 1))
    def forward(self, points):
        out = self.layers(points)
        return out
    
class pointDecoder(nn.Module):
    def __init__(self, num_part=8):
        super().__init__()
        self.num_part = num_part
        self.first_layer = sineLinear(3, 25)
        # self.layers = []
        
        self.sineLinearMLP = sineLinearMLP(in_channel=25, hidden_dims=[12, 6, 12, 25])
        
        
    def forward(self, points):
        # [bs N 3]
        h0 = self.first_layer(points)
        out = self.sineLinearMLP(h0)
               
        return out