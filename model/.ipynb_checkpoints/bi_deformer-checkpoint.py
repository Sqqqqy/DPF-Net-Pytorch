import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class deformer(nn.Module):
    def __init__(self, hidden_size=128, z_dim=256):
        super().__init__()

        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        
        self.h1 = nn.Linear(3+self.z_dim, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size,self.hidden_size)                
        self.h3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.h4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.h5 = nn.Linear(self.hidden_size,self.hidden_size)
        self.h6 = nn.Linear(self.hidden_size,3)

        nn.init.xavier_uniform_(self.h1.weight)
        nn.init.constant_(self.h1.bias,0)
        nn.init.xavier_uniform_(self.h2.weight)
        nn.init.constant_(self.h2.bias,0)
        nn.init.xavier_uniform_(self.h3.weight)
        nn.init.constant_(self.h3.bias,0)
        nn.init.xavier_uniform_(self.h4.weight)
        nn.init.constant_(self.h4.bias,0)
        nn.init.xavier_uniform_(self.h5.weight)
        nn.init.constant_(self.h5.bias,0)
        nn.init.normal_(self.h6.weight)
        nn.init.constant_(self.h6.bias,0)
        
    def forward(self, points, z, common=True):
        bs = points.shape[0]
        num_points = points.shape[1]
        
        if common:
            instance_z = z.reshape((1,1,-1))
            instance_z = torch.repeat_interleave(instance_z, bs, dim=0)
            instance_z = torch.repeat_interleave(instance_z, num_points, dim=1)
            z = instance_z
        else:
            z = torch.repeat_interleave(z, num_points, dim=1)
            
        points = torch.cat([points, z],axis=2)
        
        h1 = self.h1(points)
        h1 = self.activation(h1)
        
        h2 = self.h2(h1)
        h2 = self.activation(h2)
        
        h3 = self.h3(h2)
        h3 = self.activation(h3)
        
        h3 = h1+h3
        
        h4 = self.h4(h3)
        h4 = self.activation(h4)
        
        h5 = self.h5(h4)
        h5 = self.activation(h5)
        
        out = self.h6(h5)
        
        return out

class bi_deformer(nn.Module):
    def __init__(self, hidden_size=128, z_dim=512, common_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.common_dim = common_dim
        self.deformer_instance = deformer(z_dim=self.z_dim)
        self.deformer_common = deformer(z_dim=self.common_dim)
        self.common_feature = torch.nn.Parameter(
                torch.randn(1, self.common_dim)
            )
        
    def forward(self, points, z):
        bs = points.shape[0]
        
        instance_deform = self.deformer_instance(points, z, common=False)
        common_deform = self.deformer_common(points, self.common_feature, common=True)
        
        return instance_deform, common_deform