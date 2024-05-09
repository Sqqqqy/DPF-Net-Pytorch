import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class deformer(nn.Module):
    def __init__(self, hidden_size=128, z_dim=256, part_dim=64, common_dim=128):
        super().__init__()

        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.common_dim = common_dim

        
        
        self.h1 = nn.Linear(3+self.z_dim+self.part_dim+self.common_dim, self.hidden_size)
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
        
    def forward(self, points, z, instance_z, common_z):
        bs = points.shape[0]
        num_points = points.shape[1]
        instance_z = instance_z.reshape((1,1,-1))
        common_z = common_z.reshape((1,1,-1))
        
        part_z = torch.cat([instance_z, common_z],axis=-1)
        part_z = torch.repeat_interleave(part_z, bs, dim=0)
        part_z = torch.repeat_interleave(part_z, num_points, dim=1)
        
        z = torch.repeat_interleave(z, num_points, dim=1)
        
        points = torch.cat([points,z,part_z],axis=2)
        
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

class deformer_small(nn.Module):
    def __init__(self, hidden_size=128, z_dim=256, part_dim=64, common_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.common_dim = common_dim
        self.activation = nn.ReLU()
        
        self.h1 = nn.Linear(3+self.z_dim+self.part_dim+self.common_dim, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size,self.hidden_size)                
        self.h3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.h6 = nn.Linear(self.hidden_size,3)

        nn.init.xavier_uniform_(self.h1.weight)
        nn.init.constant_(self.h1.bias,0)
        nn.init.xavier_uniform_(self.h2.weight)
        nn.init.constant_(self.h2.bias,0)
        nn.init.xavier_uniform_(self.h3.weight)
        nn.init.constant_(self.h3.bias,0)
        nn.init.normal_(self.h6.weight)
        nn.init.constant_(self.h6.bias,0)
        
    def forward(self, points, z, instance_z, common_z):
        bs = points.shape[0]
        num_points = points.shape[1]
        instance_z = instance_z.reshape((1,1,-1))
        common_z = common_z.reshape((1,1,-1))
        
        part_z = torch.cat([instance_z,common_z],axis=-1)
        part_z = torch.repeat_interleave(part_z, bs, dim=0)
        part_z = torch.repeat_interleave(part_z, num_points, dim=1)
        
        z = torch.repeat_interleave(z, num_points, dim=1)
        
        points = torch.cat([points,z,part_z],axis=2)
        
        h1 = self.h1(points)
        h1 = self.activation(h1)
        
        h2 = self.h2(h1)
        h2 = self.activation(h2)
        
        h2 = h1+h2
        
        h3 = self.h3(h2)
        h3 = self.activation(h3)
        
        out = self.h6(h3)
        
        return out

class k_deformer(nn.Module):
    def __init__(self, hidden_size=128, z_dim=512, part_dim=128, num_template=4, common_dim=96):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.num_template = num_template
        self.common_dim = common_dim
        self.deformer = deformer(z_dim=self.z_dim, part_dim=self.part_dim, common_dim=self.common_dim)
        self.part_feature = torch.nn.Parameter(
                torch.randn(self.num_template, self.part_dim)
            )
        self.common_feature = torch.nn.Parameter(
                torch.randn(1, self.common_dim)
            )
        
    def forward(self, points, z):
        bs = points.shape[0]
        
        # deformations = None
        # for split_i in range(self.num_template):
        #     deform_part = self.deformer(points, z, self.part_feature[split_i], self.common_feature)
        #     if split_i == 0:
        #         deformations = deform_part
        #     else:
        #         deformations += deform_part
        
        deformations = []
        for split_i in range(self.num_template):
            deform_part = self.deformer(points, z, self.part_feature[split_i], self.common_feature)
            deformations.append(deform_part)
        return deformations