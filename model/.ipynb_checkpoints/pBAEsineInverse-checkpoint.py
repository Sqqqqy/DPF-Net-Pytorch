import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,farthest_point_sample
from model.inversenet import InverseImplicitFun

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

class generator(nn.Module):
    def __init__(self, gf_dim=256,gf_split=8,z_dim=512):
        super().__init__()
        self.gf_split = gf_split
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.sine = Sine()
        self.h1 = nn.Linear(self.z_dim+3, self.gf_dim*4)
        self.h2 = nn.Linear(self.gf_dim*4,self.gf_dim)                
        self.h3 = nn.Linear(self.gf_dim,self.gf_split)
        self.h1.apply(first_layer_sine_init)
        self.h2.apply(sine_init)
        self.h3.apply(last_layer_sine_init)
        
        # nn.init.xavier_uniform_(self.h1.weight)
        # nn.init.constant_(self.h1.bias,0)
        # nn.init.xavier_uniform_(self.h2.weight)
        # nn.init.constant_(self.h2.bias,0)
        # nn.init.xavier_uniform_(self.h3.weight)
        # nn.init.constant_(self.h3.bias,0)
        
    def forward(self, points, z):
        batch_size = points.size(1)
        z = torch.repeat_interleave(z,batch_size, dim=1)
        # print(z.shape, points.shape)
        
        points = torch.cat([points,z],axis=2)
#         print(points.shape)
        points = self.h1(points)
        points = self.sine(points)
        points = self.h2(points)
        points = self.sine(points)
        points = self.h3(points)
        points = self.sine(points)
        # points = torch.sigmoid(points)
#         print(points.shape)
        out = torch.max(points, axis=2, keepdims=True)
#         print(out.values.shape)
        return points, out.values

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]]

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.sa0 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        # self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        # self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        # self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv0 = nn.Conv1d(16, 1, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)
    def forward(self, xyz, is_training=True):
        bs = xyz.shape[0]
        if is_training:
            np_rot = np.array([rotMat(2*3.14159*np.random.rand()) for i in range(bs)])
        else:
            np_rot = np.array([np.eye(3).tolist() for i in range(bs)])
        rot = torch.tensor(np_rot, dtype=torch.float32).cuda()
        # print(xyz.shape, rot.shape)
        xyz = (xyz @ rot).transpose(2,1)
        # xyz = xyz[:,[0,2,1],:]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = l4_points.permute(0, 2, 1)
        l4_points = self.conv0(l4_points)
#         l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

#         x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
#         x = self.conv2(x)
#         x = F.log_softmax(x, dim=1)
#         x = x.permute(0, 2, 1)
        return l4_points

class BAE_sine_inverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.generator = generator()
        self.inversefunc = InverseImplicitFun(z_dim=512, num_branches=8)
    
    def forward(self, fps_point, point_coord):
        bs = fps_point.shape[0]
        E = self.encoder(fps_point)
        G_, G = self.generator(point_coord, E.reshape(bs,1,-1))
        G_fps_, G_fps = self.generator(fps_point, E.reshape(bs,1,-1))
        inverse_points = self.inversefunc(E, G_fps_)
        
        return {
            'G_': G_,
            'G': G,
            'inverse_points': inverse_points,
        }
    
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)