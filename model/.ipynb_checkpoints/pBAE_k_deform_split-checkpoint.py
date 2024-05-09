import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.k_deformer_split import k_deformer
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,farthest_point_sample

class generator(nn.Module):
    def __init__(self, gf_dim=256, gf_split=8,z_dim=512, num_template=4, common_dim=128):
        super().__init__()
        self.gf_split = gf_split
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.common_dim = common_dim
        self.num_template = num_template
        self.layer1 = nn.Linear(self.z_dim+3, self.gf_dim*4)
        self.layer2 = nn.Linear(self.gf_dim*4,self.gf_dim)                
        self.layer3 = nn.Linear(self.gf_dim,self.gf_split)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias,0)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.constant_(self.layer2.bias,0)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.constant_(self.layer3.bias,0)
        
    def forward(self, points, z, deformations):
        bs = points.size(0)
        num_points = points.size(1)
        z = torch.repeat_interleave(z, num_points, dim=1)
        # common_z = torch.repeat_interleave(common_z, bs, dim=0)
        # common_z = torch.repeat_interleave(common_z, num_points, dim=1)
        
        comp_deformation = deformations
        deformation = comp_deformation[:,:,:3]
        
        # delta = comp_deformation[:,:,3]
        inp = torch.cat([points+deformation,z],axis=2)
        h1 = self.layer1(inp)
        h1 = F.leaky_relu(h1, negative_slope=0.01, inplace=True)
        h2 = self.layer2(h1)
        h2 = F.leaky_relu(h2, negative_slope=0.01, inplace=True)
        h3 = self.layer3(h2)
        # + delta.unsqueeze(-1)
        h3 = torch.sigmoid(h3)
        
        # out = torch.max(h3, axis=2, keepdims=True)

        return h3
    # , out.values

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]]

class encoder(nn.Module):
    def __init__(self, num_template=4, z_dim=512):
        super(encoder, self).__init__()
        self.num_template = num_template
        self.z_dim = z_dim
        self.sa0 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, self.z_dim], False)
        
        self.conv0 = nn.Conv1d(16, 1, 1)
        
    def forward(self, xyz, is_training=True):
        bs = xyz.shape[0]
        if is_training:
            np_rot = np.array([rotMat(2*3.14159*np.random.rand()) for i in range(bs)])
        else:
            np_rot = np.array([np.eye(3).tolist() for i in range(bs)])
        rot = torch.tensor(np_rot, dtype=torch.float32).cuda()
        xyz = (xyz @ rot).transpose(2,1)
        # xyz = xyz[:,[0,2,1],:]
        # xyz = xyz.transpose(2,1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = l4_points.permute(0, 2, 1)
        # local_feat = self.conv0(l4_points)
        global_feat = self.conv0(l4_points)

        return global_feat

class BAE_k_deform_split(nn.Module):
    def __init__(self, num_template=8, z_dim=256, part_dim=128, common_dim=96):
        super().__init__()
        self.num_template = num_template
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.common_dim = common_dim
        self.hidden_dim = 128
        self.encoder = encoder(z_dim=self.z_dim, num_template=self.num_template)
        self.generator = generator(z_dim=self.part_dim, num_template=self.num_template, gf_split=1)
        self.deformer = k_deformer(z_dim=self.z_dim, num_template=self.num_template, part_dim=self.part_dim, common_dim=self.common_dim)
        self.split_layer = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.Linear(self.hidden_dim, self.num_template*(self.part_dim+3),)
    def forward(self, fps_point, point_coord):
        bs, N = point_coord.shape[0], point_coord.shape[1]
        global_feat = self.encoder(fps_point)
                                         
        embed_z = self.split_layer(global_feat.reshape(bs,1,-1))
        embed_zs = embed_z.reshape(bs, self.num_template, self.part_dim+3)
        part_poses = embed_zs[:,:,-3:]
        embed_zs = embed_zs[:,:,:-3] 
                                         
        deform_out, part_codes = self.deformer(point_coord, global_feat, embed_zs)
        
        part_probs = []
        for split_i in range(self.num_template):
            part_prob = self.generator(point_coord, part_codes[split_i].reshape(bs,1,-1), deform_out[split_i])
            part_probs.append(part_prob)
            
        G_ = torch.cat(part_probs, dim=-1)
        G = torch.max(G_, dim=-1, keepdims=True).values

        return {'G_': G_,
                'G': G,
                # 'deformation': deform_out,
               }