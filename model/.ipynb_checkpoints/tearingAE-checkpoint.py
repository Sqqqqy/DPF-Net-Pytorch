import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np

from model.tearingnet import TearingNetBasicModel
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,farthest_point_sample

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



class PointCloudAutoencoder(nn.Module):

    def __init__(self):
        super(PointCloudAutoencoder, self).__init__()
    
        self.encoder = encoder()
        self.decoder = TearingNetBasicModel()

    def forward(self, data):

        cw = self.encoder(data)
        if isinstance(self.decoder, TearingNetBasicModel):
            rec0, rec1, grid = self.decoder(cw)
            return {"rec": rec1, "rec_pre": rec0, "grid": grid, "cw": cw}
        elif isinstance(self.decoder, TearingNetGraphModel):
            rec0, rec1, rec2, grid, graph_wght = self.decoder(cw)
            return {"rec": rec2, "rec_pre": rec1, "rec_pre2": rec0, "grid": grid, "graph_wght": graph_wght, "cw": cw}
        else:
            rec = self.decoder(cw)
            return {"rec": rec, "cw": cw}