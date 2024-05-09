import torch
from torch import nn
import torch.nn.functional as F

class InverseImplicitFun(nn.Module):
    def __init__(self, z_dim=256, num_branches=12):
        super(InverseImplicitFun, self).__init__()

        self.fc1 = nn.Linear(z_dim+num_branches, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512+z_dim+num_branches, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 3)


    def forward(self, z, values):

        z = z.repeat(1, values.shape[1], 1)
        valuez = torch.cat((z, values), dim=2)

        x1 = self.fc1(valuez)
        x1 = F.leaky_relu(x1, negative_slope=0.02, inplace=True)
        x2 = self.fc2(x1)
        x2 = F.leaky_relu(x2, negative_slope=0.02, inplace=True)
        x3 = self.fc3(x2)
        x3 = F.leaky_relu(x3, negative_slope=0.02, inplace=True)
        x4 = self.fc4(x3)
        x4 = F.leaky_relu(x4, negative_slope=0.02, inplace=True)
        x4 = torch.cat((x4, valuez), dim=2)
        x5 = self.fc5(x4)
        x5 = F.leaky_relu(x5, negative_slope=0.02, inplace=True)
        x6 = self.fc6(x5)
        x6 = F.leaky_relu(x6, negative_slope=0.02, inplace=True)
        x7 = self.fc7(x6)
        x7 = F.leaky_relu(x7, negative_slope=0.02, inplace=True)
        x8 = self.fc8(x7)
        x8 = torch.tanh(x8)

        return x8