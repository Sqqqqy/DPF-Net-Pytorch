import torch
import torch.nn as nn
from model.submodules import PointwiseMLP

class FoldingNetVanilla(nn.Module):

    def __init__(self, folding1_dims, folding2_dims):
        super(FoldingNetVanilla, self).__init__()

        # The folding decoder
        self.fold1 = PointwiseMLP(folding1_dims, doLastRelu=False)
        if folding2_dims[0] > 0:
            self.fold2 = PointwiseMLP(folding2_dims, doLastRelu=False)
        else: self.fold2 = None

    def forward(self, cw, grid, **kwargs):

        cw_exp = cw.expand(-1, grid.shape[1], -1) # batch_size X point_num X code_length

        # 1st folding
        in1 = torch.cat((grid, cw_exp), 2) # batch_size X point_num X (code_length + 3)
        out1 = self.fold1(in1) # batch_size X point_num X 3

        # 2nd folding
        if not(self.fold2 is None):
            in2 = torch.cat((out1, cw_exp), 2) # batch_size X point_num X (code_length + 4)
            out2 = self.fold2(in2) # batch_size X point_num X 3
            return out2
        else: return out1

def get_Conv2d_layer(dims, kernel_size, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        if kernel_size != 1:
            layers.append(nn.ReplicationPad2d(int((kernel_size - 1) / 2)))
        layers.append(nn.Conv2d(in_channels=dims[i-1], out_channels=dims[i],
            kernel_size=kernel_size, stride=1, padding=0, bias=True))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU(inplace=True))
    return layers


class Conv2dLayers(nn.Sequential):
    def __init__(self, dims, kernel_size, doLastRelu=False):
        layers = get_Conv2d_layer(dims, kernel_size, doLastRelu)
        super(Conv2dLayers, self).__init__(*layers)


class TearingNetBasic(nn.Module):

    def __init__(self, tearing1_dims, tearing2_dims, grid_dims, kernel_size=1):
        super(TearingNetBasic, self).__init__()

        self.grid_dims = grid_dims
        self.tearing1 = Conv2dLayers(tearing1_dims, kernel_size=kernel_size, doLastRelu=False)
        self.tearing2 = Conv2dLayers(tearing2_dims, kernel_size=kernel_size, doLastRelu=False)

    def forward(self, cw, grid, pc, **kwargs):

        grid_exp = grid.contiguous().view(grid.shape[0], self.grid_dims[0], self.grid_dims[1], 2) # batch_size X dim0 X dim1 X 2
        pc_exp = pc.contiguous().view(pc.shape[0], self.grid_dims[0], self.grid_dims[1], 3) # batch_size X dim0 X dim1 X 3
        cw_exp = cw.unsqueeze(1).expand(-1, self.grid_dims[0], self.grid_dims[1], -1) # batch_size X dim0 X dim1 X code_length
        in1 = torch.cat((grid_exp, pc_exp, cw_exp), 3).permute([0, 3, 1, 2])

        # Compute the torn 2D grid
        out1 = self.tearing1(in1) # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2) # 2nd tearing
        out2 = out2.permute([0, 2, 3, 1]).contiguous().view(grid.shape[0], self.grid_dims[0] * self.grid_dims[1], 2)
        return grid + out2


class TearingNetBasicModel(nn.Module):

#     @staticmethod
#     def add_options(parser, isTrain = True):

#         # General optional(s)
#         parser.add_argument('--grid_dims', type=int, nargs='+', help='Grid dimensions.')

#         # Options related to the Folding Network
#         parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3], help='Dimensions of the first folding module.')
#         parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3], help='Dimensions of the second folding module.')

#         # Options related to the Tearing Network
#         parser.add_argument('--tearing1_dims', type=int, nargs='+', default=[523, 256, 128, 64], help='Dimensions of the first tearing module.')
#         parser.add_argument('--tearing2_dims', type=int, nargs='+', default=[587, 256, 128, 2], help='Dimensions of the second tearing module.')
#         parser.add_argument('--tearing_conv_kernel_size', type=int, default=1, help='Kernel size of the convolutional layers in the Tearing Network, 1 implies MLP.')

#         return parser

    def __init__(self):
        super(TearingNetBasicModel, self).__init__()

        # Initialize the regular 2D grid
        range_x = torch.linspace(-1.0, 1.0, 45)
        range_y = torch.linspace(-1.0, 1.0, 45)
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the Folding Network and the Tearing Network
        self.folding = FoldingNetVanilla([514, 512, 512, 3], [515, 512, 512, 3])
        self.tearing = TearingNetBasic([517, 256, 128, 64], [581, 256, 128, 2], [45, 45], 1)

    def forward(self, cw):

        grid0 = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1) # batch_size X point_num X 2
        pc0 = self.folding(cw, grid0) # Folding Network
        grid1 = self.tearing(cw, grid0, pc0) # Tearing Network
        pc1 = self.folding(cw, grid1) # Folding Network

        return pc0, pc1, grid1