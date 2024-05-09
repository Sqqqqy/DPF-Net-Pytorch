import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_bn_relu(in_dim, out_dim, kernel, stride=1, pad=0, dilate=1, group=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad, dilate, group),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )


def up_conv2d(in_dim, out_dim, kernel=3, pad=1, up_scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=up_scale, mode='nearest'),
        nn.Conv2d(in_dim, out_dim, kernel, padding=pad)
    )

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class GlobalPool(nn.Module):
    '''BxNxK -> BxK'''

    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3)
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)
        return X
