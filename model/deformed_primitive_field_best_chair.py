import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,farthest_point_sample
from sdf import box, sphere, ellipsoid
from model import modules
from model.meta_modules import HyperNetwork

class FeedForward(nn.Module):
    def __init__(self, in_dim, h_dim, out_d=None, act=nn.functional.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward_interpolation(self, queries, keys, values, alpha, mask=None):
        attention = torch.einsum('nhd,bmhd->bnmh', queries[0], keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        attention = attention * alpha[:, None, None, None]
        out = torch.einsum('bnmh,bmhd->nhd', attention, values).reshape(1, attention.shape[1], -1)
        return out, attention

    def forward(self, x, y=None, mask=None, alpha=None):
        y = y if y is not None else x
        b_a, n, c = x.shape
        b, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b_a, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        if alpha is not None:
            out, attention = self.forward_interpolation(queries, keys, values, alpha, mask)
        else:
            attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
            attention = attention.softmax(dim=2)
            out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None, alpha=None):
        x_, attention = self.attn(self.norm1(x), y, mask, alpha)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None, alpha=None):
        x = x + self.attn(self.norm1(x), y, mask, alpha)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nn.functional.relu,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = FeedForward(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None, alpha=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask, alpha)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None, alpha=None):
        for layer in self.layers:
            x = layer(x, y, mask, alpha)
        return x

    def __init__(self, dim_self, num_heads, num_layers, dim_ref=None,
                 mlp_ratio=2., act=nn.functional.relu, norm_layer=nn.LayerNorm):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.layers = nn.ModuleList([TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                                      norm_layer=norm_layer) for _ in range(num_layers)])

def quat2mat(quat):
    B = quat.shape[0]
    N = quat.shape[1]
    quat = quat.contiguous().view(-1,4)
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B*N, 3, 3)
    rotMat = rotMat.view(B,N,3,3)
    return rotMat

class deformer(nn.Module):
    def __init__(self, hidden_size=128, z_dim=256, part_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.part_dim = part_dim
        
        self.deformer = modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=128, num_hidden_layers=2, in_features=3, out_features=4)
        self.hypernet = HyperNetwork(hyper_in_features=self.part_dim*2, hyper_hidden_layers=1, hyper_hidden_features=128, hypo_module=self.deformer)
        
    
    def forward(self, points, z):
        hypo_params = self.hypernet(z)
        deform_out = self.deformer({'coords': points}, params=hypo_params)
        return deform_out['model_out']


class generator(nn.Module):
    def __init__(self, gf_dim=256,gf_split=8,z_dim=512,num_template=4, common_dim=128):
        super().__init__()
        self.gf_split = gf_split
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.common_dim = common_dim
        self.num_template = num_template
        self.layer1 = nn.Linear(3, self.gf_dim*4)
        self.layer2 = nn.Linear(self.gf_dim*4,self.gf_dim)                
        self.layer3 = nn.Linear(self.gf_dim,self.gf_split)
        
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias,0) 
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.constant_(self.layer2.bias,0)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.constant_(self.layer3.bias,0)
        
    def forward(self, points, deformations):
        bs = points.size(0)
        num_points = points.size(1)
        inp = points# + deformations
        h1 = self.layer1(inp)
        h1 = F.leaky_relu(h1, negative_slope=0.01, inplace=True)
        h2 = self.layer2(h1)
        h2 = F.leaky_relu(h2, negative_slope=0.01, inplace=True)
        h3 = self.layer3(h2)
        
        # h3 = torch.sigmoid(h3)

        return h3

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]]



class encoder(nn.Module):
	def __init__(self, ef_dim=32, num_template=4, z_dim=512):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.z_dim = z_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.constant_(self.conv_1.bias,0)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.constant_(self.conv_2.bias,0)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.constant_(self.conv_3.bias,0)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = F.leaky_relu(d_1, negative_slope=0.2, inplace=True)

		d_2 = self.conv_2(d_1)
		d_2 = F.leaky_relu(d_2, negative_slope=0.2, inplace=True)
		
		d_3 = self.conv_3(d_2)
		d_3 = F.leaky_relu(d_3, negative_slope=0.2, inplace=True)

		d_4 = self.conv_4(d_3)
		d_4 = F.leaky_relu(d_4, negative_slope=0.2, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.ef_dim*8)
		# d_5 = torch.sigmoid(d_5)

		return d_5

# class encoder(nn.Module):
#     def __init__(self, num_template=4, z_dim=512):
#         super(encoder, self).__init__()
#         self.num_template = num_template
#         self.z_dim = z_dim
#         self.sa0 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
#         self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
#         self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
#         self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, self.z_dim], False)
        
#         self.conv0 = nn.Conv1d(16, 1, 1)
        
#     def forward(self, xyz, is_training=False):
#         bs = xyz.shape[0]
#         if is_training:
#             np_rot = np.array([rotMat(2*3.14159*np.random.rand()) for i in range(bs)])
#         else:
#             np_rot = np.array([np.eye(3).tolist() for i in range(bs)])
#         rot = torch.tensor(np_rot, dtype=torch.float32).cuda()
#         xyz = (xyz @ rot).transpose(2,1)
        
#         # xyz = xyz.transpose(2,1)
#         l0_points = xyz
#         l0_xyz = xyz[:,:3,:]
#         l1_xyz, l1_points = self.sa0(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l4_points = l4_points.permute(0, 2, 1)
#         # local_feat = self.conv0(l4_points)
#         global_feat = self.conv0(l4_points)

#         return global_feat


def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False



class DPF(nn.Module):
    def __init__(self, num_template=12, z_dim=256, part_dim=64, is_stage2=False):
        super().__init__()
        self.num_template = num_template
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.hidden_dim = self.part_dim*2
        self.encoder = encoder(z_dim=self.z_dim, num_template=self.num_template)
        
        self.split_layer = nn.Conv1d(self.z_dim, self.num_template*self.part_dim, kernel_size=1, bias=False)

        self.conv_cuboid = nn.Sequential(nn.Conv1d(self.part_dim*2, 256, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                         nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True))

        self.conv_scale = nn.Conv1d(128, 4, kernel_size=1, bias=False)
#         nn.init.zeros_(self.conv_scale.bias)

        self.conv_rotate = nn.Conv1d(128, 4, kernel_size=1, bias=False)
#         self.conv_rotate.bias.data = torch.Tensor([1, 0, 0, 0])

        self.conv_trans = nn.Conv1d(128, 3, kernel_size=1, bias=False)
#         nn.init.zeros_(self.conv_trans.bias)

        self.conv_density = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        # nn.Conv1d(64, 1, kernel_size=1, bias=True)
        # nn.init.zeros_(self.conv_density.bias)

        self.stage2 = is_stage2
        # self.stage2 = True

        if self.stage2:
            self.deformers = nn.ModuleList()
            for i in range(1):
                self.deformers.append(deformer(z_dim=self.z_dim, part_dim=self.part_dim))

        self.global_feature = torch.nn.Parameter(
                torch.rand((self.num_template, self.part_dim)),
                requires_grad = True
            )
        # nn.init.normal_(self.global_feature, mean=1e-5, std=0.02)

        # self.W = nn.Parameter(torch.zeros((self.num_template, 1)))
        # nn.init.normal_(self.W, mean=1e-5, std=0.02) 
       

    def dist_s(self, q, t, s, p_type):
        '''
        q: [bs x numpoint x 3]
        t: [bs x 1 x 3]
        s: [bs x 1 x 3]
        '''
        eps = 1e-6
        if p_type in list(range(0, 8)):
            # return torch.max(torch.abs(q - t) - s, dim = -1, keepdims=True)[0]
            return torch.max(torch.abs(q - t) / (s + eps), dim = -1, keepdims=True)[0]
        elif p_type in list(range(8, 16)):
            return torch.sqrt(torch.sum((((q - t) / (s + eps))**2), dim = -1, keepdims=True))

    def primitive_field(self, points, r, s, t, p_type):
        tau=4
        rotated_points = torch.einsum('bnk,bnkl->bnl', points, r)
        dist = self.dist_s(rotated_points, t, s, p_type)
        return torch.exp(- tau * dist)
    
    def primitive_field_ori(self, points, r, s, t, p_type):
        # bs = points.shape[0]
        # ctx = points.device
        # [bs, N, 3] x [bs, N, 3, 3]
        rotated_points = torch.einsum('bnk,bnkl->bnl', points, r)
        dist = self.dist_s(rotated_points, t, s, p_type).float()
        primitive_field = 1 - dist
        primitive_field = torch.clamp(primitive_field, 0.0, 1.0)
        return primitive_field


    def prob_field(self, rotation_, scale_, pos, p_type):
        bs = pos.shape[0]
        ctx = pos.device
        center_pos = torch.Tensor(np.zeros((bs,1,3))).to(ctx) #[bs 1 3]
        
        if p_type == 'bbox':         
            primitive = box(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'sphere':
            primitive = sphere(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'ellipsoid':
            primitive = ellipsoid(center=center_pos, size=scale_).rotate(rotation_)
        center_sdf = torch.abs(primitive(center_pos))
        bs, num_points = pos.shape[0], pos.shape[1]

        primitive_field = torch.abs(primitive(pos)+center_sdf)/(2*center_sdf+1e-10)
        tau = 4
        primitive_field = torch.clamp(primitive_field, 0, 2.0)
        primitive_field = torch.exp(- tau * primitive_field)
        # primitive_field = torch.clamp(1-primitive_field, 0, 2.0)

        return primitive_field.reshape((bs, num_points, 1))
    
    def select_center(self, instance_feat):
        bs = instance_feat.shape[0]
        ins_norm = instance_feat * 1/torch.sqrt((instance_feat**2).sum(-1, keepdim=True))
        global_norm = self.global_feature * 1/torch.sqrt((self.global_feature**2).sum(-1, keepdim=True))
        # bs z * z k  --> bs k
        cos_matrix = ins_norm @ global_norm.transpose(1,0)
        values = torch.max(cos_matrix, dim=-1, keepdim=True).values
        masks = (cos_matrix == values).float()
        # bs k * k z
        return masks @ self.global_feature

    def mixing_parts(self, part_code, mask=None):
        zh_, attn = self.mixing_network.forward_with_attention(part_code, mask=mask)
        return zh_, attn

    def primitive_deform(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        deformations = []
        deform_reg_loss = 0
        correction_reg_loss = 0
        
        p_type_dict = {0:'bbox', 2:'ellipsoid', 1:'sphere'}

        for split_i in range(self.num_template):

            p_type = p_type_dict[0]
            
            # occupancy = self.primitive_field(point_coord, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), split_i)
            deform_out = self.deformers[0](point_coord, z_fusion[:,split_i,:].unsqueeze(1))
            deform_part = deform_out[:,:,:3]
            exist_part = torch.tanh(deform_out[:,:,3]).unsqueeze(-1)

            points = point_coord + deform_part
            occupancy2 = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))
            occupancy3 = torch.clamp(exist_part + occupancy2, 0, 1)

            G_.append(occupancy3)

            deformations.append(deform_part)
            deform_reg = deform_part - deform_part.mean(-2, keepdim=True)
            deform_reg_loss = deform_reg_loss + ((deform_reg**2).sum(-2))
            correction_reg_loss = correction_reg_loss + torch.abs(exist_part)
        
        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
       
        deform_loss = (deform_reg_loss.mean()*10  + correction_reg_loss.mean()*50) / self.num_template

        return G_, deform_loss

    def primitive_structure(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        p_type_dict = {0:'bbox', 2:'ellipsoid', 1:'sphere'}
        for split_i in range(self.num_template):
            p_type = p_type_dict[0]
            points = point_coord
            occupancy = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))

            G_.append(occupancy)

        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
        deform_loss = None
        return G_, deform_loss


    def querying_occupancy(self, p_type, point_coord, rotation, scale, loc, density):
        points = point_coord + loc
        occupancy = self.prob_field(rotation, scale, points, p_type) * density
        return occupancy

    def forward(self, point_voxels, point_coord, apply_grad_loss=False):
        bs, N = point_coord.shape[0], point_coord.shape[1]
        ctx = point_voxels.device
        ################# feature extractor ##################
        instance_feat = self.encoder(point_voxels).unsqueeze(-1)  # [bs, z_dim, 1]


        ################# part decompostion ##################
        ins_codes = self.split_layer(instance_feat).transpose(2,1).reshape(bs, self.num_template, self.part_dim)
        com_codes = torch.repeat_interleave(self.global_feature.unsqueeze(0), bs, dim=0)
        z_fusion = torch.cat([ins_codes, com_codes], axis=2)
        z_fusion = self.conv_cuboid(z_fusion.transpose(2, 1)).transpose(2, 1)

        ################# structure extractor ##################
        scale_ = torch.sigmoid(self.conv_scale(z_fusion.transpose(2, 1))).transpose(2, 1)
        q_ = self.conv_rotate(z_fusion.transpose(2, 1)).transpose(2, 1)
        loc_ = torch.tanh(self.conv_trans(z_fusion.transpose(2, 1))).transpose(2, 1)*0.5
        rotation_ = quat2mat(F.normalize(q_,dim=-1,p=2))
        density_ = torch.sigmoid(self.conv_density(z_fusion.transpose(2, 1)).transpose(2, 1))
        # print(density_.mean(0).mean(1))
        # print(scale_.mean(0).mean(0))
        ################# k occupancy indicator ##################

        if self.stage2:
            G_, deform_loss = self.primitive_deform(point_coord, z_fusion, rotation_, scale_, loc_, density_)
            values = torch.max(G_, dim=-1, keepdim=True).values
            # masks = (G_ == values).float()+0.01
            # G = torch.clamp(torch.sum(masks * G_, dim=-1, keepdims=True), 0.0, 1.0)
            G = torch.clamp(torch.max(G_, axis=2, keepdims=True)[0], 0.0, 1.0)
        else:
            G_, deform_loss = self.primitive_structure(point_coord, z_fusion, rotation_, scale_, loc_, density_)
            values = torch.max(G_, dim=-1, keepdim=True).values
            masks = (G_ == values).float()+0.01
            G = torch.clamp(torch.sum(masks * G_, dim=-1, keepdims=True), 0.0, 1.0)
            # G = torch.clamp(torch.max(G_, axis=2, keepdims=True)[0], 0.0, 1.0)

        com_norm_code = com_codes * 1/torch.sqrt((com_codes**2).sum(-1, keepdim=True))
        
        # 
        return {'G_': G_,
                'G': G,
                'deformation_loss': deform_loss,
                'part_codes_loss': ((F.normalize(q_,dim=-1,p=2)-torch.Tensor([1,0,0,0]).to(ctx))**2).mean(),#+0.01*((torch.abs(com_norm_code @ (com_norm_code.transpose(2,1)))-self.num_template)**2).sum(-1).mean(),
                'global_feature_loss': None,
                'W_loss': None,
               }