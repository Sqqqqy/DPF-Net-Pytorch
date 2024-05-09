import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,farthest_point_sample
from sdf import box, sphere, ellipsoid, cylinder
from model import modules
from model.meta_modules import HyperNetwork


class resnet_block(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(resnet_block, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		if self.dim_in == self.dim_out:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_1 = nn.BatchNorm2d(self.dim_out)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_2 = nn.BatchNorm2d(self.dim_out)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
		else:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
			self.bn_1 = nn.BatchNorm2d(self.dim_out)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.bn_2 = nn.BatchNorm2d(self.dim_out)
			self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
			self.bn_s = nn.BatchNorm2d(self.dim_out)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
			nn.init.xavier_uniform_(self.conv_s.weight)

	def forward(self, input, is_training=False):
		if self.dim_in == self.dim_out:
			output = self.bn_1(self.conv_1(input))
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
			output = self.bn_2(self.conv_2(output))
			output = output+input
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
		else:
			output = self.bn_1(self.conv_1(input))
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
			output = self.bn_2(self.conv_2(output))
			input_ = self.bn_s(self.conv_s(input))
			output = output+input_
			output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
		return output

class img_encoder(nn.Module):
	def __init__(self, img_ef_dim=64, z_dim=256):
		super(img_encoder, self).__init__()
		self.img_ef_dim = img_ef_dim
		self.z_dim = z_dim
		self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
		self.bn_0 = nn.BatchNorm2d(self.img_ef_dim)
		self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim*2)
		self.res_4 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*2)
		self.res_5 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*4)
		self.res_6 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*4)
		self.res_7 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*8)
		self.res_8 = resnet_block(self.img_ef_dim*8, self.img_ef_dim*8)
		self.conv_9 = nn.Conv2d(self.img_ef_dim*8, self.img_ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.bn_9 = nn.BatchNorm2d(self.img_ef_dim*8)
		self.conv_10 = nn.Conv2d(self.img_ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_0.weight)
		nn.init.xavier_uniform_(self.conv_9.weight)
		nn.init.xavier_uniform_(self.conv_10.weight)

	def forward(self, view, is_training=False):
		layer_0 = self.bn_0(self.conv_0(1-view))
		layer_0 = F.leaky_relu(layer_0, negative_slope=0.02, inplace=True)

		layer_1 = self.res_1(layer_0, is_training=is_training)
		layer_2 = self.res_2(layer_1, is_training=is_training)
		
		layer_3 = self.res_3(layer_2, is_training=is_training)
		layer_4 = self.res_4(layer_3, is_training=is_training)
		
		layer_5 = self.res_5(layer_4, is_training=is_training)
		layer_6 = self.res_6(layer_5, is_training=is_training)
		
		layer_7 = self.res_7(layer_6, is_training=is_training)
		layer_8 = self.res_8(layer_7, is_training=is_training)
		
		layer_9 = self.bn_9(self.conv_9(layer_8))
		layer_9 = F.leaky_relu(layer_9, negative_slope=0.02, inplace=True)
		
		layer_10 = self.conv_10(layer_9)
		layer_10 = layer_10.view(-1,self.z_dim)
		layer_10 = torch.sigmoid(layer_10)

		return layer_10


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
        
        self.deformer = modules.SingleBVPNet(type='relu',mode='mlp', hidden_features=128, num_hidden_layers=2, in_features=4, out_features=4)
        self.hypernet = HyperNetwork(hyper_in_features=self.part_dim*2, hyper_hidden_layers=1, hyper_hidden_features=128, hypo_module=self.deformer)
        
    
    def forward(self, points, z):
        hypo_params = self.hypernet(z)
        deform_out = self.deformer({'coords': points}, params=hypo_params)
        return deform_out['model_out']


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


def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False


class DPF_SVR(nn.Module):
    def __init__(self, num_template=8, z_dim=128, part_dim=64):
        super().__init__()
        self.num_template = num_template
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.hidden_dim = self.part_dim*2
        self.SVR_encoder = img_encoder(z_dim=self.z_dim)

    def network_loss(self, pred_z, gt_z):
        return torch.mean((pred_z - gt_z)**2)

    def forward(self, point_voxels, image, z_3d, point_coord):
        bs, N = point_voxels.shape[0], point_voxels.shape[1]
        ctx = point_voxels.device
        ################# feature extractor ##################
        im_feat = self.SVR_encoder(image).unsqueeze(-1).transpose(2,1)
        # print(z_3d.shape, im_feat.shape)
        svr_loss = self.network_loss(z_3d, im_feat)
        return {'im_feat': im_feat,
                'instance_feat': z_3d,
                'svr_loss': svr_loss}

class DPF_IM(nn.Module):
    def __init__(self, num_template=8, z_dim=256, part_dim=64, is_stage2=False, primitive_type='cuboid'):
        super().__init__()
        self.num_template = num_template
        self.z_dim = z_dim
        self.part_dim = part_dim
        self.hidden_dim = self.part_dim*2
        self.encoder = encoder(z_dim=self.z_dim, num_template=self.num_template)
        self.SVR_encoder = img_encoder(z_dim=self.z_dim)

        self.split_layer = nn.Conv1d(self.z_dim, self.num_template*self.part_dim, kernel_size=1, bias=False)

        self.conv_cuboid = nn.Sequential(nn.Conv1d(self.part_dim*2, 256, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                         nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True))

        self.conv_scale = nn.Conv1d(128, 6, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv_scale.bias)

        self.conv_rotate = nn.Conv1d(128, 4, kernel_size=1, bias=True)
        self.conv_rotate.bias.data = torch.Tensor([1, 0, 0, 0])

        self.conv_trans = nn.Conv1d(128, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv_trans.bias)

        self.conv_density = nn.Conv1d(128, 1, kernel_size=1, bias=False)

        self.conv_geometry = nn.Sequential(nn.Conv1d(128+14, 256, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                         nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True))

        self.primitive_type = primitive_type
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

    def prob_field(self, rotation_, scale_, pos, p_type):
        bs = pos.shape[0]
        ctx = pos.device
        center_pos = torch.Tensor(np.zeros((bs,1,3))).to(ctx) #[bs 1 3]
        
        if p_type == 'cuboid':         
            primitive = box(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'sphere':
            primitive = sphere(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'ellipsoid':
            primitive = ellipsoid(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'cylinder':
            primitive = cylinder(size=scale_).rotate(rotation_)

        center_sdf = torch.abs(primitive(center_pos))
        bs, num_points = pos.shape[0], pos.shape[1]

        primitive_field = torch.abs(primitive(pos)+center_sdf)/(2*center_sdf+1e-10)
        tau = 4
        primitive_field = torch.exp(- tau * primitive_field)

        return primitive_field.reshape((bs, num_points, 1))
    
    def querying_occupancy(self, p_type, point_coord, rotation, scale, loc, density):
        points = point_coord - loc
        occupancy = self.prob_field(rotation, scale, points, p_type) * density
        return occupancy


    def primitive_deform(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        G_structure = []
        deformations = []
        deform_reg_loss = 0
        correction_reg_loss = 0
        
        p_type_dict = {0:'cuboid', 1:'cylinder', 2:'ellipsoid', 3:'sphere'}

        for split_i in range(self.num_template):

            p_type = self.primitive_type
            # occupancy = self.primitive_field(point_coord, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), split_i)
            
            occupancy1 = self.querying_occupancy(p_type, point_coord, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))
            deform_input = torch.cat([occupancy1, point_coord], dim=-1)
            G_structure.append(occupancy1)

            deform_out = self.deformers[0](deform_input, z_fusion[:,split_i,:].unsqueeze(1))
            deform_part = deform_out[:,:,:3]
            exist_part = torch.tanh(deform_out[:,:,3]).unsqueeze(-1)

            points = point_coord + deform_part
            occupancy2 = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))
            occupancy3 = torch.clamp(occupancy2+exist_part, 0, 1)
            
            G_.append(occupancy3)


            deform_mean_reg = deform_part.mean(-2, keepdim=True)
            deform_reg_loss = deform_reg_loss + ((deform_reg**2).sum(-1)) + ((deform_mean_reg**2).sum(-1)) * 0
            correction_reg_loss = correction_reg_loss + torch.abs(exist_part)
        
        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
        G_structure = torch.cat(G_structure, dim=-1)
        # deform_loss = (deform_reg_loss.mean()*50 + correction_reg_loss.mean()*100) / self.num_template
        deform_loss = (deform_reg_loss.mean()*100 + correction_reg_loss.mean()*0.1) / self.num_template
        # deform_loss = (deform_reg_loss.mean()*10 + correction_reg_loss.mean()*50) / self.num_template
        # deform_loss = (deform_reg_loss.mean()*10*10 + correction_reg_loss.mean()*2) / self.num_template
        # (100 1) for airplane (50, 0.1)
        # (30 100) for table
        return G_, G_structure, deform_loss


    def primitive_structure(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        p_type_dict = {0:'cuboid', 1:'cylinder', 2:'ellipsoid', 3:'sphere'}
        for split_i in range(self.num_template):
            p_type = self.primitive_type
            points = point_coord
            occupancy = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))

            G_.append(occupancy)

        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
        deform_loss = None
        return G_, deform_loss


    def forward(self, point_voxels, image, point_coord, is_training=False, apply_grad_loss=False):
        bs, N = point_coord.shape[0], point_coord.shape[1]
        ctx = point_voxels.device
        ################# feature extractor ##################
        instance_feat = self.SVR_encoder(image).unsqueeze(-1)  # [bs, z_dim, 1]


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

        ################# k occupancy indicator ##################
        # z_fusion, _ = self.mixing_parts(z_fusion)
        if self.stage2:
            G_, G_structure, deform_loss = self.primitive_deform(point_coord, z_fusion, rotation_, scale_, loc_, density_)

            if is_training:
                G_fps = G_[:,-1024:,:]
                G_sampled = G_[:, :-1024,:]
                G_structure_sampled = G_structure[:, :-1024,:]
            else:
                G_fps = None
                G_sampled = G_
                G_structure_sampled = G_structure
            G = torch.clamp(torch.max(G_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
            G_coarse = torch.clamp(torch.max(G_structure_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
        else:
            G_, deform_loss = self.primitive_structure(point_coord, z_fusion, rotation_, scale_, loc_, density_)
            G_structure = G_
            if is_training:
                G_fps = G_[:,-1024:,:]
                G_sampled = G_[:, :-1024,:]
                G_structure_sampled = G_structure[:, :-1024,:]
            else:
                G_fps = None
                G_sampled = G_
                G_structure_sampled = G_structure
            G = torch.clamp(torch.max(G_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
            G_coarse = torch.clamp(torch.max(G_structure_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)

        return {'G_': G_sampled,
                'G_fps': G_fps,
                'G_coarse': G_structure_sampled,
                'G': G,
               }

class DPF(nn.Module):
    def __init__(self, num_template=8, z_dim=256, part_dim=64, is_stage2=False, primitive_type='cuboid'):
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

        self.conv_scale = nn.Conv1d(128, 6, kernel_size=1, bias=True)
        # self.conv_scale.bias.data = torch.Tensor([0.1, 0.1, 0.1, 0.1, 0, 0])
        nn.init.zeros_(self.conv_scale.bias)

        self.conv_rotate = nn.Conv1d(128, 4, kernel_size=1, bias=True)
        self.conv_rotate.bias.data = torch.Tensor([1, 0, 0, 0])

        self.conv_trans = nn.Conv1d(128, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv_trans.bias)
        # self.conv_trans.bias.data = torch.Tensor([0.1,0.1,0.1])

        self.conv_density = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        # nn.Conv1d(64, 1, kernel_size=1, bias=True)
        # nn.init.zeros_(self.conv_density.bias)

        self.conv_geometry = nn.Sequential(nn.Conv1d(128+14, 256, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                         nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2, inplace = True))

        self.primitive_type = primitive_type
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


    def prob_field(self, rotation_, scale_, pos, p_type):
        bs = pos.shape[0]
        ctx = pos.device
        center_pos = torch.Tensor(np.zeros((bs,1,3))).to(ctx) #[bs 1 3]
        
        if p_type == 'cuboid':         
            primitive = box(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'sphere':
            primitive = sphere(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'ellipsoid':
            primitive = ellipsoid(center=center_pos, size=scale_).rotate(rotation_)
        elif p_type == 'cylinder':
            primitive = cylinder(size=scale_).rotate(rotation_)

        center_sdf = torch.abs(primitive(center_pos))
        bs, num_points = pos.shape[0], pos.shape[1]

        primitive_field = torch.abs(primitive(pos)+center_sdf)/(2*center_sdf+1e-10)
        tau = 4
        primitive_field = torch.exp(- tau * primitive_field)

        return primitive_field.reshape((bs, num_points, 1))

    def querying_occupancy(self, p_type, point_coord, rotation, scale, loc, density):
        points = point_coord
        occupancy = self.prob_field(rotation, scale, points, p_type) * density
        return occupancy

    def mixing_parts(self, part_code, mask=None):
        zh_, attn = self.mixing_network.forward_with_attention(part_code, mask=mask)
        return zh_, attn

    def primitive_deform(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        G_structure = []
        deform_reg_loss = 0
        correction_reg_loss = 0
        

        for split_i in range(self.num_template):

            p_type = self.primitive_type
            primitive_coord = point_coord - loc_[:,split_i,:].unsqueeze(1)
            occupancy1 = self.querying_occupancy(p_type, primitive_coord, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))

            deform_input = torch.cat([occupancy1, point_coord], dim=-1)
            G_structure.append(occupancy1)

            deform_out = self.deformers[0](deform_input, z_fusion[:,split_i,:].unsqueeze(1))
            deform_part = deform_out[:,:,:3]
            exist_part = torch.tanh(deform_out[:,:,3]).unsqueeze(-1)

            points = primitive_coord + deform_part
            occupancy2 = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))
            occupancy3 = torch.clamp(occupancy2+exist_part, 0, 1)
# 
            G_.append(occupancy3)

            deform_reg = deform_part# - deform_part.mean(-2, keepdim=True)
            deform_reg_loss = deform_reg_loss + ((deform_reg**2).sum(-1))
            correction_reg_loss = correction_reg_loss + torch.abs(exist_part)
        
        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
        G_structure = torch.cat(G_structure, dim=-1)
        # deform_loss = (deform_reg_loss.mean()*50 + correction_reg_loss.mean()*100) / self.num_template
        # deform_loss = (deform_reg_loss.mean()*30 + correction_reg_loss.mean()*0.01) / self.num_template
        # deform_loss = (deform_reg_loss.mean()*30 + correction_reg_loss.mean()*50) / self.num_template
        deform_loss = (deform_reg_loss.mean()*30 + correction_reg_loss.mean()*100) / self.num_template
        # deform_loss = (deform_reg_loss.mean()*10*10 + correction_reg_loss.mean()*2) / self.num_template
        # (100 1) for airplane
        # (30 100) for table
        return G_, G_structure, deform_loss

    def primitive_structure(self, point_coord, z_fusion, rotation_, scale_, loc_, density_):
        G_ = []
        for split_i in range(self.num_template):
            p_type = self.primitive_type
            points = point_coord
            occupancy = self.querying_occupancy(p_type, points, rotation_[:,split_i,:,:].unsqueeze(1), scale_[:,split_i,:].unsqueeze(1), loc_[:,split_i,:].unsqueeze(1), density_[:,split_i,:].unsqueeze(1))

            G_.append(occupancy)

        G_ = torch.cat(G_, dim=-1) # [bs, N, num_temp]
        deform_loss = None
        return G_, deform_loss


    def forward(self, point_voxels, point_coord, is_training=False, apply_grad_loss=False):
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

        structure = torch.cat([z_fusion, scale_, q_, loc_, density_], axis=2)
        z_geometry = self.conv_geometry(structure.transpose(2,1)).transpose(2,1)

        ################# k occupancy indicator ##################
        if self.stage2:
            G_, G_structure, deform_loss = self.primitive_deform(point_coord, z_geometry, rotation_, scale_, loc_, density_)
            if is_training:
                G_fps = G_[:,-1024:,:]
                G_sampled = G_[:, :-1024,:]
                G_structure_sampled = G_structure[:, :-1024,:]
            else:
                G_fps = None
                G_sampled = G_
                G_structure_sampled = G_structure
            G = torch.clamp(torch.max(G_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
            G_coarse = torch.clamp(torch.max(G_structure_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
        else:
            G_, deform_loss = self.primitive_structure(point_coord, z_geometry, rotation_, scale_, loc_, density_)
            G_structure = G_
            if is_training:
                G_fps = G_[:,-1024:,:]
                G_sampled = G_[:, :-1024,:]
                G_structure_sampled = G_structure[:, :-1024,:]
            else:
                G_fps = None
                G_sampled = G_
                G_structure_sampled = G_structure
            G = torch.clamp(torch.max(G_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)
            G_coarse = torch.clamp(torch.max(G_structure_sampled, axis=2, keepdims=True)[0], 0.0, 1.0)


        return {'G_': G_sampled,
                'G_fps': G_fps,
                'G_coarse': G_structure_sampled,
                'G': G,
                'center': loc_,
                'existence': density_,
                'G_coarse': G_coarse,
                'deformation_loss': deform_loss,
                'part_codes_loss': ((F.normalize(q_,dim=-1,p=2)-torch.Tensor([1,0,0,0]).to(ctx))**2).mean(),
                'global_feature_loss': None,
                'W_loss': None,
               }