import torch
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def l1_regularization(model, l1_alpha=0.00001):
    l1_loss = []
    model = model.module
    for name,param in model.named_parameters():
        if 'layer3.weight' in name:
            l1_loss.append(torch.abs(param).sum())
    return l1_alpha * sum(l1_loss)

def W_regularization(pred_dict, alpha=0.1):
    loss = pred_dict['W_loss']
    return alpha * loss
    # torch.abs(model.module.W-1).sum()

def deform_loss(pred_dict, alpha=0.01):
    loss = pred_dict['deformation_loss']
    return loss*alpha

def code_reg_loss(pred_dict, alpha=0.0001):
    loss = pred_dict['global_feature_loss']
    return loss*alpha

def part_code_reg_loss(pred_dict, alpha=0.0001):
    loss = pred_dict['part_codes_loss'] #[1, num_temp, part_dim]
    # ins_codes = pred_dict['ins_codes'] #[bs, num_temp, part_dim]
    # loss = (part_codes ** 2).mean()
    return loss*alpha


def chamfer_loss(pred_dict, gt_dict, alpha=0.3): 
    """Compute the bidirectional Chamfer loss between the target and the predicted shape.
    """
    # Extract the target surface points and some shapes into local variables
    targets = gt_dict['data_fps_points']
    B, N2, _ = targets.shape
    points = targets[:, :, :3]
    # Extract the predicted surface.

    N = torch.sum(torch.abs(pred_dict["G"]-0.5) < 0.05, dim=1).min().item()
    # print(N)
    if N == 0:
        return None
    valid_idx = torch.repeat_interleave((torch.abs(pred_dict["G"]-0.5) < 0.05).float(), 3, dim=-1)
    surface_idx = torch.topk(valid_idx, N, dim=1)[1]
    y_prim = torch.gather(gt_dict['data_points'],dim=1,index=surface_idx).reshape(B, -1, 3)
    # y_prim = [surface_idx].reshape(B, -1, 3)  # [bs x N x 3]
    _, N1, _ = y_prim.shape

    diff = y_prim[:, :, None] - points[:,None]
    dists = torch.square(diff).sum(-1)

    assert dists.shape == (B, N1, N2)

    target_to_prim = dists.min(1)[0].mean()
    prim_to_target = dists.min(2)[0].mean()
    loss = target_to_prim + prim_to_target

    return loss*alpha

# def compact_loss(pred_dict, gt_dict, eps=0.01, alpha=0.1):
#     pred_ = pred_dict['G_'] # [bs x num_points x num_part]
#     gt = gt_dict['data_values'] # [bs x num_points x 1]

#     return (torch.sum(torch.sqrt(torch.mean(pred_*gt, dim=1) + eps), dim=-1)**2).mean()*alpha

def center_loss(pred_dict, gt_dict, num_template=16, alpha=0.1):
    centering = pred_dict['center']
    existence = pred_dict['existence']
    G_fps = pred_dict['G_fps'].unsqueeze(-1)
    assign_matrix =  (G_fps == torch.max(G_fps, dim=-2, keepdim=True).values)
    # [bs x N x num_part x 1]
    fps_points = torch.repeat_interleave(gt_dict['data_fps_points'].unsqueeze(-2), num_template, dim=-2)
    # [bs x N x num_part x 3]
    assigned_points = assign_matrix * fps_points
    assigned_center = assigned_points.sum(1) / (assign_matrix.sum(1)+1)
    # print(assigned_center.mean(0),centering.mean(0))
    diff = torch.norm((assigned_center.detach() - centering) * existence.detach(), p = 2, dim = -1)
    diff = torch.mean(torch.mean(diff, -1), -1)
    return diff * alpha
    # [bs x 1 x num_part x 3]
    # G_fps = torch.softmax(G_fps, dim=-1)
    # G_fps = G_fps / G_fps.sum(-1, keepdims=True)
    # return (torch.sum(torch.sqrt(torch.mean(G_fps, dim=1) + eps), dim=-1)**2).mean()*alpha
    

def fps_compact_loss(pred_dict, gt_dict, eps=0.01, alpha=0.1):
    G_fps = pred_dict['G_fps'] # [bs x num_points x num_part]
    # G_fps = torch.softmax(G_fps, dim=-1)
    # G_fps = G_fps / (G_fps.sum(-1, keepdims=True)+1e-6)
    return (torch.sum(torch.sqrt(torch.mean(G_fps, dim=1) + eps), dim=-1)**2).mean()*alpha
    
def compact_loss(pred_dict, gt_dict, eps=0.01, alpha=0.1):
    pred_ = pred_dict['G_'] # [bs x num_points x num_part]
    gt = gt_dict['data_values'] # [bs x num_points x 1]
    W = (pred_ * gt)
    # pred_ = torch.softmax(pred_, dim=-1)
    # return (torch.sum(torch.sqrt(torch.mean(pred_, dim=1) + eps), dim=-1)**2).mean()*alpha
    # return (torch.sum(torch.abs(torch.sum(W, dim=1) / torch.sum(gt, dim=1)), dim=-1)).mean()*alpha
    return (torch.sum(torch.sqrt(torch.sum(W, dim=1) / torch.sum(gt, dim=1)  + eps), dim=-1)**2).mean()*alpha
    # return (torch.sum(torch.sqrt(torch.mean(pred_[:,:,:8] * gt, dim=1) + eps), dim=-1)**2).mean()*alpha + (torch.sum(torch.sqrt(torch.mean(pred_[:,:,8:] * gt, dim=1) + eps), dim=-1)**2).mean()*10


def min_points_loss(pred_dict, gt, min_points=3, alpha=0.01, tau=1.2):
    pred_ = pred_dict['G_']
    occupancy = torch.topk(pred_ * gt, min_points, dim=1, largest=True, sorted=False)[0] #[bs, min_points, parts]
    occupancy2 = torch.topk(pred_ * (1-gt), min_points, dim=1, largest=True, sorted=False)[0] #[bs, min_points, parts]
    # loss = -torch.log(occupancy+1e-6).mean() + torch.relu(torch.log(occupancy2+1e-6).mean())
    loss = - torch.log(occupancy+1e-6).sum(-2).mean() - (torch.log(1-occupancy2+1e-6).sum(-2).mean())
    # loss = torch.relu(torch.log(occupancy2+1e-6).sum(-2).mean())
    return loss*alpha

def overlap_loss(pred_dict, gt, tau=1.5, alpha=0.1):
    pred_ = pred_dict["G_"]
    pred = torch.clamp(torch.max(pred_, axis=2, keepdims=True)[0], 0.0, 1.0)
    # pred = pred_dict["G"]
    return ((torch.sum(pred_, dim=-1, keepdims=True)-pred)**2).mean()*alpha
    # return (F.relu(torch.sum(pred_, dim=-1)-tau)**2).mean()*alpha


def compute_KLD(pred_dict, alpha=0.0003):
    mu , log_var = pred_dict['mu'], pred_dict['log_var']
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kld_loss * alpha

def homeo_loss(pred_dict, gt_dict, tau=0.01, k0=0.5, alpha=0.1):
    pred = pred_dict["G"].squeeze()
    gt = gt_dict["data_values"].squeeze()
    loss = torch.nn.functional.mse_loss(torch.sigmoid((pred-k0)/tau), gt)
    # loss = (F.relu(torch.sum(pred_, dim=-1)-tau)**2).mean()
    return loss*alpha

def mse_loss(pred, gt):
    # loss = torch.nn.functional.mse_loss(pred, gt)
    # loss_dict['mse_loss'] = loss
    return (torch.sum((pred-gt)**2, dim=-1)).mean(-1).mean()
    # return loss


def all_loss(args, pred_dict, gt_dict, epoch):
    if args.SVR:
        loss_dict = {}
        loss_dict['svr_loss'] = pred_dict['svr_loss']
        return loss_dict
    else:
        pred_ = pred_dict["G_"].float()
        pred = pred_dict["G"].float()
        pred_coarse = pred_dict["G_coarse"].float()
        gt = gt_dict['data_values']

        loss_dict = {}

        # loss = torch.nn.functional.mse_loss(pred, gt)

        # loss_dict['recon_loss'] = mse_loss(pred, gt) * 10 #+ homeo_loss(pred_dict, gt_dict) * 10

        # loss_dict['recon_coarse_loss'] = mse_loss(pred_coarse, gt) 

        if pred_dict['global_feature_loss'] != None:
            loss_dict['code_reg'] = code_reg_loss(pred_dict)    

        if pred_dict['part_codes_loss'] != None:
            loss_dict['part_codes_loss'] = part_code_reg_loss(pred_dict)

        if pred_dict['deformation_loss'] != None:
            loss_dict['deform_loss'] = deform_loss(pred_dict)
        
        if pred_dict['W_loss'] != None:
            loss_dict['W_loss'] = W_regularization(pred_dict)
            
        # for chair
        # category = 'chair'
        if 'airplane' in args.shapenet_path:
            category = 'airplane'
        elif 'table' in args.shapenet_path:
            category = 'table'
        elif 'chair' in args.shapenet_path:
            category = 'chair'

        if category == 'chair':
            loss_dict['recon_loss'] = mse_loss(pred, gt) * 1
            loss_dict['coarse_recon_loss'] = mse_loss(pred_coarse, gt) * 1
            loss_dict['center_loss'] = center_loss(pred_dict, gt_dict, args.num_part, alpha=1)
            loss_dict['fps_compact_loss'] = fps_compact_loss(pred_dict, gt_dict, alpha=0.0001)
            loss_dict['part_codes_loss'] = loss_dict['part_codes_loss']
        elif category == 'airplane':
            # for airplane
            loss_dict['recon_loss'] = mse_loss(pred, gt) * 1
            loss_dict['coarse_recon_loss'] = mse_loss(pred_coarse, gt) * 1
            loss_dict['center_loss'] = center_loss(pred_dict, gt_dict, args.num_part, alpha=1)
            loss_dict['fps_compact_loss'] = fps_compact_loss(pred_dict, gt_dict, alpha=0.0001)
            
            loss_dict['part_codes_loss'] = loss_dict['part_codes_loss']
        elif category == 'table':
            # for table
            loss_dict['recon_loss'] = mse_loss(pred, gt) * 1
            loss_dict['coarse_recon_loss'] = mse_loss(pred_coarse, gt) * 1
            loss_dict['center_loss'] = center_loss(pred_dict, gt_dict, args.num_part, alpha=1)
            loss_dict['fps_compact_loss'] = fps_compact_loss(pred_dict, gt_dict, alpha=0.0001)
            loss_dict['part_codes_loss'] = loss_dict['part_codes_loss']
    return loss_dict
