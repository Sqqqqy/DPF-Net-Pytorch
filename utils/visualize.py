import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import mcubes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gc 
import os
from point_cloud_utils import k_nearest_neighbors
import warnings
warnings.filterwarnings('ignore')


import random
import colorsys
def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = []
    # np.zeros((0,3))
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append(str(r)+' '+str(g)+' '+str(b))
    return rgb_colors

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[-np.sin(theta), 0, np.cos(theta)],[0,1,0],]


def sample_points_from_mesh(vertices_t, triangles_t):
    mesh = trimesh.Trimesh(vertices_t, triangles_t)
    # mesh.vertices = o3d.utility.Vector3dVector(vertices_t)
    # mesh.triangles = o3d.utility.Vector3iVector(triangles_t)

    ps = mesh.sample(4096)
    ps = np.array(ps).astype(np.float32).reshape((1,-1,3))
    return ps

def compute_iou(occ1, occ2, IOU_Resolution=64):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    
    # x_min, y_min, z_min = np.where(occ2.sum(-1).sum(-1)>0)[0].min(), np.where(occ2.sum(0).sum(-1)>0)[0].min(), np.where(occ2.sum(0).sum(0)>0)[0].min()
    # x_max, y_max, z_max = np.where(occ2.sum(-1).sum(-1)>0)[0].max(), np.where(occ2.sum(0).sum(-1)>0)[0].max(), np.where(occ2.sum(0).sum(0)>0)[0].max()
	
    # sampled_iou = np.array([np.random.sample(100000),np.random.sample(100000),np.random.sample(100000)]).transpose()
    # sampled_iou[:,0] = sampled_iou[:,0]*(x_max-x_min)/IOU_Resolution + x_min/IOU_Resolution
    # sampled_iou[:,1] = sampled_iou[:,1]*(y_max-y_min)/IOU_Resolution + y_min/IOU_Resolution
    # sampled_iou[:,2] = sampled_iou[:,2]*(z_max-z_min)/IOU_Resolution + z_min/IOU_Resolution
    
    # sampled_iou = np.around(sampled_iou*(IOU_Resolution-0.5)).astype('int')
	
    # occ1 = occ1[sampled_iou[:,0],sampled_iou[:,1],sampled_iou[:,2]]
    # occ2 = occ2[sampled_iou[:,0],sampled_iou[:,1],sampled_iou[:,2]]
    maxpooling = torch.nn.MaxPool3d(2,stride=2)
    occ1 = maxpooling(torch.tensor(occ1).float().unsqueeze(0)).squeeze(0).numpy()
    occ2 = maxpooling(torch.tensor(occ2).float().unsqueeze(0)).squeeze(0).numpy()

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum()
    area_intersect = (occ1 & occ2).astype(np.float32).sum()

    iou = area_intersect / area_union

    return iou


def sym_CD(x, y, return_index=False, p_norm=2, max_points_per_leaf=10):
    """
    Compute the chamfer distance between two point clouds x, and y
    Parameters
    ----------
    x : A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    y : A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    return_index: If set to True, will return a pair (corrs_x_to_y, corrs_y_to_x) where
                  corrs_x_to_y[i] stores the index into y of the closest point to x[i]
                  (i.e. y[corrs_x_to_y[i]] is the nearest neighbor to x[i] in y).
                  corrs_y_to_x is similar to corrs_x_to_y but with x and y reversed.
    max_points_per_leaf : The maximum number of points per leaf node in the KD tree used by this function.
                          Default is 10.
    p_norm : Which norm to use. p_norm can be any real number, inf (for the max norm) -inf (for the min norm),
             0 (for sum(x != 0))
    Returns
    -------
    The chamfer distance between x an dy.
    If return_index is set, then this function returns a tuple (chamfer_dist, corrs_x_to_y, corrs_y_to_x) where
    corrs_x_to_y and corrs_y_to_x are described above.
    """

    dists_x_to_y, corrs_x_to_y = k_nearest_neighbors(x, y, k=1,
                                                     squared_distances=False,
                                                     max_points_per_leaf=max_points_per_leaf)
    dists_y_to_x, corrs_y_to_x = k_nearest_neighbors(y, x, k=1,
                                                     squared_distances=False,
                                                     max_points_per_leaf=max_points_per_leaf)
    if p_norm == 2:
        dists_x_to_y = (np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=p_norm)**2).mean()
        dists_y_to_x = (np.linalg.norm(y[corrs_x_to_y] - x, axis=-1, ord=p_norm)**2).mean()
    else:
        dists_x_to_y = (np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=p_norm)).mean()
        dists_y_to_x = (np.linalg.norm(y[corrs_x_to_y] - x, axis=-1, ord=p_norm)).mean()
    cham_dist = np.mean(dists_x_to_y) + np.mean(dists_y_to_x)

    if return_index:
        return cham_dist, corrs_x_to_y, corrs_y_to_x
    return np.mean(dists_x_to_y), np.mean(dists_y_to_x)


def distance_p2p(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    
    B, N2, _ = points_tgt.shape #[bs, N, 3]
    _, N1, _ = points_src.shape #[bs, N, 3]

    prim_to_target_L2, target_to_prim_L2 = sym_CD(points_tgt.reshape(-1,3), points_src.reshape(-1,3), p_norm=2)
    prim_to_target_L1, target_to_prim_L1 = sym_CD(points_tgt.reshape(-1,3), points_src.reshape(-1,3), p_norm=1)
    return {
                'chamfer_L2': [prim_to_target_L2, target_to_prim_L2],
                'chamfer_L1': [prim_to_target_L1, target_to_prim_L1],
           }


def eval_pointcloud(pointcloud, pointcloud_tgt):
        ''' Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        #occnet计算cd前，先输入1+padding的采样点得到reconstructed mesh，这里label是充满整个空间的。
        #从reconstructed mesh uniform采样100k点，从normalize到0~1之后的target mesh采样100k点，如此计算cd
        
        #ours计算cd前，先输入-0.5：0.5：1/resolution的采样点得到reconstructed mesh，这里的label未充满整个空间。
        #从reconstructed mesh uniform采样4096点，从normalize到0~1之后的target mesh（mcube from voxel）采样4096点，如此计算cd
        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        chamfer_distance = distance_p2p(
            pointcloud, pointcloud_tgt
        )
        
        # Chamfer distance
        chamferL1 = 0.5 * (chamfer_distance['chamfer_L1'][0] + chamfer_distance['chamfer_L1'][1])
        chamferL2 = 0.5 * (chamfer_distance['chamfer_L2'][0] + chamfer_distance['chamfer_L2'][1])
        print('chamferL1:',chamferL1,' chamferL2:',chamferL2)
        return chamferL1, chamferL2


def save_recon(model, data_loader, args, sample_num=20, part_thres=0.4, Resolution=32):

    model.eval()
    num_part = model.module.num_template

    
    space_3d = np.array([[x,y,z] for x in np.arange(-0.5,0.5,1/Resolution) for y in np.arange(-0.5,0.5,1/Resolution) for z in np.arange(-0.5,0.5,1/Resolution)])
    x_cords, y_cords, z_cords = space_3d.transpose()
    space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()
    
    for data_idx, data in enumerate(data_loader):
        if data_idx > sample_num:
            break
        test_voxels, _, _, _ = data
        test_voxels = Variable(test_voxels).cuda().float()
        test_voxels = test_voxels.reshape((1,1,64,64,64))
        model_out = model(test_voxels, space_3d)['G_'].cpu().detach().numpy()
        
        thres = [part_thres for i in range(num_part)]
        vertices_num, triangles_num = 0, 0
        vertices_list, triangles_list = [], []
        vertices_num_list = [0]

        for part_i in range(num_part):
            voxel = np.zeros((Resolution, Resolution, Resolution))
            for i, (x, y, z) in enumerate(zip(x_cords, y_cords, z_cords)):
                x,y,z = map(lambda y:int((y+0.5)*Resolution), [x,y,z])
                voxel[x][y][z] = model_out[0,i,part_i]
            vertices, triangles = mcubes.marching_cubes(voxel, thres[part_i])
            
            vertices_num += len(vertices)
            triangles_num += len(triangles)
            vertices_list.append(vertices)
            # @rotMat(np.pi/2)
            triangles_list.append(triangles)
            vertices_num_list.append(vertices_num)
        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        fout = open(args.result_dir+'/'+f"sample_{data_idx}_shapenet_recon.ply", 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(vertices_num)+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("element face "+str(triangles_num)+"\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")

        for split in range(num_part):
            vertices = ((vertices_list[split])/Resolution-0.5)#
            for i in range(len(vertices)):
                color = color_list[split]
                fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
  
        for split in range(num_part):
            triangles = (triangles_list[split] + vertices_num_list[split])
            for i in range(len(triangles)):
                color = color_list[split]
                fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")

def save_field(model, data_loader, args, sample_num=20, part_thres=0.4, Resolution=32):

    model.eval()
    num_part = model.module.num_template

    
    space_3d = np.array([[x,y,z] for x in np.arange(-0.5,0.5,1/Resolution) for y in np.arange(-0.5,0.5,1/Resolution) for z in np.arange(-0.5,0.5,1/Resolution)])
    x_cords, y_cords, z_cords = space_3d.transpose()
    space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()
    space_3d_np = space_3d.cpu().numpy().reshape(-1,3)
    for data_idx, data in enumerate(data_loader):
        if data_idx > sample_num:
            break
        test_voxels, _, _, _ = data
        test_voxels = Variable(test_voxels).cuda().float()
        test_voxels = test_voxels.reshape((1,1,64,64,64))
        model_out = model(test_voxels, space_3d)['G_'].cpu().detach().numpy().reshape(-1,num_part)
        
        valid_idx = np.max(model_out, axis=-1) > 3e-2
        model_out = model_out[valid_idx,:]
        valid_space_3d_np = space_3d_np[valid_idx,:]

        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        vertices_num = len(valid_space_3d_np)
        fout = open(args.result_dir+'/'+f"sample_{data_idx}_shapenet_recon.ply", 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(vertices_num)+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("end_header\n")

        
        vertices = valid_space_3d_np#
        for i in range(len(vertices)):
            color = f"{int(np.max(model_out[i,:])*255)} 0 0"
            fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")


def sample_points_from_mesh(vertices_t, triangles_t):
    mesh = trimesh.Trimesh(vertices_t, triangles_t)
    # mesh.vertices = o3d.utility.Vector3dVector(vertices_t)
    # mesh.triangles = o3d.utility.Vector3iVector(triangles_t)

    ps = mesh.sample(4096)
    ps = np.array(ps).astype(np.float32).reshape((1,-1,3))
    return ps

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def test_metric(model, data_loader, args, sample_num=20, part_thres=0.4, Resolution=32):

    chamfer_metric = []
    model.eval()

    shapenet_filename_list = open('/gpfs/home/sist/cq14/kBAE/dataset/data/'+args.category_info+'/'+args.category_id+'_test_vox.txt',"r").read().splitlines()

    num_part = model.module.num_template
    space_3d = np.array([[x,y,z] for x in np.arange(-0.5,0.5,1/Resolution) for y in np.arange(-0.5,0.5,1/Resolution) for z in np.arange(-0.5,0.5,1/Resolution)])
    space_3d += np.array([1/Resolution/2, 1/Resolution/2, 1/Resolution/2])
    x_cords, y_cords, z_cords = space_3d.transpose()
    space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()
    
    for data_idx, data in tqdm(enumerate(data_loader)):
        if data_idx > sample_num:
            break
        test_voxels, _, _, _ = data
        test_voxels_tensor = Variable(test_voxels).cuda().float()
        test_voxels_tensor = test_voxels_tensor.reshape((1,1,64,64,64))
        model_out = model(test_voxels_tensor, space_3d)['G'].cpu().detach().numpy()
        
        thres = [part_thres for i in range(num_part)]
        vertices_num, triangles_num = 0, 0
        vertices_list, triangles_list = [], []
        vertices_num_list = [0]

        # voxel = np.zeros((Resolution, Resolution, Resolution))
        # for i, (x, y, z) in enumerate(zip(x_cords, y_cords, z_cords)):
        #     x,y,z = map(lambda y:int((y+0.5)*Resolution), [x,y,z])
        #     voxel[x][y][z] = model_out[0,i]
        voxel = np.reshape(model_out, [Resolution,Resolution,Resolution])
        vertices, triangles = mcubes.marching_cubes(voxel, thres[0])

        if use_gt_mesh:
            gt_mesh = trimesh.load_mesh('/gpfs/home/sist/cq14/kBAE/dataset/shapenet_gt/'+args.category_id+'/'+shapenet_filename_list[data_idx]+'/model.obj')
            gt_mesh = as_mesh(gt_mesh)
            gt_ps = gt_mesh.sample(4096)
            
            gt_ps = np.array(gt_ps).astype(np.float32).reshape((1,-1,3)) #+ 0.5
            # print(gt_ps.min(), gt_ps.max())
            gt_ps[:,:,[0,-1]] = gt_ps[:,:,[-1,0]].copy()
            gt_ps[:,:,2]=-gt_ps[:,:,2]
            pred_ps = sample_points_from_mesh(vertices, triangles)
            pred_ps = ((pred_ps)/Resolution) - 0.5 + 1/Resolution/2
            
        else:  
            vertices_gt, triangles_gt = mcubes.marching_cubes(test_voxels[0,0,:,:,:].numpy(), thres[0])
            gt_ps = sample_points_from_mesh(vertices_gt, triangles_gt)
            gt_ps[:,:,[0,-1]] = gt_ps[:,:,[-1,0]].copy()
            gt_ps[:,:,2]=-gt_ps[:,:,2]
            gt_ps = ((gt_ps)/64)

            pred_ps = sample_points_from_mesh(vertices, triangles)
            # pred_ps = (pred_ps+0.5)*Resolution
            # gt_ps = (gt_ps+0.5)*64
            pred_ps = ((pred_ps)/Resolution-1/Resolution/2) 
        
        gt_voxels = test_voxels.numpy().reshape([1,1,64,64,64])[0,0,:,:,:]
        # gt_voxels = gt_voxels.transpose(2,1,0)
        # gt_voxels[:,:,:] = gt_voxels[:,:,::-1]
        # [indices[:,2],indices[:,1],15-indices[:,0]]
        viou = compute_iou(voxel, gt_voxels)
        print(viou)
        chamferL1, chamferL2 = eval_pointcloud(pred_ps, gt_ps)
        chamfer_metric.append([chamferL1, chamferL2, viou])

        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        fout = open(args.result_dir+'/'+f"miou{np.round(chamferL2, 4):.4f}_sample_{data_idx}_shapenet_recon.ply", 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(pred_ps.reshape(-1,3)) + len(gt_ps.reshape(-1,3)))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("end_header\n")

        vertices = pred_ps.reshape(-1,3)
        for i in range(len(vertices)):
            color = color_list[0]
            fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
  
        vertices = gt_ps.reshape(-1,3)
        for i in range(len(vertices)):
            color = color_list[1]
            fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        fout.close()
    print(np.array(chamfer_metric).mean(0))

def grouping(path):
    import numpy as np
    from tqdm import tqdm
    import os

    color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255",
                "180 180 180", "100 100 100",
                "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255",
                "180 100 255","100 180 255"] + ["240 0 0","0 240 0","0 0 240","240 240 0","240 0 240","0 240 240",
                "200 200 200", "60 60 60",
                "240 140 140","140 240 140","140 140 240","240 240 140","240 140 240","140 240 240",
                "200 60 240","60 200 240"]

    color_map = {}
    for i in range(len(color_list)):
        color_map[color_list[i]] = i
    group_set = {}

    filenames=os.listdir(f"{path}")

    for filename in tqdm(filenames):
        with open(f"{path}/{filename}", 'r') as f:
            lines = f.readlines()
            num_points = int(lines[2].split(' ')[-1]) // 2
            points = [line.strip().split(' ') for line in lines[10:]]
            # points = np.array(points).astype('float')
            # print(set(points[:num_points,-3:]))
            pc1 = points[:num_points]
            pc2 = points[num_points:]
            
            # gt_set = {}
            # for i in range(len(pc2)):
            #     gt_set[(pc2[i][3], pc2[i][4], pc2[i][5])] = 1

            # pred_set = {}
            # for key in gt_set.keys():
            #     gt_set[key] = []

            # for label in gt_set.keys():
            #     if pc2[i][0], pc2[i][4], pc2[i][5]
            #         pred_set[label].append((pc1[i][3], pc1[i][4], pc1[i][5]))

            # print(pred_set)
            
            for i in range(num_points):
                gt_label = pc2[i][3]+' '+pc2[i][4]+' '+pc2[i][5]
                pred_label = pc1[i][3]+' '+pc1[i][4]+' '+pc1[i][5]

                gt_label = color_map[gt_label]
                pred_label = color_map[pred_label]

                if not group_set.__contains__(gt_label):
                    group_set[gt_label] = {}
                else:
                    if not group_set[gt_label].__contains__(pred_label):
                        group_set[gt_label][pred_label] = 1
                    else:
                        group_set[gt_label][pred_label] += 1

            # for key in group_set.keys():
            #     group_set[key] = set(group_set[key])

    print(group_set)

def test_partnet_pointcloud(model, data_loader, args, use_post_processing=True, sample_num=20, part_thres=0.4, Resolution=32):
    model.eval()

    num_part = model.module.num_template
    shape_mIOU = []

    if args.category_info == '04379243_table':
        import json
        with open('/gpfs/home/sist/cq14/kBAE/dataset/shapenet/04379243_table/partnet_table.json', 'r') as f:
            partnet_json = json.load(f)
            f.close()
        ref_points = np.array(partnet_json['points'])
        ref_values = np.array(partnet_json['gt_label'])
        shapenet2partnet = partnet_json['shapenet2partnet']
        shapenet_ids = np.array(list(shapenet2partnet.values())).astype('int').tolist()
        valid_id = partnet_json['valid_id']
    print(shapenet_ids)
    for data_idx, data in enumerate(data_loader):
        if data_idx not in shapenet_ids:
            continue
        
        partnet_id = np.where(np.array(shapenet_ids)==data_idx)[0][0]
        
        test_voxels, _, _, _ = data
        test_voxels_tensor = Variable(test_voxels).cuda().float()
        test_voxels_tensor = test_voxels_tensor.reshape((1,1,64,64,64))
        b_point_num = ref_point_num[data_idx]
        branch_value = ref_values[partnet_id, :b_point_num]

        space_3d = ref_points[partnet_id, :b_point_num].reshape((1,-1,3))
        space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()

        
        model_out = model(test_voxels_tensor, space_3d)['G_'].cpu().detach().numpy()

        space_3d = space_3d.cpu().numpy().reshape((-1, 3))
        model_out = model_out.reshape(-1,model_out.shape[-1])

        label_gt = branch_value.reshape(-1,1).astype('int')
        label_out = np.argmax(model_out, axis=1)
        
        if use_post_processing:
            from sklearn.neighbors import KDTree
            valid_labels = np.max(model_out, axis=1)>1e-2
            valid_branch_coord = space_3d[valid_labels]
            valid_pred_part_labels = label_out[valid_labels]
            kd_tree = KDTree(valid_branch_coord, leaf_size=8)
            _, closest_idx = kd_tree.query(space_3d)
            label_out = valid_pred_part_labels[np.reshape(closest_idx,[-1])]

            label_out = label_out.reshape(-1,1)
        # label_out = np.argmax(model_out, axis=-1).reshape(-1,1)
        
        # ######################## airplane
        # group_map = {2:0, 1:2, 0:2, 6:1, 5:1, 7:1}
        # # group_map = {2:0, 1:2, 0:2, 6:1, 5:1, 7:3}# best 8cuboid{14:0, 13:2, 7:2, 8:2, 4:1, 6:1, 11:3,}
        # for i in range(len(label_out)):
        #     if group_map.__contains__(label_out[i,0]):
        #         label_out[i,0] = group_map[label_out[i,0]]
        #     else:
        #         label_out[i,0] = 0
        
        ######################## chair

        # group_map = {15:0, 2:0, 13:0, 9:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best
        # group_map = {15:0, 2:0, 13:0, 9:1, 1:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best structure
        # group_map = {1:0, 10:1, 14:2, 8:2, 3:2, 2:2, 0:2, 12:3, 6:3} # best 8_8
        # group_map = {1:0, 11:0, 10:1, 12:2, 2:2, 5:2, 14:2} # best 16cylinder
        # group_map = {8:0, 2:0, 4:1, 11:1, 14:2, 5:2, 3:2, 1:2, 13:3} # no_deformer, 16cuboid
        # group_map = {12:0, 11:1, 5:2, 14:2, 7:2, 1:2} # no_deformer, 16cylinder
        # for i in range(len(label_out)):
        #     if group_map.__contains__(label_out[i,0]):
        #         label_out[i,0] = group_map[label_out[i,0]]
        #     else:
        #         label_out[i,0] = 0

        ######################## table
        # gt_map = {
        #     0:0,
        #     1:1,
        #     2:1,
        # }
        # # best
        # group_map = {4:0, 8:0, 12:0}
        # # best 16cylinder
        # group_map = {12:0, 10:0, 8:0, 2:0}
        # # best 16cuboid_nodeformer
        # group_map =  {15:0, 1:0, 8:0, 13:0}
        # # best 16cylinder_nodeformer
        # group_map = {10:0, 9:0, 15:0, 8:0}
        
        # for i in range(len(label_out)):
        #     if group_map.__contains__(label_out[i,0]):
        #         label_out[i,0] = group_map[label_out[i,0]]
        #     else:
        #         label_out[i,0] = 1
        # for i in range(len(label_gt)):
        #     label_gt[i,0] = gt_map[label_gt[i,0]]


        ##################################
        part_ious = [0.0] * len(labels_unique)
        for i in range(len(labels_unique)):
            if (np.sum(label_gt==i) == 0) and (np.sum(label_out==i) == 0): # part is not present, no prediction as well
                part_ious[i] = 1.0
            else:
                part_ious[i] = np.sum(( label_gt==i ) & ( label_out==i )) / float(np.sum(   ( label_gt==i ) | ( label_out==i ) ))

        shape_mIOU.append(np.mean(part_ious))
        print(f'sample: {data_idx}, m-IOU: {np.mean(part_ious)}')
        #output ply
        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        fout = open(args.result_dir+'/'+f"sample_{data_idx}_shapenet_recon.ply", 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(b_point_num*2)+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("end_header\n")

        vertices = space_3d.reshape(-1,3)
        for i in range(b_point_num):
            color = color_list[label_out[i,0]]
            fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
        vertices = space_3d.reshape(-1,3)
        for i in range(b_point_num):
            color = color_list[label_gt[i,0]]
            fout.write(str(vertices[i,0]+0.8)+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
        fout.close()
    cate_mIOU = np.mean(shape_mIOU)*100.0
    grouping(args.result_dir)
    print(cate_mIOU)


def test_pointcloud(model, data_loader, args, use_post_processing=True, sample_num=20, part_thres=0.4, Resolution=32):
    model.eval()

    num_part = model.module.num_template
    shape_mIOU = [None] * len(data_loader)

    for data_idx, data in enumerate(data_loader):
        test_voxels, _, _, _ = data
        test_voxels_tensor = Variable(test_voxels).cuda().float()
        test_voxels_tensor = test_voxels_tensor.reshape((1,1,64,64,64))
        b_point_num = ref_point_num[data_idx]
        branch_value = ref_values[data_idx, :b_point_num]

        space_3d = ref_points[data_idx, :b_point_num].reshape((1,-1,3))
        space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()

        
        model_out = model(test_voxels_tensor, space_3d)['G_'].cpu().detach().numpy()

        space_3d = space_3d.cpu().numpy().reshape((-1, 3))
        model_out = model_out.reshape(-1,model_out.shape[-1])

        # group_map = {5: 0, 6: 0, 7: 0}
        # for i in group_map.keys():
        #     model_out[:,i] += 0.1
            
        label_gt = np.argmax(branch_value, axis=-1).reshape(-1,1)
        label_out = np.argmax(model_out, axis=1)
        
        if use_post_processing:
            from sklearn.neighbors import KDTree
            valid_labels = np.max(model_out, axis=1)>1.0e-1
            valid_branch_coord = space_3d[valid_labels]
            valid_pred_part_labels = label_out[valid_labels]
            kd_tree = KDTree(valid_branch_coord, leaf_size=8)
            _, closest_idx = kd_tree.query(space_3d)
            label_out = valid_pred_part_labels[np.reshape(closest_idx,[-1])]

            label_out = label_out.reshape(-1,1)
        # label_out = np.argmax(model_out, axis=-1).reshape(-1,1)
        
        ######################## airplane
        # group_map = {0: 2, 1: 1, 2: 2, 3: 0, 4: 1, 5: 3, 6: 0, 7: 2}#best 8cuboid 30 0.1
        
        # # {1:0, 0:1, 5:1, 6:2, 8:2, 4:2, 9:3, 7:3}#best 10cuboid
        # #{13:0, 14:0, 5:1, 4:1, 3:2, 8:2, 12:1, 9:1} #best 16cuboid
        # # {7:0, 15:0, 9:1, 8:1, 5:2, 14:2, 12:2, 10:1,3:1}#best 16cylinder
        
        # # group_map = {2:0, 1:2, 0:2, 6:1, 5:1, 7:3}# best 8cuboid{14:0, 13:2, 7:2, 8:2, 4:1, 6:1, 11:3,}
        # for i in range(len(label_out)):
        #     if group_map.__contains__(label_out[i,0]):
        #         label_out[i,0] = group_map[label_out[i,0]]
        #     else:
        #         label_out[i,0] = 0
        
        ######################## chair

        group_map = {15:0, 2:0, 13:0, 9:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best
        group_map = {15:0, 2:0, 13:0, 9:1, 1:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best structure
        group_map = {1:0, 10:1, 14:2, 8:2, 3:2, 2:2, 0:2, 12:3, 6:3} # best 8_8
        group_map = {1:0, 11:0, 10:1, 12:2, 2:2, 5:2, 14:2} # best 16cylinder
        group_map = {8:0, 2:0, 4:1, 11:1, 14:2, 5:2, 3:2, 1:2, 13:3} # no_deformer, 16cuboid
        group_map = {12:0, 11:1, 5:2, 14:2, 7:2, 1:2} # no_deformer, 16cylinder
        group_map = {0: 2, 1: 2, 2: 3, 3: 0, 4: 3, 5: 2, 6: 1, 7: 0, 8: 1, 9: 2, 10: 0, 11: 0, 12: 1, 13: 1, 14: 2, 15: 1} 
        group_map = {0: 2, 1: 0, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 2, 8: 0, 9: 0, 10: 0, 11: 0} # 12cuboid
        group_map = {0: 1, 1: 2, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 3, 9: 2, 10: 2, 11: 0, 12: 0, 13: 2, 14: 1, 15: 2, 16: 0, 17: 0, 18: 0, 19: 0}
        for i in range(len(label_out)):
            if group_map.__contains__(label_out[i,0]):
                label_out[i,0] = group_map[label_out[i,0]]
            else:
                label_out[i,0] = 0

        ######################## table
        # gt_map = {
        #     0:0,
        #     1:1,
        #     2:1,
        # }
        # # best
        # group_map = {4:0, 8:0, 12:0}
        # # best 16cylinder
        # group_map = {12:0, 10:0, 8:0, 2:0}
        # # best 16cuboid_nodeformer
        # group_map =  {15:0, 1:0, 8:0, 13:0}
        # # best 16cylinder_nodeformer
        # group_map = {10:0, 9:0, 15:0, 8:0}
        # # group_map = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0}#{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0}
        # # group_map = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0} #best 8cuboid
        # group_map = {12:0, 10:0, 8:0, 2:0}
        # for i in range(len(label_out)):
        #     if group_map.__contains__(label_out[i,0]):
        #         label_out[i,0] = group_map[label_out[i,0]]
        #     else:
        #         label_out[i,0] = 1
        
        # gt_map = {
        #     0:0,
        #     1:1,
        #     2:1,
        # }
        # for i in range(len(label_gt)):
        #     label_gt[i,0] = gt_map[label_gt[i,0]]


        ##################################
        part_ious = [0.0] * len(labels_unique)
        for i in range(len(labels_unique)):
            if (np.sum(label_gt==i) == 0) and (np.sum(label_out==i) == 0): # part is not present, no prediction as well
                part_ious[i] = 1.0
            else:
                part_ious[i] = np.sum(( label_gt==i ) & ( label_out==i )) / float(np.sum(   ( label_gt==i ) | ( label_out==i ) ))

        shape_mIOU[data_idx] = np.mean(part_ious)
        print(f'sample: {data_idx}, m-IOU: {np.mean(part_ious)}')
        #output ply
        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        fout = open(args.result_dir+'/'+f"miou{np.round(np.mean(part_ious), 3):.3f}_sample_{data_idx}_shapenet_recon.ply", 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(b_point_num*2)+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("end_header\n")

        vertices = space_3d.reshape(-1,3)
        for i in range(b_point_num):
            color = color_list[label_out[i,0]]
            fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
        vertices = space_3d.reshape(-1,3)
        for i in range(b_point_num):
            color = color_list[label_gt[i,0]]
            fout.write(str(vertices[i,0]+0.8)+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
        fout.close()
    cate_mIOU = np.mean(shape_mIOU)*100.0
    grouping(args.result_dir)
    print(cate_mIOU)

# def test_pointcloud(model, data_loader, args, use_post_processing=True, sample_num=20, part_thres=0.4, Resolution=32):
#     model.eval()

#     num_part = model.module.num_template
#     shape_mIOU = [None] * len(data_loader)

#     for data_idx, data in enumerate(data_loader):
#         test_voxels, _, _, _ = data
#         test_voxels_tensor = Variable(test_voxels).cuda().float()
#         test_voxels_tensor = test_voxels_tensor.reshape((1,1,64,64,64))
#         b_point_num = ref_point_num[data_idx]
#         branch_value = ref_values[data_idx, :b_point_num]

#         space_3d = ref_points[data_idx, :b_point_num].reshape((1,-1,3))
#         space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()

        
#         model_out = model(test_voxels_tensor, space_3d)['G_'].cpu().detach().numpy()


#         vertices_gt, triangles_gt = mcubes.marching_cubes(test_voxels[0,0,:,:,:].numpy(), 0.5)
#         gt_mesh = trimesh.Trimesh(vertices_gt, triangles_gt)

#         space_3d = space_3d.cpu().numpy().reshape((-1, 3))
#         model_out = model_out.reshape(-1,model_out.shape[-1])

#         # ps = gt_mesh.sample(4096)
#         # ps = np.array(ps).astype(np.float32).reshape((1,-1,3))
#         # print(ps.min(), ps.max(), space_3d.min(), space_3d.max())


#         surface_idx = (~gt_mesh.contains(space_3d*64+30.5)) | (~gt_mesh.contains(space_3d*64+32.5))
#         label_gt = np.argmax(branch_value, axis=-1).reshape(-1,1)[surface_idx]
#         label_out = np.argmax(model_out, axis=-1)[surface_idx]
        
#         if use_post_processing:
#             from sklearn.neighbors import KDTree
#             valid_labels = np.max(model_out[surface_idx,:], axis=1)>1e-2
#             valid_branch_coord = space_3d[surface_idx][valid_labels]
#             valid_pred_part_labels = label_out[valid_labels]
#             kd_tree = KDTree(valid_branch_coord, leaf_size=8)
#             _, closest_idx = kd_tree.query(space_3d[surface_idx])
#             label_out = valid_pred_part_labels[np.reshape(closest_idx,[-1])]

#             label_out = label_out.reshape(-1,1)
#         # label_out = np.argmax(model_out, axis=-1).reshape(-1,1)
#         # ######################## airplane
#         group_map = {13:0, 14:0, 5:1, 4:1, 3:2, 8:2, 12:1, 9:1}
#         # {7:0, 15:0, 9:1, 8:1, 5:2, 14:2, 12:2, 10:1,3:1}#{2:0, 1:2, 0:2, 6:1, 5:1, 7:1}
#         # group_map = {2:0, 1:2, 0:2, 6:1, 5:1, 7:3}# best 8cuboid{14:0, 13:2, 7:2, 8:2, 4:1, 6:1, 11:3,}
#         for i in range(len(label_out)):
#             if group_map.__contains__(label_out[i,0]):
#                 label_out[i,0] = group_map[label_out[i,0]]
#             else:
#                 label_out[i,0] = 0
        
#         ######################## chair

#         # group_map = {15:0, 2:0, 13:0, 9:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best
#         # group_map = {15:0, 2:0, 13:0, 9:1, 1:1, 5:2, 12:2, 7:2, 10:2, 6:2, 0:3, 4:3} # best structure
#         # group_map = {1:0, 10:1, 14:2, 8:2, 3:2, 2:2, 0:2, 12:3, 6:3} # best 8_8
#         # group_map = {1:0, 11:0, 10:1, 12:2, 2:2, 5:2, 14:2} # best 16cylinder
#         # group_map = {8:0, 2:0, 4:1, 11:1, 14:2, 5:2, 3:2, 1:2, 13:3} # no_deformer, 16cuboid
#         # group_map = {12:0, 11:1, 5:2, 14:2, 7:2, 1:2} # no_deformer, 16cylinder
#         # for i in range(len(label_out)):
#         #     if group_map.__contains__(label_out[i,0]):
#         #         label_out[i,0] = group_map[label_out[i,0]]
#         #     else:
#         #         label_out[i,0] = 0

#         ######################## table
#         # gt_map = {
#         #     0:0,
#         #     1:1,
#         #     2:1,
#         # }
#         # # best
#         # group_map = {4:0, 8:0, 12:0}
#         # # best 16cylinder
#         # group_map = {12:0, 10:0, 8:0, 2:0}
#         # # best 16cuboid_nodeformer
#         # group_map =  {15:0, 1:0, 8:0, 13:0}
#         # # best 16cylinder_nodeformer
#         # group_map = {10:0, 9:0, 15:0, 8:0}
        
#         # for i in range(len(label_out)):
#         #     if group_map.__contains__(label_out[i,0]):
#         #         label_out[i,0] = group_map[label_out[i,0]]
#         #     else:
#         #         label_out[i,0] = 1
#         # for i in range(len(label_gt)):
#         #     label_gt[i,0] = gt_map[label_gt[i,0]]


#         ##################################
#         part_ious = [0.0] * len(labels_unique)
#         for i in range(len(labels_unique)):
#             if (np.sum(label_gt==i) == 0) and (np.sum(label_out==i) == 0): # part is not present, no prediction as well
#                 part_ious[i] = 1.0
#             else:
#                 part_ious[i] = np.sum(( label_gt==i ) & ( label_out==i )) / float(np.sum(   ( label_gt==i ) | ( label_out==i ) ))

#         shape_mIOU[data_idx] = np.mean(part_ious)
#         print(f'surface_idx: {surface_idx.sum(0)/b_point_num}, sample: {data_idx}, m-IOU: {np.mean(part_ious)}')
#         #output ply
#         if args.result_dir != '' and not os.path.isdir(args.result_dir):
#             os.mkdir(args.result_dir)

#         fout = open(args.result_dir+'/'+f"miou{np.round(np.mean(part_ious), 3):.3f}_sample_{data_idx}_shapenet_recon.ply", 'w')
#         fout.write("ply\n")
#         fout.write("format ascii 1.0\n")
#         fout.write("element vertex "+str(len(label_gt)*2)+"\n")
#         fout.write("property float x\n")
#         fout.write("property float y\n")
#         fout.write("property float z\n")
#         fout.write("property uchar red\n")
#         fout.write("property uchar green\n")
#         fout.write("property uchar blue\n")
#         fout.write("end_header\n")

#         vertices = space_3d[surface_idx].reshape(-1,3)
#         for i in range(len(label_gt)):
#             color = color_list[label_out[i,0]]
#             fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
#         vertices = space_3d[surface_idx].reshape(-1,3)
#         for i in range(len(label_gt)):
#             color = color_list[label_gt[i,0]]
#             fout.write(str(vertices[i,0]+0.8)+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
        
#         fout.close()
#     cate_mIOU = np.mean(shape_mIOU)*100.0
#     # grouping(args.result_dir)
#     print(cate_mIOU)

def load_pretrained_model(model, ckpt_path='', ckpt_key='model_state_dict'):
    pretrain = torch.load(ckpt_path)[ckpt_key]
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in pretrain.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255",
                  "180 180 180", "100 100 100",
                  "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255",
                  "180 100 255","100 180 255"] + ["240 0 0","0 240 0","0 0 240","240 240 0","240 0 240","0 240 240",
                  "200 200 200", "60 60 60",
                  "240 140 140","140 240 140","140 140 240","240 240 140","240 140 240","140 240 240",
                  "200 60 240","60 200 240"]

if __name__ == '__main__':
    import sys 
    sys.path.append("/home/qyshuai/code/paper/DPF-Net/torch_base/") 
    from data.data_entry import select_train_loader, select_eval_loader, select_test_loader
    from model.model_entry import select_model
    from options import prepare_train_args
    from utils.common import (
        load_match_dict,
    )
    import random
    import numpy as np
    import torch
    import trimesh
    from bae_utils import parse_txt_list
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    args = prepare_train_args()
    test_loader = select_test_loader(args)
    model = select_model(args)

    model = torch.nn.DataParallel(model)
    # copy model to GPU
    model = model.cuda()

    model = load_pretrained_model(model, args.load_model_path, 'state_dict')

    use_gt_mesh = True

    if 'airplane' in args.shapenet_path:
        category_info = '02691156_airplane'
    elif 'table' in args.shapenet_path:
        category_info = '04379243_table'
    elif 'chair' in args.shapenet_path:
        category_info = '03001627_chair'

    num_part = model.module.num_template
    
    category_id = category_info.split('_')[0]
    args.category_id = category_id
    args.category_info = category_info
    
    mode = 'reconstruction'
    # mode = 'test_metric'
    # mode = 'segmentation_ori'
    # mode = 'segmentation'
    # mode = 'segmentation_partnet'
    # mode = 'segmentation_partnet_ori'
    # mode = 'field'
    if mode == 'reconstruction':
        if args.stage2:
            args.result_dir = os.path.join(args.result_dir, 'eval_results')
            print(args.result_dir)
        else:
            args.result_dir = os.path.join(args.result_dir, 'eval_structure_results')
        save_recon(model, test_loader, args, part_thres=0.3, sample_num = np.inf, Resolution=64)
    elif mode == 'field':
        if args.stage2:
            args.result_dir = os.path.join(args.result_dir, 'vis_field')
        else:
            args.result_dir = os.path.join(args.result_dir, 'vis_structure_field')
        save_field(model, test_loader, args, part_thres=0.3, sample_num = np.inf, Resolution=32)

    elif mode == 'test_metric':
        if args.stage2:
            args.result_dir = os.path.join(args.result_dir, 'sample_point_results')
            test_metric(model, test_loader, args, part_thres=0.6, sample_num=np.inf, Resolution=64)
        else:
            args.result_dir = os.path.join(args.result_dir, 'structure_sample_point_results')
            test_metric(model, test_loader, args, part_thres=0.5, sample_num=np.inf, Resolution=64)
    elif 'segmentation' in mode:
        data_dir = '/gpfs/home/sist/cq14/kBAE/dataset/data/'+category_info+'/'
        ref_txt_name = '/gpfs/home/sist/cq14/kBAE/dataset/data/'+category_info+'/'+category_id+'_test_vox.txt'
        allset_txt_name = '/gpfs/home/sist/cq14/kBAE/dataset/data/'+category_info+'/'+category_id+'_test_vox.txt'

        ref_points, ref_values, ref_point_num, gf_split, idx, labels_unique, ref_names = parse_txt_list(ref_txt_name, data_dir+"/points", allset_txt_name)

        if mode == 'segmentation':
            args.result_dir = os.path.join(args.result_dir, 'segmentation_results')
            test_pointcloud(model, test_loader, args, part_thres=0.6, sample_num=np.inf, Resolution=32)
        elif mode == 'segmentation_ori':
            args.result_dir = os.path.join(args.result_dir, 'segmentation_results_ori')
            test_pointcloud(model, test_loader, args, part_thres=0.6, sample_num=np.inf, Resolution=32)
        if mode == 'segmentation_partnet':
            args.result_dir = os.path.join(args.result_dir, 'segmentation_results_partnet')
            test_partnet_pointcloud(model, test_loader, args, part_thres=0.6, sample_num=np.inf, Resolution=32)
        elif mode == 'segmentation_partnet_ori':
            args.result_dir = os.path.join(args.result_dir, 'segmentation_results_partnet_ori')
            test_partnet_pointcloud(model, test_loader, args, part_thres=0.6, sample_num=np.inf, Resolution=32)
        
