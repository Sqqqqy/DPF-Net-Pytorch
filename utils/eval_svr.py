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

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[-np.sin(theta), 0, np.cos(theta)],[0,1,0],]


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

def test_svr_metric(model, data_loader, args, sample_num=20, part_thres=0.4, Resolution=32):

    chamfer_metric = []
    model.eval()

    eval_ids = open('/gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_test.txt',"r").read().splitlines()

    if 'airplane' in category_info:
        start, end = 0, 809
    elif 'table' in category_info:
        start, end = 6461, 8163
    elif 'chair' in category_info:
        start, end = 2988, 4344

    eval_ids = eval_ids[start:end]

    num_part = model.module.num_template
    space_3d = np.array([[x,y,z] for x in np.arange(-0.5,0.5,1/Resolution) for y in np.arange(-0.5,0.5,1/Resolution) for z in np.arange(-0.5,0.5,1/Resolution)])
    space_3d += np.array([1/Resolution/2, 1/Resolution/2, 1/Resolution/2])
    x_cords, y_cords, z_cords = space_3d.transpose()
    space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()
    
    for data_idx, data in tqdm(enumerate(data_loader)):
        if data_idx > sample_num:
            break
        test_voxels, test_imgs, _ = data
        test_voxels_tensor = Variable(test_voxels).cuda().float().reshape((1,1,64,64,64))
        test_imgs_tensor = Variable(test_imgs).cuda().float()
        model_out = model(test_voxels_tensor, test_imgs_tensor, space_3d)['G'].cpu().detach().numpy()
        
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
            gt_mesh = trimesh.load_mesh('/gpfs/home/sist/cq14/kBAE/dataset/shapenet_gt/'+eval_ids[data_idx]+'/model.obj')
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
        
        chamferL1, chamferL2 = eval_pointcloud(pred_ps, gt_ps)
        chamfer_metric.append([chamferL1, chamferL2])

        if args.result_dir != '' and not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)

        fout = open(args.result_dir+'/'+f"sample_{data_idx}_shapenet_recon.ply", 'w')
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
    sys.path.append("/home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base") 
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
    # import open3d as o3d
    
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

    model = load_pretrained_model(model, args.load_svr_path, 'state_dict')
    model = load_pretrained_model(model, args.load_model_path, 'state_dict')

    if 'airplane' in args.dump_path:
        category_info = '02691156_airplane'
    elif 'table' in args.dump_path:
        category_info = '04379243_table'
    elif 'chair' in args.dump_path:
        category_info = '03001627_chair'

    num_part = model.module.num_template
    
    category_id = category_info.split('_')[0]
    args.category_id = category_id
    args.category_info = category_info

    use_gt_mesh = True
    
    args.result_dir = os.path.join(args.result_dir, 'sample_point_results')
    test_svr_metric(model, test_loader, args, part_thres=0.7, sample_num=np.inf, Resolution=64)
# 8273