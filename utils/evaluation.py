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

def rotMat(theta):
    return [[np.cos(theta), 0, np.sin(theta)],[-np.sin(theta), 0, np.cos(theta)],[0,1,0],]


def distance_p2p(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    
    B, N2, _ = points_tgt.shape #[bs, N, 3]
#     points = points_tgt[:, :, :3]
    # Extract the predicted surface.
    _, N1, _ = points_src.shape #[bs, N, 3]
    
    points_tgt = torch.Tensor(points_tgt).cuda()
    points_src = torch.Tensor(points_src).cuda()
    
    
    diff = points_src[:, :, None] - points_tgt[:,None] #[bs, N1, 1, 3] - [bs, 1, N2, 3] = [bs, N1, N2, 3]
    
    dists_L2 = torch.square(diff).sum(-1)

    dists_L1 = torch.abs(diff).sum(-1)
    
    assert dists_L1.shape == (B, N1, N2)
    target_to_prim_L2 = dists_L2.min(1)[0].mean().item()
    prim_to_target_L2 = dists_L2.min(2)[0].mean().item()
    
    target_to_prim_L1 = dists_L1.min(1)[0].mean().item()
    prim_to_target_L1 = dists_L1.min(2)[0].mean().item()
    
    
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
        # Return maximum losses if pointcloud is empty


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
        # print('chamferL1:',chamferL1,' chamferL2:',chamferL2)
        return chamferL1, chamferL2

def batched_test_recon(model, data_loader, args, sample_num=10, part_thres=0.4, Resolution=64):
    model.eval()
    color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255",
                  "180 180 180", "100 100 100",
                  "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255",
                  "180 100 255","100 180 255"] + ["240 0 0","0 240 0","0 0 240","240 240 0","240 0 240","0 240 240",
                  "200 200 200", "60 60 60",
                  "240 140 140","140 240 140","140 140 240","240 240 140","240 140 240","140 240 240",
                  "200 60 240","60 200 240"]
    
    cd_L1 = []
    cd_L2 = []
    space_3d = np.array([[x,y,z] for x in np.arange(-0.5,0.5,1/Resolution) for y in np.arange(-0.5,0.5,1/Resolution) for z in np.arange(-0.5,0.5,1/Resolution)])
    x_cords, y_cords, z_cords = space_3d.transpose()
    space_3d = torch.Tensor(space_3d.copy().reshape((1,-1,3))).cuda()

    for data_idx, data in enumerate(data_loader):

        test_voxels, _, _, _ = data
        test_voxels = Variable(test_voxels).cuda().float()
        test_voxels = test_voxels.reshape((1,1,64,64,64))
        model_out = model(test_voxels, space_3d)['G_'].cpu().detach().numpy()
        
        
        thres = [part_thres for i in range(num_part)]
        vertices_num = 0
        triangles_num = 0
        vertices_num_list = [0]

        vertices_t = []
        triangles_t = []
        for part_i in range(num_part):
            voxel = np.zeros((Resolution, Resolution, Resolution))
            for i, (x, y, z) in enumerate(zip(x_cords, y_cords, z_cords)):
                x,y,z = map(lambda y:int((y+0.5)*Resolution), [x,y,z])
                voxel[x][y][z] = model_out[0,i,part_i]
            vertices, triangles = mcubes.marching_cubes(voxel, thres[0])


            vertices_t.append(vertices)
            triangles_t.append(triangles+vertices_num)
            vertices_num_list.append(vertices_num)
            vertices_num += len(vertices)
            triangles_num += len(triangles)
        
        vertices_t = np.concatenate(vertices_t)
        triangles_t = np.concatenate(triangles_t)
        
        vertices_t = (vertices_t/Resolution-0.5)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_t)
        mesh.triangles = o3d.utility.Vector3iVector(triangles_t)
        
        ps = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=10000).points
        ps = np.array(ps).astype(np.float32).reshape(1, -1, 3)
        ps[:,:,0],ps[:,:,2] = -ps[:,:,0],-ps[:,:,2]
        tmp = ps[:,:,1].copy()
        ps[:,:,1] = -ps[:,:,2]
        ps[:,:,2] = tmp
        
        
        vertices_gt, triangles_gt = mcubes.marching_cubes(test_voxels[t].reshape((64,64,64)), thres[0])

        vertices_gt = (vertices_gt/64-0.5)
        
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_gt)
        mesh.triangles = o3d.utility.Vector3iVector(triangles_gt)
        
        ps2 = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=4096).points
        ps2 = np.array(ps2).astype(np.float32).reshape(1, -1, 3)
        
        cd_l1, cd_l2 = eval_pointcloud(ps, ps2)
        cd_L1.append(cd_l1)
        cd_L2.append(cd_l2)
        IOU.append(volumetric_iou)
#         output ply
        fout = open(args.result_dir+'/'+f"sample_{data_idx}_shapenet_recon.ply", 'w')
        
            
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        
        fout.write("element vertex "+str(ps2.shape[1]+ps.shape[1])+"\n")
        
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("end_header\n")

        shapenet_points_gt = np.array(ps2.reshape((-1,3)))
        for i in range(len(shapenet_points_gt)):
            color = color_list[0]
            fout.write(str(float(shapenet_points_gt[i][0]))+" "+str(float(shapenet_points_gt[i][1]))+" "+str(float(shapenet_points_gt[i][2]))+" "+color+"\n")
                
        shapenet_points = np.array(ps.reshape((-1,3)))
        for i in range(len(shapenet_points)):
            fout.write(str(float(shapenet_points[i][0]))+" "+str(float(shapenet_points[i][1]))+" "+str(float(shapenet_points[i][2]))+" 0 0 0"+"\n")

        
    return cd_L1, cd_L2