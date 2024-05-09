from torch.utils.data import Dataset
import numpy as np
import h5py

class ShapeNetDataset(Dataset):
    def __init__(self, args, dataset_type='train'):
        # usually we need args rather than single datalist to init the dataset
        super().__init__()

        self.args = args
        if args.SVR:
            data_dict = h5py.File(args.shapenet_svr_path, 'r')
            data_points_int = data_dict['points_'+str(args.REAL_SIZE)][:]
            self.data_points = (data_points_int+0.5)/args.REAL_SIZE-0.5

            if dataset_type == 'train':
                data_path = args.shapenet_path
            elif dataset_type == 'val':
                data_path = args.shapenet_path.replace('train', 'test')
            elif dataset_type == 'test':
                data_path = args.shapenet_path.replace('train', 'test')

            data_dict = h5py.File(data_path, 'r')
            self.view_size = 137
            self.crop_size = 128
            self.view_num = 24
            self.crop_edge = self.view_size-self.crop_size
            offset_x = int(self.crop_edge/2)
            offset_y = int(self.crop_edge/2)
            if dataset_type == 'train':
                if 'airplane' in args.load_model_path:
                    start, end = 0, 3236
                elif 'table' in args.load_model_path:
                    start, end = 25820, 32627
                elif 'chair' in args.load_model_path:
                    start, end = 11940, 17362
            elif dataset_type == 'test':
                if 'airplane' in args.load_model_path:
                    start, end = 0, 809
                elif 'table' in args.load_model_path:
                    start, end = 6461, 8163
                elif 'chair' in args.load_model_path:
                    start, end = 2988, 4344
            else:
                start, end = 0, 0
            self.data_imgs = np.reshape(data_dict['pixels'][start:end,:,offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size], [-1,self.view_num,1,self.crop_size,self.crop_size])
            self.data_voxels = data_dict['voxels'][start:end]
            if self.args.SVR_z:
                 self.z_list = np.load(args.shapenet_svr_z_path)
        else:
            if dataset_type == 'train':
                data_path = args.shapenet_path
            elif dataset_type == 'val':
                data_path = args.shapenet_path.replace('train', 'val')
            elif dataset_type == 'test':
                data_path = args.shapenet_path.replace('train', 'test')

            data_dict = h5py.File(data_path, 'r')
            data_points_int = data_dict['points_'+str(args.REAL_SIZE)][:]
            
            self.data_voxels = data_dict['voxels'][:]
            self.data_points = (data_points_int+0.5)/args.REAL_SIZE-0.5
            self.data_values = data_dict['values_'+str(args.REAL_SIZE)][:]
            self.data_fps_points = np.load(args.shapenet_fps_path)

        
    def __len__(self):
        if self.args.SVR:
            return self.data_voxels.shape[0]
        else:
            return self.data_points.shape[0]

    def __getitem__(self, idx):
        if self.args.SVR:
            data_voxels = self.data_voxels[idx].reshape([-1,64,64,64])
            
            if self.args.load_svr_path != '':
                which_view = 23
                data_points = self.data_points[idx]
            else:
                which_view = np.random.randint(self.view_num)
                data_points = self.data_points[0]
            data_imgs = self.data_imgs[idx,which_view].astype(np.float32)/255.0
            if self.args.SVR_z:
                data_zs = self.z_list[idx]
                return data_voxels, data_imgs, data_zs, data_points
            else:
                return data_voxels, data_imgs, data_points
            

        else:
            data_voxels = self.data_voxels[idx].reshape([-1,64,64,64])
            data_points = self.data_points[idx]
            data_fps_points = self.data_fps_points[idx]
            data_values = self.data_values[idx]
            return data_voxels, data_points, data_fps_points, data_values