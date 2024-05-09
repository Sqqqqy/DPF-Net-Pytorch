from torch.utils.data import Dataset
import numpy as np

class ShapeNetDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super().__init__()
        if is_train:
            data_path = args.scannet_path
        else:
            data_path = args.val_list
        data_dict = h5py.File(data_path, 'r')
        data_points_int = data_dict['points_'+str(REAL_SIZE)][:]
        self.data_points = (data_points_int+0.5)/REAL_SIZE-0.5
        self.data_values = data_dict['values_'+str(REAL_SIZE)][:]
        self.data_fps_points = np.load(args.scannet_fps_path)
        
    def __len__(self):
        return self.data_points.shape[0]

    def __getitem__(self, idx):
        data_points = self.data_points[idx]
        data_fps_points = self.data_fps_points[idx]
        data_values = self.data_values[idx]
        return data_points, data_fps_points, data_values