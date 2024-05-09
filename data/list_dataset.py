from torch.utils.data import Dataset
import cv2
import numpy as np

from utils.common import (
    bool_flag,
)

class ListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super().__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        infos = [line.split() for line in open(data_list).readlines()]
        self.img_paths = [info[0] for info in infos]
        self.label_paths = [info[1] for info in infos]

    def preprocess(self, img, label):
        # cv: h, w, c, tensor: c, h, w
        img = img.transpose((2, 0, 1)).astype(np.float32)
        # you can add other process method or augment here
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, (128, 128))
        # label = cv2.imread(label_path)
        label = int(bool_flag(self.label_paths[idx]))
        img, label = self.preprocess(img, label)
        return img, label