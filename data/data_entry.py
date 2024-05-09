from data.list_dataset import ListDataset
from data.mem_list_dataset import MemListDataset
from torch.utils.data import DataLoader
from data.shapenet import ShapeNetDataset
import torch

def get_dataset_by_type(args, dataset_type='train'):
    type2data = {
        # 'list': ListDataset(args, is_train),
        # 'mem_list': MemListDataset(args, is_train),
        'shapenet': ShapeNetDataset(args, dataset_type),
    }
    dataset = type2data[args.data_type]
    return dataset


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, dataset_type='train')
    print('{} samples found in train'.format(len(train_dataset)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=4, pin_memory=False, drop_last=False, sampler=train_sampler)
    return train_loader


def select_eval_loader(args):
    eval_dataset = get_dataset_by_type(args, dataset_type='val')
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)
    return val_loader

def select_test_loader(args):
    test_dataset = get_dataset_by_type(args, dataset_type='test')
    print('{} samples found in val'.format(len(test_dataset)))
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    return test_loader