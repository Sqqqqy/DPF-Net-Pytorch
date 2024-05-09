import csv
import time

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast, GradScaler
import numpy as np
import math

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from loss import all_loss

from utils.common import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    load_match_dict,
)
from utils.visualize import (
    save_recon,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
seed = 31
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        fix_random_seeds(args.seed)
        self.args = args
        
        torch.cuda.set_device(int(int(args.local_rank)))
        dist.init_process_group(backend='nccl')
        fix_random_seeds(args.seed)

        logger, training_stats= initialize_exp(args, "epoch", "loss")
        self.logger = logger
        self.stats = training_stats

        self.train_loader = select_train_loader(args)
        if not self.args.disable_val:
            self.val_loader = select_eval_loader(args)

        self.model = select_model(args)
        # copy model to GPU
        self.model = self.model.cuda()
        
        if self.args.use_apex:
            self.scaler = GradScaler()
        
        # load a pre-trained weights
        if args.load_model_path != '' and dist.get_rank() == 0:
            print("=> using pre-trained weights")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path, 'state_dict')
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())
        
        # self.model = DDP(self.model, device_ids=[int(args.local_rank)], output_device=int(args.local_rank), find_unused_parameters=True)
        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = torch.optim.Adam(self.model.module.parameters(),
                                          self.args.base_lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)
        
        # lr schedule generation
        warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(self.train_loader) * args.warmup_epochs)
        iters = np.arange(len(self.train_loader) * (args.epochs - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                             math.cos(math.pi * t / (len(self.train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        if dist.get_rank() == 0:
            self.logger.logger.info("Building optimizer done.")
        
        # resume from a checkpoint if there exists
        if args.enable_load_checkpoint:
            to_restore = {"epoch": 0}
            restart_from_checkpoint(
                args, 
                run_variables=to_restore,
                logger=self.logger,
                state_dict=self.model,
                # optimizer=self.optimizer,
                # amp=apex.amp,
            )
            self.args.start_epoch = to_restore["epoch"]

    def train(self):
        if dist.get_rank() == 0:
            self.logger.logger.info("Start Training")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            # if not self.args.disable_val and epoch % self.args.checkpoint_freq == 0 or epoch == self.args.epochs-1:
            #     self.val_per_epoch(epoch)
            if dist.get_rank() == 0:
                self.logger.save_curves(epoch)
                # self.logger.update_checkpoint(self.model, self.optimizer, epoch)
                if (epoch % self.args.checkpoint_freq == 0 or epoch == self.args.epochs-1):
                    if not self.args.SVR:
                        save_recon(self.model, self.val_loader, self.args)
                    self.logger.save_checkpoint(self.model, self.optimizer, epoch)
            # torch.distributed.barrier()

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()

        self.train_loader.sampler.set_epoch(epoch)
        
        for i, data in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            # update learning rate
            iteration = epoch * len(self.train_loader) + i
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[iteration]
                
            if self.args.use_apex:
                with autocast():
                    # forward pass
                    pred_dict, gt_dict = self.step(data)
                    # compute loss
                    loss_dict = self.compute_loss(pred_dict, gt_dict, epoch)
                    # loss_dict['l1_reg'] = l1_regularization(self.model)

                    loss_dict_scalar = {}
                    loss = 0
                    for key in loss_dict.keys():
                        self.logger.record_scalar(key, loss_dict[key].item())
                        loss_dict_scalar[key] = loss_dict[key].item()
                        loss = loss + loss_dict[key]
                    
            else:
                # forward pass
                pred_dict, gt_dict = self.step(data)

                # compute loss
                loss_dict = self.compute_loss(pred_dict, gt_dict)
                loss = 0
                for key in loss_dict.keys():
                    self.logger.record_scalar(key, loss_dict[key].item())
                    loss += loss_dict[key]
            
            
            # compute gradient and optim step
            self.optimizer.zero_grad()
            if self.args.use_apex:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()


            # logger record
            losses.update(loss.item(), data[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # compute metrics
            # metrics = self.compute_metrics(pred, label, is_train=True)
            # for key in metrics.keys():
            #     self.logger.record_scalar(key, metrics[key])
            if dist.get_rank() == 0:        
                self.logger.record_scalar('loss', loss.item())


            # monitor training progress
            if i % self.args.log_freq == 0 and dist.get_rank() == 0:

                self.logger.logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        i,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                )
                self.logger.logger.info(
                    loss_dict_scalar
                )

    def val_per_epoch(self, epoch):

        # switch to val mode
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()

        self.val_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            if self.args.use_apex:
                with autocast():
                    # forward pass
                    pred_dict, gt_dict = self.step(data)
                    # compute loss
                    loss_dict = self.compute_loss(pred_dict, gt_dict, epoch)
                    # loss_dict['l1_reg'] = l1_regularization(self.model)
                    
                    if dist.get_rank() == 0:
                        print(loss_dict)

                    loss = 0
                    for key in loss_dict.keys():
                        self.logger.record_scalar(key, loss_dict[key].item())
                        loss = loss + loss_dict[key]


            # logger record
            losses.update(loss.item(), data[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # compute metrics
            metrics = self.compute_metrics(pred_dict, gt_dict)
            
            if dist.get_rank() == 0:
                print(metrics)
                self.logger.record_scalar('loss', loss.item())

                for key in metrics.keys():
                    self.logger.record_scalar(key, metrics[key])
                    

            # monitor training progress
            if i % self.args.log_freq == 0 and dist.get_rank() == 0:
                self.logger.logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        i,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                )

    def step(self, data):
        if self.args.SVR:
            if self.args.SVR_z:
                data_voxels, data_imgs, data_zs, data_points = data
                data_zs = Variable(data_zs).cuda().float()
            else:
                data_voxels, data_imgs, data_points = data
            
            data_voxels = Variable(data_voxels).cuda().float()
            data_imgs = Variable(data_imgs).cuda().float()
            data_points = Variable(data_points).cuda().float()

            gt_dict = gt_dict = {
                'data_voxels': data_voxels,
                'data_points': data_points,
                'data_imgs': data_imgs}
            if self.args.SVR_z:
                gt_dict['data_zs'] = data_zs
        else:
            data_voxels, data_points, data_fps_points, data_values = data
            # warp input
            data_voxels = Variable(data_voxels).cuda().float()
            data_fps_points = Variable(data_fps_points).cuda().float()
            data_points = Variable(data_points).cuda().float()
            data_values = Variable(data_values).cuda().float()
            data_points_extend = torch.cat([data_points, data_fps_points], dim=1)
            gt_dict = {
                'data_voxels': data_voxels,
                'data_fps_points': data_fps_points,
                'data_points': data_points,
                'data_values': data_values,
            }

        # compute output
        if self.args.SVR:
            if self.args.SVR_z:
                pred_dict = self.model(data_voxels, data_imgs, data_zs, data_points)
            else:
                pred_dict = self.model(data_voxels, data_imgs, data_points)
        else:
            pred_dict = self.model(data_voxels, data_points_extend, is_training=True)
        return pred_dict, gt_dict

    def compute_metrics(self, pred, gt):
        
        # chamfer loss-L1
        # iou
        # per-iou

        metrics = {
            'xxx': 0,
        }
        return metrics

    def compute_loss(self, pred_dict, gt_dict, epoch):
        # unzip the predict and gt dictionary
        return all_loss(self.args, pred_dict, gt_dict, epoch)

        
def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()