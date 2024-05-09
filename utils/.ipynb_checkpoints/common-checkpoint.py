import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

from utils.logger import Logger, PD_Stats

import torch.distributed as dist

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    - create a panda object to keep track of the training statistics
    """
    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.model_dir, "params.pkl"), "wb"))

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.model_dir, "stats" + ".pkl"), args
    )

    # create a logger
    logger = Logger(params)
    logger.logger.info("============ Initialized logger ============")
    logger.logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.logger.info("The experiment will be stored in %s\n" % params.model_dir)
    logger.logger.info("")
    return logger, training_stats


def restart_from_checkpoint(args, *, run_variables=None, logger=None, **kwargs):
    """
    Re-start from checkpoint
    """ 
    assert isinstance(logger, Logger)
    ckp_paths = []
    ckp_paths.append(args.ckp_path)
    ckp_paths.append(os.path.join(args.dump_checkpoints, 'checkpoint.pth.tar'))
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    if args.distributed:
        checkpoint = torch.load(
            ckp_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
        )
    else:
        checkpoint = torch.load(
            ckp_path, map_location="cuda:0"
        )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def load_match_dict(model, model_path, model_key=None):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    if model_key is None:
        pretrain_dict = torch.load(model_path)
    else:
        pretrain_dict = torch.load(model_path)[model_key]
    
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('module.', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    msg = model.load_state_dict(model_dict)
    print(msg)

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
