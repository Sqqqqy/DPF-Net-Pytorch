import argparse
import os

from utils.common import (
    bool_flag,
)

def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='bae', help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='shapenet', help='used in data_entry.py')
    parser.add_argument('--dump_path', type=str, default='./', help='path to save log and models')
    parser.add_argument('--ckp_path', type=str, default='', help='path for loading pretrained checkpoint')
    parser.add_argument('--save_prefix', type=str, default='', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='model path for pretrain or test')
    parser.add_argument('--load_svr_path', type=str, default='',
                        help='model path for svr')
    parser.add_argument('--enable_load_checkpoint', action='store_true',
                        help='enable_load_checkpoint')
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--disable_val', action='store_true',
                        help='Disable val')
    parser.add_argument('--log_freq', type=int, default=50,
                        help='Print the info periodically')
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--local_rank", default=-1)
    return parser
    
def parse_train_args(parser):
    parser = parse_common_args(parser)
    
    ##########################
    #### data parameters  ####
    ##########################
    parser.add_argument('--shapenet_path', type=str, default=r'/data/Sqqqqy/BSP-Dataset/03001627_vox.hdf5')
    parser.add_argument('--shapenet_svr_z_path', type=str, default=r'/home/sist/cq14/kBAE/code/RIM-Net-main/z_out.npy')
    parser.add_argument('--shapenet_svr_path', type=str, default=r'/home/sist/cq14/kBAE/dataset/shapenet/02691156_airplane/02691156_train_vox.hdf5')
    parser.add_argument('--shapenet_fps_path', type=str, default=r'/mnt/e/dataset/shapenet/shapenet_fps_1024_03001627_chair_train.npy')
    
    ##########################
    ## model specific params #
    ##########################
    parser.add_argument('--loss', default='ce',
                        help='different loss', choices=['l1', 'ce', 'mse', 'cd'])
    parser.add_argument('--REAL_SIZE', type=int, default=32,
                        help='the size sample grid')
    parser.add_argument('--enable_recon_loss', action='store_true',
                        help='using self reconstruction loss for training')
    
    parser.add_argument('--stage2', action='store_true',
                        help='training stage 2')
    
    parser.add_argument('--SVR', action='store_true',
                        help='training SVR')
    parser.add_argument('--SVR_z', action='store_true',
                        help='training SVR')
                        
    parser.add_argument('--num_part', type=int, default=16,
                    help='the number of part')
    
    parser.add_argument('--primitive_type', type=str, default='cuboid',
                    help='the shape prior type of part')

    parser.add_argument('--use_apex', action='store_true',
                        help='using apex for training')
    
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='If you cannot understand this param, leave blank, auto generated.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0, help='final learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    

    #########################
    #### other parameters ###
    #########################
    parser.add_argument('--workers', default=12, type=int, metavar='N', 
                        help='number of data loading workers (default: 32)')
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated, experiment dump path for checkpoints and log')
    parser.add_argument('--distributed', type=bool_flag, default=False,
                        help="whether to train with distributed mode or not")
    parser.add_argument('--dump_checkpoints', type=str, default='',
                        help="leave blank, auto generated, path to save checkpoints")
    parser.add_argument("--tensorboard_path", type=str, default='/output/logs',
                        help='path to save tensorboard log')
    
    
    return parser
    
    
def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser

def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args
    
def get_train_model_dir(args):
    model_dir = os.path.join(args.dump_path, args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir
    args.dump_checkpoints = os.path.join(args.model_dir, "checkpoints")
    
    if not os.path.isdir(args.dump_checkpoints):
        try:
            os.mkdir(args.dump_checkpoints)
        except OSError:
            pass    
    
    if not os.path.isdir(args.tensorboard_path):
        try:
            os.mkdir(args.tensorboard_path)
        except OSError:
            pass    

def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_info = args.load_model_path.replace(ext, '')[:-1]
    # val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    val_info = 'visualize_results'
    result_dir = os.path.join(args.model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir
    
def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
