# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20221 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 3010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 32 \
#     --base_lr 1e-4 --save_prefix test --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1229_no_wloss_stage1/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1229_wloss_stage1/deformed_primitive_field_test/ckp-300.pth



# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20221 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 1010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 10 \
#     --REAL_SIZE 32 \
#     --base_lr 1e-4 --save_prefix test --loss mse --momentum 0.5 --checkpoint_freq 20 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1219_stage1/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict 

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20221 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 3010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 32 \
#     --base_lr 1e-4 --save_prefix test --loss mse --momentum 0.5 --checkpoint_freq 20 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1227_stage2/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict 
#     # --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1219_stage2/ckp-100.pth 
#     # >> deformed_primitive_field_1219_stage2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20422 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 3010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 32 \
#     --base_lr 1e-4 --save_prefix test --loss mse --momentum 0.5 --checkpoint_freq 20 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1219_stage2/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1219_stage2/ckp-100.pth
# exit
# --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1123/deformed_primitive_field_e-kx_wloss_2type_test/ckp-700.pth \


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20422 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
    --epochs 3010 \
    --batch_size 128 \
    --log_freq 10 \
    --warmup_epochs 10 \
    --REAL_SIZE 32 \
    --base_lr 1e-4 --save_prefix test --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1229_stage2/ --use_apex \
    --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
    --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_train_vox.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict \
    --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1229_stage2/deformed_primitive_field_test/ckp-300.pth
    # --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1121/deformed_primitive_field_ori_w_1type_test/ckp-700.pth
    #--load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1123/deformed_primitive_field_e-kx_wloss_2type_test/ckp-700.pth