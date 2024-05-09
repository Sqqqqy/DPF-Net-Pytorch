CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20421 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
    --epochs 1010 \
    --batch_size 128 \
    --log_freq 100 \
    --warmup_epochs 10 \
    --REAL_SIZE 32 \
    --base_lr 1e-4 --save_prefix 100_1_10cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/ --use_apex \
    --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/02691156_airplane/shapenet_fps_1024_02691156_airplane_train.npy \
    --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/02691156_airplane/02691156_train_vox.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict  \
    --num_part 10 \
    --primitive_type cuboid \
    --stage2 
    # --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/deformed_primitive_field_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/ckp-400.pth
    
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 20426 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 1010 \
#     --batch_size 32 \
#     --log_freq 50 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 64 \
#     --base_lr 1e-4 --save_prefix 64_10_5_10cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/02691156_airplane/shapenet_fps_1024_02691156_airplane_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/02691156_airplane/02691156_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict  \
#     --stage2 \
#     --primitive_type cuboid \
#     --num_part 10 \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/deformed_primitive_field_10_5_12cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-1000.pth
    # /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/deformed_primitive_field_8cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-800.pth
