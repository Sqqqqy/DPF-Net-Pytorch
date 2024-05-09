# CUDA_VISIBLE_DEVICES=4,5,6,7 
# python -m torch.distributed.launch --nproc_per_node 8  --master_port 20422 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 5010 \
#     --batch_size 4 \
#     --log_freq 100 \
#     --warmup_epochs 10 \
#     --REAL_SIZE 64 \
#     --base_lr 1e-4 --save_prefix 30_50_16cuboid_SVR_bs4 --loss mse --momentum 0.5 --checkpoint_freq 200 \
#     --model_type dpf_svr \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/ --use_apex \
#     --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --num_part 16 \
#     --primitive_type cuboid \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/deformed_primitive_field_30_50_16cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
#     --stage2 \
#     --SVR

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node 4  --master_port 20423 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
    --epochs 2010 \
    --batch_size 32 \
    --log_freq 100 \
    --warmup_epochs 10 \
    --REAL_SIZE 64 \
    --base_lr 1e-4 --save_prefix RIM_airplane --loss mse --momentum 0.5 --checkpoint_freq 200 \
    --model_type dpf_svr \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/ --use_apex \
    --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict \
    --num_part 8 \
    --primitive_type cuboid \
    --stage2 \
    --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/deformed_primitive_field_50_0.1_8cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
    --SVR \
    --SVR_z
    
#50_0.1_8cuboid_SVR

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4  --master_port 20424 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 2010 \
#     --batch_size 32 \
#     --log_freq 100 \
#     --warmup_epochs 10 \
#     --REAL_SIZE 64 \
#     --base_lr 1e-4 --save_prefix 10_30_100_8cuboid_SVR --loss mse --momentum 0.5 --checkpoint_freq 200 \
#     --model_type dpf_svr \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/ --use_apex \
#     --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --num_part 8 \
#     --primitive_type cuboid \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/deformed_primitive_field_10_30_100_8cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
#     --stage2 \
#     --SVR