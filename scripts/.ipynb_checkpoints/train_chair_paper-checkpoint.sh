CUDA_VISIBLE_DEVICES=2,3,5,6 python -m torch.distributed.launch --nproc_per_node 4 --master_port 20421 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
    --epochs 1510 \
    --batch_size 128 \
    --log_freq 100 \
    --warmup_epochs 10 \
    --REAL_SIZE 32 \
    --base_lr 1e-4 --save_prefix 30_50_4cuboid_paper_SDF --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field_final \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/ --use_apex \
    --shapenet_fps_path /gpfs/home/sist/cq14/kBAE/dataset/shapenet/03001627_chair/shapenet_fps_1024_03001627_chair_train.npy \
    --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_chair/03001627_train_vox.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict \
    --num_part 4 \
    --primitive_type cuboid \
    --stage2 \
    # --load_model_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1229_stage2/deformed_primitive_field_test/ckp-300.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 20422 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 1010 \
#     --batch_size 32 \
#     --log_freq 100 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 64 \
#     --base_lr 1e-4 --save_prefix 10_30_50_16cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/ --use_apex \
#     --shapenet_fps_path /gpfs/home/sist/cq14/kBAE/dataset/shapenet/03001627_chair/shapenet_fps_1024_03001627_chair_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/03001627_chair/03001627_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --num_part 16 \s
#     --primitive_type cuboid \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/deformed_primitive_field_30_50_16cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-1500.pth \
#     --stage2 
#     # --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/deformed_primitive_field_32part_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-1500.pth