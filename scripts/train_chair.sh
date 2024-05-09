python -m torch.distributed.launch --nproc_per_node 1 /home/qyshuai/code/paper/DPF-Net/torch_base/train.py \
    --epochs 510 \
    --batch_size 128 \
    --log_freq 100 \
    --warmup_epochs 10 \
    --REAL_SIZE 32 \
    --base_lr 1e-4 --save_prefix 30_50_16cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field \
    --dump_path /home/qyshuai/code/paper/DPF-Net/output/chair/ --use_apex \
    --shapenet_fps_path /mnt/e/dataset/shapenet/shapenet_fps_1024_03001627_chair_train.npy \
    --shapenet_path /home/qyshuai/code/paper/data/03001627_chair/03001627_train_vox.hdf5 \
    --tensorboard_path /home/qyshuai/code/paper/DPF-Net/output/chair/logs \
    --load_not_strict \
    --num_part 16 \
    --primitive_type cuboid \
    --stage2 


python -m torch.distributed.launch --nproc_per_node 1 /home/qyshuai/code/paper/DPF-Net/torch_base/train.py \
    --epochs 510 \
    --batch_size 128 \
    --log_freq 100 \
    --warmup_epochs 10 \
    --REAL_SIZE 64 \
    --base_lr 1e-4 --save_prefix 64_30_50_16cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field \
    --dump_path /home/qyshuai/code/paper/DPF-Net/output/chair/ --use_apex \
    --shapenet_fps_path /mnt/e/dataset/shapenet/shapenet_fps_1024_03001627_chair_train.npy \
    --shapenet_path /home/qyshuai/code/paper/data/03001627_chair/03001627_train_vox.hdf5 \
    --tensorboard_path /home/qyshuai/code/paper/DPF-Net/output/chair/logs \
    --load_not_strict \
    --num_part 16 \
    --primitive_type cuboid \
    --stage2 \
    --load_model_path /home/qyshuai/code/paper/DPF-Net/output/chair/deformed_primitive_field_30_50_16cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth