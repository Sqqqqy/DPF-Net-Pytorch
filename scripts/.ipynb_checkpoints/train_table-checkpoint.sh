# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 20426 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
#     --epochs 1010 \
#     --batch_size 128 \
#     --log_freq 100 \
#     --warmup_epochs 10 \
#     --REAL_SIZE 32 \
#     --base_lr 1e-4 --save_prefix 100_50_8cuboid_coarse_exist_recon_1_bs_128_fps_compact_0.0001_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type deformed_primitive_field \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/ --use_apex \
#     --shapenet_fps_path /gpfs/home/sist/cq14/kBAE/dataset/shapenet/04379243_table/shapenet_fps_1024_04379243_table_train.npy \
#     --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/04379243_table/04379243_train_vox.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --num_part 8 \
#     --stage2 \
#     --primitive_type cuboid
    

    # nohup 
    #  >> deformed_primitive_field_1121.log 2>&1 &
# source activate pt1.9

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node 4 --master_port 20421 /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/train.py \
    --epochs 1010 \
    --batch_size 32 \
    --log_freq 100 \
    --warmup_epochs 0 \
    --REAL_SIZE 64 \
    --base_lr 1e-4 --save_prefix stage2_16cylinder_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type deformed_primitive_field_best \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/ --use_apex \
    --shapenet_fps_path /gpfs/home/sist/cq14/kBAE/dataset/shapenet/04379243_table/shapenet_fps_1024_04379243_table_train.npy \
    --shapenet_path /home/sist/cq14/kBAE/dataset/shapenet/04379243_table/04379243_train_vox.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict \
    --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/deformed_primitive_field_16cylinder_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_tau4_deform4input/checkpoints/ckp-1000.pth \
    --num_part 16 \
    --primitive_type cylinder \
    --stage2

    # >> deformed_primitive_field_1219_stage2.log 2>&1 &

# sh torch_base/scripts/train_airplane.sh
# exit
# --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/deformed_primitive_field_1123/deformed_primitive_field_e-kx_wloss_2type_test/ckp-700.pth \
    