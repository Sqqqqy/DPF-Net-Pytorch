CUDA_VISIBLE_DEVICES=5 python /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/utils/eval_svr.py \
    --epochs 5010 \
    --batch_size 32 \
    --log_freq 10 \
    --warmup_epochs 0 \
    --REAL_SIZE 32 \
    --base_lr 3e-5 --save_prefix chair_svr_eval --loss mse --momentum 0.5 --checkpoint_freq 50 \
    --model_type dpf_im \
    --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/ --use_apex \
    --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
    --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
    --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
    --load_not_strict \
    --primitive_type cuboid \
    --num_part 16 \
    --stage2 \
    --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/deformed_primitive_field_30_50_16cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
    --load_svr_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_chair/dpf_svr_30_50_16cuboid_SVR_bs4/checkpoints/ckp-2200.pth \
    --SVR 

    # 0.0007857 0.00076725

    # 0.00077579 1000
    # 0.00075745 1200
    # 0.00075071 1400
    # 0.00075090 1600
    # 0.00074664 1800
#################################################### airplane

# CUDA_VISIBLE_DEVICES=4 python /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/utils/eval_svr.py \
#     --epochs 5010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 32 \
#     --base_lr 3e-5 --save_prefix chair_svr_eval --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type dpf_im \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --primitive_type cuboid \
#     --num_part 8 \
#     --stage2 \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/deformed_primitive_field_50_0.1_8cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
#     --load_svr_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_airplane/dpf_svr_50_0.1_8cuboid_SVR/checkpoints/ckp-800.pth \
#     --SVR 

# # 0.00041559 800
# # 0.00038403 1000
# # 0.00038842 1200
# # 0.00039308 1400

#################################################### table

# CUDA_VISIBLE_DEVICES=0 python /home/sist/cq14/kBAE/code/deformed_primitive_field/torch_base/utils/eval_svr.py \
#     --epochs 5010 \
#     --batch_size 128 \
#     --log_freq 10 \
#     --warmup_epochs 0 \
#     --REAL_SIZE 32 \
#     --base_lr 3e-5 --save_prefix chair_svr_eval --loss mse --momentum 0.5 --checkpoint_freq 50 \
#     --model_type dpf_im \
#     --dump_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/ --use_apex \
#     --shapenet_fps_path /home/sist/cq14/kBAE/dataset/shapenet/shapenet_fps_train.npy \
#     --shapenet_path /gpfs/home/sist/cq14/kBAE/dataset/SVR/all_vox256_img_train.hdf5 \
#     --tensorboard_path /home/sist/cq14/kBAE/code/deformed_primitive_field/output/logs \
#     --load_not_strict \
#     --primitive_type cuboid \
#     --num_part 8 \
#     --stage2 \
#     --load_model_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/deformed_primitive_field_10_30_100_8cuboid_64_coarse_exist_recon_1_bs_128_fps_compact_0.0001_stage2_tau4_deform4input/checkpoints/ckp-500.pth \
#     --load_svr_path /gpfs/home/sist/cq14/kBAE/code/deformed_primitive_field/output/dpf_table/dpf_svr_10_30_100_8cuboid_SVR/checkpoints/ckp-1000.pth \
#     --SVR 

#     # 0.00090979 800
#     # 0.00090286 1000
#     # 0.00094061 1200