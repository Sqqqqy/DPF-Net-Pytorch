python ./utils/visualize.py \
    --save_prefix airplane \
    --model_type deformed_primitive_field \
    --shapenet_path /home/qyshuai/code/paper/data/02691156_airplane/02691156_train_vox.hdf5 \
    --load_not_strict \
    --primitive_type cuboid \
    --stage2 \
    --num_part 8 \
    --load_model_path /mnt/e/dataset/ckpt/airplane.pth