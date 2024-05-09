python ./utils/visualize.py \
    --save_prefix chair \
    --model_type deformed_primitive_field \
    --shapenet_path /home/qyshuai/code/paper/data/03001627_chair/03001627_train_vox.hdf5 \
    --load_not_strict \
    --primitive_type cuboid \
    --stage2 \
    --num_part 16 \
    --load_model_path /mnt/e/dataset/ckpt/chair.pth