export CUDA_VISIBLE_DEVICES=0
python main_mujoco_robot_extended_1_obj_no_sup_attn.py \
    --data_source_root '/home/yzhou298/data/part2/pick_push_rotate_40_fovy' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/yzhou298/data/ckpts/pick_push_rotate_40_fovy' \
    --save_ckpt \
    --ckpt None \
    --name 'extended-1-obj-dalle-no-sup-attn'
