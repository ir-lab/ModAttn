export CUDA_VISIBLE_DEVICES=0
python main_mujoco_robot_extended_2_obj_range_no_sup_attn.py \
    --data_source_root '/home/yzhou298/data/part3/put_left_right_to_40_fovy' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/yzhou298/data/ckpts/put_left_right_to_40_fovy' \
    --save_ckpt \
    --ckpt None \
    --name extended-2-obj-dalle-range-no-sup-attn
