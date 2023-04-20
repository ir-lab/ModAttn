export CUDA_VISIBLE_DEVICES=0
python main_mujoco_robot_extended_2_obj.py \
    --data_source_root '/data/Documents/yzhou298/dataset/extended_modattn/might_be_wrong/put_left_right_to' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/local/ASUAD/yzhou298/Documents/ckpts/dalle_3_obj_put_left_right_to' \
    --save_ckpt \
    --ckpt None \
    --name extended-2-obj-dalle
