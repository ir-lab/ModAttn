export CUDA_VISIBLE_DEVICES=0
python main_mujoco_robot_extended_1_obj.py \
    --data_source_root '/data/Documents/yzhou298/dataset/extended_modattn/might_be_wrong/pick_push_rotate' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/local/ASUAD/yzhou298/Documents/ckpts/extended_trial' \
    --save_ckpt \
    --ckpt None \
    --name extended-1-obj-dalle
