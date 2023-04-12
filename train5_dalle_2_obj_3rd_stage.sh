export CUDA_VISIBLE_DEVICES=1
python main_mujoco_robot_vision_embed_3rd_stage_dalle_2_obj.py \
    --data_source_root '/data/Documents/yzhou298/dataset/extended_modattn/might_be_wrong/put_left_right_to' \
    --data_target_root '/home/local/ASUAD/yzhou298/Documents/dataset/extended_modattn/mae_embed/put_left_right_to/' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/local/ASUAD/yzhou298/Documents/ckpts/dalle_3_obj_put_left_right_to' \
    --save_ckpt \
    --supervised_attn \
    --curriculum_learning \
    --ckpt '/home/local/ASUAD/yzhou298/Documents/ckpts/dalle_3_obj_put_left_right_to/train-rgb-sub-attn-abs-action-vision-embed-no-stage-dalle-2-obj-fov40/0.pth' \
    --mae_root '/home/local/ASUAD/yzhou298/github/mae'
