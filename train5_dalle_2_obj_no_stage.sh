export CUDA_VISIBLE_DEVICES=0
python main_mujoco_robot_vision_embed_no_stage_dalle_2_obj.py \
    --data_source_root '/home/yzhou298/data/part3/put_left_right_to/' \
    --data_target_root '/home/yzhou298/data2/part3_mae/put_left_right_to/' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/yzhou298/data/ckpts/dalle_3_obj_put_left_right_to' \
    --save_ckpt \
    --supervised_attn \
    --curriculum_learning \
    --ckpt None \
    --mae_root '/home/yzhou298/mae'
