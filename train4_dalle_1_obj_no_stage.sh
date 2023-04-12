export CUDA_VISIBLE_DEVICES=1
python main_mujoco_robot_vision_embed_no_stage_dalle_1_obj.py \
    --data_source_root '/home/yzhou298/data/part2/3_obj_pick_push_rotate' \
    --data_target_root '/home/yzhou298/data/part2_mae/3_obj_pick_push_rotate' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/yzhou298/data/ckpts/dalle_3_obj_pick_push_rotate' \
    --save_ckpt \
    --supervised_attn \
    --curriculum_learning \
    --ckpt None \
    --mae_root '/home/yzhou298/mae'
