export CUDA_VISIBLE_DEVICES=1
python main_mujoco_robot_vision_embed_3_stage_dalle_1_obj_fov40.py \
    --data_source_root '/data/Documents/yzhou298/dataset/extended_modattn/might_be_wrong/pick_push_rotate' \
    --data_target_root '/home/local/ASUAD/yzhou298/Documents/dataset/extended_modattn/mae_embed/pick_push_rotate/' \
    --train_set_path 'train' \
    --val_set_path 'val' \
    --ckpt_path '/home/local/ASUAD/yzhou298/Documents/ckpts/extended_trial' \
    --save_ckpt \
    --supervised_attn \
    --curriculum_learning \
    --ckpt None \
    --mae_root '/home/local/ASUAD/yzhou298/github/mae'
