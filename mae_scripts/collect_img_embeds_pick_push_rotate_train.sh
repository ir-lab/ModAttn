export CUDA_VISIBLE_DEVICES=1
python collect_img_embeds.py \
    --source_root '/mnt/disks/disk2/part2/3_obj_pick_push_rotate/train' \
    --target_root '/mnt/disks/disk2/part2_mae/3_obj_pick_push_rotate/train' \
    --chkpt_dir '/home/yzhou298/mae/mae_pretrain_vit_large.pth' \
    --mae_folder '/home/yzhou298/mae'
