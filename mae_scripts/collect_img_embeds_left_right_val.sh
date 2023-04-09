export CUDA_VISIBLE_DEVICES=0
python collect_img_embeds.py \
    --source_root '/mnt/disks/disk2/part3/put_left_right_to/val' \
    --target_root '/mnt/disks/disk3/part3_mae/put_left_right_to/val' \
    --chkpt_dir '/home/yzhou298/mae/mae_pretrain_vit_large.pth' \
    --mae_folder '/home/yzhou298/mae'
