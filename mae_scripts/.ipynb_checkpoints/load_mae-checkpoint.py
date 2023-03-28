import sys
import torch

sys.path.append('/home/local/ASUAD/yzhou298/github/mae')
import models_mae

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:2')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

# chkpt_dir = 'mae_pretrain_vit_large.pth'
chkpt_dir = 'mae_visualize_vit_large.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print(model)