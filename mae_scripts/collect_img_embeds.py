import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random
from PIL import Image
import argparse

# For MAE vision encoder
import sys

def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = getattr(models_vit, arch)(global_pool=True)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


class ImgEncoder(nn.Module):
    def __init__(self, chkpt_dir = 'mae_pretrain_vit_large.pth', arch='vit_large_patch16'):
        super(ImgEncoder, self).__init__()
        self.model_mae = prepare_model(chkpt_dir, arch)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.model_mae.forward_features(x)
        return out


class DMPDatasetEERandTarXYLang(Dataset):
    def __init__(self, data_dirs, source_root, target_root):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--img1
        #         |--imgx
        #         |--states.json
        #     |--trial1
        #     |--...

        all_dirs = []
        for data_dir in data_dirs:
            abs_data_dir = os.path.join(source_root, data_dir)
            all_dirs = all_dirs + [ os.path.join(data_dir, f.name) for f in os.scandir(abs_data_dir) if f.is_dir() ]
        # print(all_dirs)
        # print(len(all_dirs))

        self.trials = []
        self.lengths_index = []
        length = 0
        for idx, trial in enumerate(all_dirs):
            print(idx, len(all_dirs), trial)

            # trial_id = int(trial.strip().split(r'/')[-1])
            # if not ((trial_id >= 1700) and (trial_id < 1725)):
            #     continue


            trial_dict = {}

            states_json = os.path.join(source_root, trial, 'states.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            
            # There are (trial_dict['len']) states
            trial_dict['len'] = len(states_dict)
            trial_dict['img_paths'] = [os.path.join(source_root, trial, str(i) + '.png') for i in range(trial_dict['len'])]
            trial_dict['img_paths_npy'] = [os.path.join(target_root, trial, str(i) + '.npy') for i in range(trial_dict['len'])]
            # print(target_root)
            # print(trial)
            # input()
            # print(trial_dict['img_paths_npy'])
            # input()
            
            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict['len'] -= 1
            self.trials.append(trial_dict)
            length = length + trial_dict['len']
            self.lengths_index.append(length)

        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])


    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]

        # img = io.imread(self.trials[trial_idx]['img_paths'][step_idx])[::-1,:,:3] / 255.
        img = np.array(Image.open(self.trials[trial_idx]['img_paths'][step_idx]))[:,:,:3] / 255.

        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)

        return img, self.trials[trial_idx]['img_paths'][step_idx], self.trials[trial_idx]['img_paths_npy'][step_idx]


def pad_collate_xy_lang(batch):
    (img, img_path, npy_path) = zip(*batch)

    img = torch.stack(img)

    return img, img_path, npy_path


if __name__ == '__main__':
    data_dirs = [
        'extended_modattn/put_right_to/split1',
        'extended_modattn/put_right_to/split2'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', default='/home/local/ASUAD/yzhou298/Documents/dataset/')
    parser.add_argument('--target_root', default='/media/yzhou298/e/')
    parser.add_argument('--chkpt_dir', default='./mae_pretrain_vit_large.pth')
    parser.add_argument('--arch', default='vit_large_patch16')
    parser.add_argument('--mae_folder', default='/home/local/ASUAD/yzhou298/github/mae')
    args = parser.parse_args()

    # source_root = '/home/local/ASUAD/yzhou298/Documents/dataset/'
    # target_root = '/media/yzhou298/e/'
    # chkpt_dir='./mae_pretrain_vit_large.pth'
    # arch='vit_large_patch16'

    source_root = args.source_root
    target_root = args.target_root
    chkpt_dir = args.chkpt_dir
    mae_folder = args.mae_folder
    arch = args.arch


    sys.path.append(mae_folder)

    import models_vit
    dataset = DMPDatasetEERandTarXYLang(data_dirs, source_root, target_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=384,
                                          shuffle=False, num_workers=16,
                                          collate_fn=pad_collate_xy_lang)
    visual_encoder = ImgEncoder(chkpt_dir, arch).to('cuda')

    for idx, (img, img_path, npy_path) in enumerate(dataloader):

        print(idx, len(dataloader))

        print(img.shape)
        print(len(img_path))
        print(len(npy_path))
        print(img_path[:5])
        print(npy_path[:5])
        with torch.no_grad():
            img_embedding = visual_encoder(img.to('cuda'))
        img_embedding = img_embedding.detach().cpu().numpy()
        for i in range(img_embedding.shape[0]):
            folder = r'/' + r'/'.join(npy_path[i].split(r'/')[:-1])
            if not os.path.exists(folder):
                # os.mkdir(folder)
                os.system(f'mkdir -p {folder}')
            np.save(npy_path[i], img_embedding[i])
