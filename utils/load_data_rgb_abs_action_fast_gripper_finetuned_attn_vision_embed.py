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
import cv2
from PIL import Image
import sys

# For MAE vision encoder
import models_vit

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
    def __init__(self, data_dirs, 
    random=True, normalize='separate', length_total=91, depth_scale=1000., 
    chkpt_dir='mae_scripts/mae_pretrain_vit_large.pth', arch='vit_large_patch16', 
    source_root = '/home/local/ASUAD/yzhou298/Documents/dataset/',
    target_root = '/media/yzhou298/e/'):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--img1
        #         |--imgx
        #         |--states.json
        #     |--trial1
        #     |--...

        assert normalize in ['separate', 'together', 'none', 'panda', 'jaco2']

        self.visual_encoder = None
        self.chkpt_dir = chkpt_dir
        self.arch = arch


        all_dirs = []
        for data_dir in data_dirs:
            abs_data_dir = os.path.join(source_root, data_dir)
            all_dirs = all_dirs + [ os.path.join(data_dir, f.name) for f in os.scandir(abs_data_dir) if f.is_dir() ]

        # print(all_dirs)
        # print(len(all_dirs))
        self.random = random
        self.normalize = normalize
        self.length_total = length_total
        self.trials = []
        self.lengths_index = []
        self.target_name_to_idx = {
            'target2': 0,
            'coke': 1,
            'pepsi': 2,
            'milk': 3,
            'bread': 4,
            'bottle': 5,
        }

        self.idx_to_name = {
            0: 'target2',
            1: 'coke',
            2: 'pepsi',
            3: 'milk',
            4: 'bread',
            5: 'bottle',
        }

        self.action_inst_to_verb = {
            'push': ['push', 'move'],
            'pick': ['pick', 'pick up', 'raise', 'hold'],
            'put_down': ['put down', 'place down']
        }

        length = 0
        for trial in all_dirs:

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
            trial_dict['depth_paths'] = [os.path.join(trial, str(i) + '_depth_map.npy') for i in range(trial_dict['len'])]
            trial_dict['joint_angles'] = np.asarray([states_dict[i]['q'] for i in range(trial_dict['len'])])
            
            trial_dict['EE_xyzrpy'] = np.asarray([states_dict[i]['objects_to_track']['EE']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['EE']['rpy']) for i in range(trial_dict['len'])])
            
            trial_dict['target2'] = np.asarray([states_dict[i]['objects_to_track']['target2']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['target2']['rpy']) for i in range(trial_dict['len'])])
            trial_dict['coke'] = np.asarray([states_dict[i]['objects_to_track']['coke']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['coke']['rpy']) for i in range(trial_dict['len'])])
            trial_dict['pepsi'] = np.asarray([states_dict[i]['objects_to_track']['pepsi']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['pepsi']['rpy']) for i in range(trial_dict['len'])])
            trial_dict['milk'] = np.asarray([states_dict[i]['objects_to_track']['milk']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['milk']['rpy']) for i in range(trial_dict['len'])])
            trial_dict['bread'] = np.asarray([states_dict[i]['objects_to_track']['bread']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['bread']['rpy']) for i in range(trial_dict['len'])])
            trial_dict['bottle'] = np.asarray([states_dict[i]['objects_to_track']['bottle']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['bottle']['rpy']) for i in range(trial_dict['len'])])
            
            trial_dict['displacement'] = {}
            trial_dict['displacement']['target2'] = trial_dict['target2'] - trial_dict['EE_xyzrpy']
            trial_dict['displacement']['coke'] = trial_dict['coke'] - trial_dict['EE_xyzrpy']
            trial_dict['displacement']['pepsi'] = trial_dict['pepsi'] - trial_dict['EE_xyzrpy']
            trial_dict['displacement']['milk'] = trial_dict['milk'] - trial_dict['EE_xyzrpy']
            trial_dict['displacement']['bread'] = trial_dict['bread'] - trial_dict['EE_xyzrpy']
            trial_dict['displacement']['bottle'] = trial_dict['bottle'] - trial_dict['EE_xyzrpy']

            trial_dict['target_1_id'] = states_dict[0]['goal_object'][0]
            trial_dict['target_2_id'] = states_dict[0]['goal_object'][1]
            trial_dict['action_inst'] = states_dict[0]['action_inst']
            
            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict['len'] -= 1
            self.trials.append(trial_dict)
            length = length + trial_dict['len']
            self.lengths_index.append(length)

        self.weight = np.array([[-123.1531,  -0.4878,  -7.9859],
                [-1.5388, 84.6228,  -65.4262]]).T
        self.bias = np.array([111.9999, 108.1999])

        self.mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
        self.var = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1])
        self.mean_joints = np.array([-2.26736831e-01, 5.13238925e-01, -1.84928474e+00, 7.77270127e-01, 1.34229937e+00, 1.39107280e-03, 2.12295943e-01])
        self.var_joints = np.array([1.41245676e-01, 3.07248648e-02, 1.34113984e-01, 6.87947763e-02, 1.41992804e-01, 7.84910314e-05, 5.66411791e-02])
        self.mean_joints_together = 0.07375253452255098
        self.var_joints_together = 1.1682192251792096
        self.mean_joints_panda = np.array([-0.00357743, 0.29354134, 0.03703507, -2.01260356, -0.03319358, 0.76566389, 0.05069619, 0.01733641])
        self.std_joints_panda = np.array([0.07899751, 0.04528939, 0.27887484, 0.10307656, 0.06242473, 0.04195134, 0.27607541, 0.00033524]) ** (1/2)
        self.mean_joints_jaco = np.array([1.55675253, 4.4066693, 1.15504435, 1.71089821, 2.93128305, 1.74011258, 0.04558132])
        self.std_joints_jaco = np.array([0.29413278, 0.03133729, 0.33075052, 0.36203287, 1.37433513, 0.74981066, 1.17590218]) ** (1/2)

        self.mean_displacement = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])
        self.std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
        # print(np.mean(trial_dict['displacement']['target2'], axis=0))
        # print(np.std(trial_dict['displacement']['target2'], axis=0))
        # exit()
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])





    def rpy2rrppyy(self, rpy):
        rrppyy = [0] * 6
        for i in range(3):
            rrppyy[i * 2] = np.sin(rpy[i])
            rrppyy[i * 2 + 1] = np.cos(rpy[i])
        return rrppyy

    def noun_phrase_template(self, target_id):
        self.noun_phrase = {
            0: {
                'name': ['red', 'maroon'],
                'object': ['object', 'cube', 'square'],
            },
            1: {
                'name': ['red', 'coke'],
                'object': ['can', 'bottle'],
            },
            2: {
                'name': ['blue', 'pepsi'],
                'object': ['can', 'bottle'],
            },
            3: {
                'name': ['milk', 'white'],
                'object': ['carton', 'box'],
            },
            4: {
                'name': ['bread', 'yellow object', 'brown object'],
                'object': [''],
            },
            5: {
                'name': ['green', '', 'glass', 'green glass'],
                'object': ['bottle'],
            }
        }
        id_name = np.random.randint(len(self.noun_phrase[target_id]['name']))
        id_object = np.random.randint(len(self.noun_phrase[target_id]['object']))
        name = self.noun_phrase[target_id]['name'][id_name]
        obj = self.noun_phrase[target_id]['object'][id_object]
        return (name + ' ' + obj).strip()

    def verb_phrase_template(self, action_inst):
        if action_inst is None:
            action_inst = random.choice(list(self.action_inst_to_verb.keys()))
        action_id = np.random.randint(len(self.action_inst_to_verb[action_inst]))
        verb = self.action_inst_to_verb[action_inst][action_id]
        return verb.strip()

    def sentence_template(self, target_1_id, target_2_id, action_inst=None):
        # sentence = ''
        # verb = self.verb_phrase_template(action_inst)
        # sentence = sentence + verb
        # sentence = sentence + ' ' + self.noun_phrase_template(target_id)
        sentence = f'put {self.noun_phrase_template(target_1_id)} to the right of {self.noun_phrase_template(target_2_id)}'
        return sentence.strip()

    def xyz_to_xy(self, xyz, cam_pos=np.array([0, 1.5, 1]), theta=np.array([-1, 0, 3.14]), fov=60, output_size_x=224, output_size_y=224):
        """
        https://github.com/yusukeurakami/mujoco_2d_projection
        :param a: 3D coordinates of the joint in nparray [m]
        :param c: 3D coordinates of the camera in nparray [m]
        :param theta: camera 3D rotation (Rotation order of x->y->z) in nparray [rad]
        :param fov: field of view in integer [degree]
        :param e: 
        :return:
            - (bx, by) ==> 2D coordinates of the obj [pixel]
            - d ==> 3D coordinates of the joint (relative to the camera) [m]
        """
        e = 1
        output_size_x = output_size_x / 2
        output_size_y = output_size_y / 2
        # Get the vector from camera to object in global coordinate.
        ac_diff = xyz - cam_pos

        # Rotate the vector in to camera coordinate
        if not hasattr(self, 'transform'):
            self.transform = None
        if self.transform is None:
            x_rot = np.array([[1 ,0, 0],
                            [0, np.cos(theta[0]), np.sin(theta[0])],
                            [0, -np.sin(theta[0]), np.cos(theta[0])]])

            y_rot = np.array([[np.cos(theta[1]) ,0, -np.sin(theta[1])],
                        [0, 1, 0],
                        [np.sin(theta[1]), 0, np.cos(theta[1])]])

            z_rot = np.array([[np.cos(theta[2]) ,np.sin(theta[2]), 0],
                        [-np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]])

            transform = z_rot.dot(y_rot.dot(x_rot))
            self.transform = transform
        transform = self.transform
        d = transform.dot(ac_diff)    

        # scaling of projection plane using fov
        fov_rad = np.deg2rad(fov)    
        e *= output_size_y*1/np.tan(fov_rad/2.0)

        # Projection from d to 2D
        bx = e*d[0]/(d[2]) + output_size_x
        by = e*d[1]/(d[2]) + output_size_y
        # bx = output_size_x - bx
        bx = output_size_x * 2 - bx

        # return (bx, by), d
        return np.array([bx, by])

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]
        
        # img = io.imread(self.trials[trial_idx]['img_paths'][step_idx])[:,:,:3] / 255.
        # cv2.imshow('img', img)
        # cv2.waitKey(-1)

        if os.path.isfile(self.trials[trial_idx]['img_paths_npy'][step_idx]):
            img = np.load(self.trials[trial_idx]['img_paths_npy'][step_idx])
            img = torch.tensor(img, dtype=torch.float32)
        else:
            print(f"file missing: {self.trials[trial_idx]['img_paths_npy'][step_idx]}")
            folder = r'/' + r'/'.join(self.trials[trial_idx]['img_paths_npy'][step_idx].split(r'/')[:-1])
            
            if not os.path.exists(folder):
                os.system(f'mkdir -p {folder}')
            
            if self.visual_encoder is None:
                self.visual_encoder = ImgEncoder(self.chkpt_dir, self.arch)
            
            img = np.array(Image.open(self.trials[trial_idx]['img_paths'][step_idx]))[:,:,:3] / 255.
            img = img - imagenet_mean
            img = img / imagenet_std
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                img = self.visual_encoder(img)[0]
            img_np = img.clone().detach().cpu().numpy()
            np.save(self.trials[trial_idx]['img_paths_npy'][step_idx], img_np)

        length = torch.tensor(self.trials[trial_idx]['len'] - step_idx, dtype=torch.float32)
        ee_pos = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        ee_traj = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx:] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        ee_xy = self.xyz_to_xy(self.trials[trial_idx]['EE_xyzrpy'][step_idx][:3])


        if self.random:
            target_1, target_2 = random.sample(list(np.arange(6)), 2)
            action = None
        else:
            target_1 = self.target_name_to_idx[self.trials[trial_idx]['target_1_id']]
            target_2 = self.target_name_to_idx[self.trials[trial_idx]['target_2_id']]
            action = self.trials[trial_idx]['action_inst']

        sentence = self.sentence_template(target_1, target_2, action)
        sentence = clip.tokenize([sentence])
        idx_to_name = {
            0: 'target2',
            1: 'coke',
            2: 'pepsi',
            3: 'milk',
            4: 'bread',
            5: 'bottle',
        }
        target_1_pos = torch.tensor((self.trials[trial_idx][idx_to_name[target_1]][step_idx] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        target_2_pos = torch.tensor((self.trials[trial_idx][idx_to_name[target_2]][step_idx] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        target_1_xy = self.xyz_to_xy(self.trials[trial_idx][idx_to_name[target_1]][step_idx][:3])
        target_2_xy = self.xyz_to_xy(self.trials[trial_idx][idx_to_name[target_2]][step_idx][:3])
        displacement_1 = torch.tensor((self.trials[trial_idx]['displacement'][idx_to_name[target_1]][step_idx] - self.mean_displacement) / self.std_displacement, dtype=torch.float32)
        displacement_2 = torch.tensor((self.trials[trial_idx]['displacement'][idx_to_name[target_2]][step_idx] - self.mean_displacement) / self.std_displacement, dtype=torch.float32)
        # displacement_traj = torch.tensor((self.trials[trial_idx]['displacement'][idx_to_name[target]][step_idx:] - self.mean_displacement) / self.std_displacement)
        target_1 = torch.tensor(target_1, dtype=torch.int64)
        target_2 = torch.tensor(target_2, dtype=torch.int64)

        if self.normalize == 'separate':
            joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints) / (self.var_joints ** (1/2)), dtype=torch.float32)
            joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints) / (self.var_joints ** (1/2)), dtype=torch.float32)
        elif self.normalize == 'together':
            joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints_together) / (self.var_joints_together ** (1/2)), dtype=torch.float32)
            joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints_together) / (self.var_joints_together ** (1/2)), dtype=torch.float32)
        elif self.normalize == 'none':
            joint_angles = torch.tensor(self.trials[trial_idx]['joint_angles'][step_idx], dtype=torch.float32)
            joint_angles_traj = torch.tensor(self.trials[trial_idx]['joint_angles'][step_idx:], dtype=torch.float32)
        elif self.normalize == 'panda':
            joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints_panda) / self.std_joints_panda, dtype=torch.float32)
            joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints_panda) / self.std_joints_panda, dtype=torch.float32)
        elif self.normalize == 'jaco2':
            joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints_jaco) / self.std_joints_jaco, dtype=torch.float32)
            joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints_jaco) / self.std_joints_jaco, dtype=torch.float32)


        length_total = self.length_total
        length_left = max(length_total - ee_traj.shape[0], 0)

        if length_left > 0:
            ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
            ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

            joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(length_left, 1)
            joint_angles_traj = torch.cat((joint_angles_traj, joint_angles_traj_appendix), axis=0)

            # displacement_traj_appendix = displacement_traj[-1:].repeat(length_left, 1)
            # displacement_traj = torch.cat((displacement_traj, displacement_traj_appendix), axis=0)
        else:
            ee_traj = ee_traj[:length_total]
            joint_angles_traj = joint_angles_traj[:length_total]
            # displacement_traj = displacement_traj[:length_total]

        phis = torch.tensor(np.linspace(0.0, 1.0, length_total, dtype=np.float32))
        mask = torch.ones(phis.shape)

        return img, target_1, target_2, joint_angles, ee_pos, ee_traj, ee_xy, length, target_1_pos, target_2_pos, phis, mask, target_1_xy, target_2_xy, sentence[0], joint_angles_traj, displacement_1, displacement_2#, displacement_traj


def pad_collate_xy_lang(batch):
    (img, target_1, target_2, joint_angles, ee_pos, ee_traj, ee_xy, length, target_1_pos, target_2_pos, phis, mask, target_1_xy, target_2_xy, sentence, joint_angles_traj, displacement_1, displacement_2) = zip(*batch)

    img = torch.stack(img)
    target_1 = torch.stack(target_1)
    target_2 = torch.stack(target_2)
    joint_angles = torch.stack(joint_angles)
    ee_pos = torch.stack(ee_pos)
    length = torch.stack(length)
    target_1_pos = torch.stack(target_1_pos)
    target_2_pos = torch.stack(target_2_pos)
    ee_traj = torch.nn.utils.rnn.pad_sequence(ee_traj, batch_first=True, padding_value=0)
    ee_traj = torch.transpose(ee_traj, 1, 2)
    ee_xy = np.stack(ee_xy)
    phis = torch.nn.utils.rnn.pad_sequence(phis, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    target_1_xy = np.stack(target_1_xy)
    target_2_xy = np.stack(target_2_xy)
    sentence = torch.stack(sentence)
    joint_angles_traj = torch.nn.utils.rnn.pad_sequence(joint_angles_traj, batch_first=True, padding_value=0)
    joint_angles_traj = torch.transpose(joint_angles_traj, 1, 2)
    displacement_1 = torch.stack(displacement_1)
    displacement_2 = torch.stack(displacement_2)

    return  img, target_1, target_2, joint_angles, ee_pos, ee_traj, ee_xy, length, target_1_pos, target_2_pos, phis, mask, target_1_xy, target_2_xy, sentence, joint_angles_traj, displacement_1, displacement_2


if __name__ == '__main__':
    train_set_path = 'extended_modattn/put_right_to/split1'
    val_set_path = 'extended_modattn/put_right_to/split2'
    data_dirs = [
       val_set_path,
    ]
    dataset_train = DMPDatasetEERandTarXYLang(data_dirs, random=False, length_total=120, chkpt_dir='../mae_scripts/mae_pretrain_vit_large.pth')
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,
                                          shuffle=True, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    for img, target_1, target_2, joint_angles, ee_pos, ee_traj, ee_xy, length, target_1_pos, target_2_pos, phis, mask, target_1_xy, target_2_xy, sentence, joint_angles_traj, displacement_1, displacement_2 in data_loader_train:
        
        print(ee_xy)
        continue