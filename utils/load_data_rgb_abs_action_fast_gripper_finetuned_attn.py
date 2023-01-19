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


class DMPDatasetEERandTarXYLang(Dataset):
    def __init__(self, data_dirs, random=True, normalize='separate', length_total=91, depth_scale=1000.):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--img1
        #         |--imgx
        #         |--states.json
        #     |--trial1
        #     |--...

        assert normalize in ['separate', 'together', 'none', 'panda', 'jaco2']

        all_dirs = []
        for data_dir in data_dirs:
            all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
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

            states_json = os.path.join(trial, 'states.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            
            # There are (trial_dict['len']) states
            trial_dict['len'] = len(states_dict)
            trial_dict['img_paths'] = [os.path.join(trial, str(i) + '.png') for i in range(trial_dict['len'])]
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

            trial_dict['target_id'] = states_dict[0]['goal_object']
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
                'name': ['red', 'coke', 'cocacola'],
                'object': ['can', 'bottle'],
            },
            2: {
                'name': ['blue', 'pepsi', 'pepsi coke'],
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

    def sentence_template(self, target_id, action_inst=None):
        sentence = ''
        verb = self.verb_phrase_template(action_inst)
        sentence = sentence + verb
        sentence = sentence + ' ' + self.noun_phrase_template(target_id)
        return sentence.strip()

    def xyz_to_xy(self, xyz):
        xy = np.dot(xyz, self.weight) + self.bias
        xy[1] = 224 - xy[1]
        return xy

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]


        img = torch.tensor(io.imread(self.trials[trial_idx]['img_paths'][step_idx])[::-1,:,:3] / 255, dtype=torch.float32)
        # depth = np.load(self.trials[trial_idx]['depth_paths'][step_idx])[::-1,:]
        # depth = np.float32(depth) / 1000
        # depth[depth > 30] = 0
        # depth = torch.tensor(depth, dtype=torch.float32)

        # img = torch.cat((img, depth.unsqueeze(axis=2)), axis=2)

        length = torch.tensor(self.trials[trial_idx]['len'] - step_idx, dtype=torch.float32)
        ee_pos = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        ee_traj = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx:] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        ee_xy = self.xyz_to_xy(self.trials[trial_idx]['EE_xyzrpy'][step_idx][:3])


        if self.random:
            target = np.random.randint(6)
            action = None
        else:
            target = self.target_name_to_idx[self.trials[trial_idx]['target_id']]
            action = self.trials[trial_idx]['action_inst']

        sentence = self.sentence_template(target, action)
        sentence = clip.tokenize([sentence])
        idx_to_name = {
            0: 'target2',
            1: 'coke',
            2: 'pepsi',
            3: 'milk',
            4: 'bread',
            5: 'bottle',
        }
        target_pos = torch.tensor((self.trials[trial_idx][idx_to_name[target]][step_idx] - self.mean) / (self.var ** (1/2)), dtype=torch.float32)
        target_xy = self.xyz_to_xy(self.trials[trial_idx][idx_to_name[target]][step_idx][:3])
        displacement = torch.tensor((self.trials[trial_idx]['displacement'][idx_to_name[target]][step_idx] - self.mean_displacement) / self.std_displacement, dtype=torch.float32)
        # displacement_traj = torch.tensor((self.trials[trial_idx]['displacement'][idx_to_name[target]][step_idx:] - self.mean_displacement) / self.std_displacement)
        target = torch.tensor(target, dtype=torch.int64)

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

        return img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence[0], joint_angles_traj, displacement#, displacement_traj


def pad_collate_xy_lang(batch):
    (img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement) = zip(*batch)

    img = torch.stack(img)
    target = torch.stack(target)
    joint_angles = torch.stack(joint_angles)
    ee_pos = torch.stack(ee_pos)
    length = torch.stack(length)
    target_pos = torch.stack(target_pos)
    ee_traj = torch.nn.utils.rnn.pad_sequence(ee_traj, batch_first=True, padding_value=0)
    ee_traj = torch.transpose(ee_traj, 1, 2)
    ee_xy = np.stack(ee_xy)
    phis = torch.nn.utils.rnn.pad_sequence(phis, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    target_xy = np.stack(target_xy)
    sentence = torch.stack(sentence)
    joint_angles_traj = torch.nn.utils.rnn.pad_sequence(joint_angles_traj, batch_first=True, padding_value=0)
    joint_angles_traj = torch.transpose(joint_angles_traj, 1, 2)
    displacement = torch.stack(displacement)

    return  img, target, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj, displacement


if __name__ == '__main__':
    data_dirs = [
        '/mnt/disk3/dataset/mujoco_dataset_pick_push_RGBD_different_angles_fast_gripper_224_jaco2_test/'
    ]
    dataset = DMPDatasetEERandTarXYLang(data_dirs, random=False, normalize='jaco2')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                          shuffle=True, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    for img, target, joint_angles, ee_pos, ee_traj, length, target_pos, phis, mask, target_xy, sentence, joint_angles_traj in dataloader:
        # print(target, joint_angles, ee_pos, ee_traj, length, target_pos)
        # print(length, len(ee_traj))
        print(target.shape, joint_angles.shape, ee_pos.shape, ee_traj.shape, length.shape, target_pos.shape, phis.shape, mask.shape, sentence.shape, img.shape, joint_angles_traj.shape)
        print(target[0], target_pos[0])

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img[0].numpy())

        ax = fig.add_subplot(1, 2, 2)
        xs = np.arange(joint_angles_traj.shape[2])
        ax.plot(xs, joint_angles_traj[0, 0, :], label='0')
        ax.plot(xs, joint_angles_traj[0, 1, :], label='1')
        ax.plot(xs, joint_angles_traj[0, 2, :], label='2')
        ax.plot(xs, joint_angles_traj[0, 3, :], label='3')
        ax.plot(xs, joint_angles_traj[0, 4, :], label='4')
        ax.plot(xs, joint_angles_traj[0, 5, :], label='5')
        ax.plot(xs, joint_angles_traj[0, 6, :], label='6')
        ax.legend()
        
        # plt.show()
