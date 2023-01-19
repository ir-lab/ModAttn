import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import clip
import contextlib


class TaskIDEncoder(nn.Module):
    def __init__(self, num_tasks, embedding_size):
        super(TaskIDEncoder, self).__init__()
        self.id_to_embedding = nn.Embedding(num_tasks, 32 * 32)
        self.layer1 = nn.Linear(32 * 32, 16 * 16)
        self.layer2 = nn.Linear(16 * 16, embedding_size)

    def forward(self, task_id):
        x = self.id_to_embedding(task_id)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        return x


# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ImgEncoder(nn.Module):
    def __init__(self, img_size=224, ngf=64, channel_multiplier=4, input_nc=3):
        super(ImgEncoder, self).__init__()
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*channel_multiplier//2,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier//2),
                                   nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier // 2,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.layer4 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.layer5 = nn.Sequential(ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier),
                                    ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier),
                                    ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier))
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py
class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

class Controller(nn.Module):
    def __init__(self, num_traces=3, num_weight_points=11, embedding_size=128):
        super(Controller, self).__init__()

        self.layer1_1 = nn.Linear(embedding_size, 16 * 16 * 8)
        self.layer2 = nn.Linear(16 * 16 * 8, 16 * 16 * 4)
        self.layer3 = nn.Linear(16 * 16 * 4, 16 * 16 * 1)
        self.layer4 = nn.Linear(16 * 16 * 1, num_traces * num_weight_points)

        self.layer1_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer2_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer3_bn = nn.BatchNorm1d(16 * 16 * 4)

    def forward(self, goal_embedding):
        x = F.selu(self.layer1_1(goal_embedding))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        x = self.layer4(x)
        return x


class JointsEncoder(nn.Module):
    def __init__(self, num_traces, embedding_size):
        super(JointsEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(num_traces, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, embedding_size)

    def forward(self, joints):
        x = self.flatten(joints)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        return x


# Some code from DETR
class Backbone(nn.Module):
    def __init__(self, img_size, embedding_size, num_traces_in=7, num_traces_out=10, num_weight_points=91, input_nc=4, device=torch.device('cuda')):
        super(Backbone, self).__init__()

        self.device = device
        self.num_traces_in = num_traces_in
        self.num_traces_out = num_traces_out
        self.num_weight_points = num_weight_points

        # Visual Pathway
        self.visual_encoder = ImgEncoder(input_nc=input_nc)
        self.visual_encoder_narrower = nn.Sequential(
            nn.Linear(256, embedding_size),
            nn.ReLU())
        self.img_embed_merge_pos_embed = nn.Sequential(
            nn.Linear(embedding_size + 2, embedding_size),
            nn.ReLU())
        self.img_embed_to_qkv = []
        for i in range(3):
            self.img_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.img_embed_to_qkv = nn.ModuleList(self.img_embed_to_qkv)

        # Task Pathway
        self.task_id_encoder, _ = clip.load("ViT-B/32", self.device)
        self.task_id_embedding_narrower = nn.Linear(512, embedding_size)
        self.task_embed_to_qkv = []
        for i in range(3):
            self.task_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.task_embed_to_qkv = nn.ModuleList(self.task_embed_to_qkv)

        # Joints Pathway
        self.joints_encoder = JointsEncoder(num_traces_in, embedding_size)
        self.joints_embed_to_qkv = []
        for i in range(3):
            self.joints_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.joints_embed_to_qkv = nn.ModuleList(self.joints_embed_to_qkv)

        # Requests and their mappings to qkv
        self.tar_pos_slot = nn.Parameter(torch.rand(embedding_size))
        self.displace_slot = nn.Parameter(torch.rand(embedding_size))
        self.ee_pos_slot = nn.Parameter(torch.rand(embedding_size))
        self.ee_pos2_slot = nn.Parameter(torch.rand(embedding_size))
        self.action_slot = nn.Parameter(torch.rand(embedding_size))
        self.requests = [self.tar_pos_slot, self.displace_slot, self.ee_pos_slot, self.ee_pos2_slot, self.action_slot]
        self.requests_to_queries = []
        self.requests_to_keys = []
        self.requests_to_values = []
        for i in range(len(self.requests)):
            self.requests_to_queries.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
            self.requests_to_keys.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
            self.requests_to_values.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
        self.requests_to_queries = nn.ModuleList(self.requests_to_queries)
        self.requests_to_keys = nn.ModuleList(self.requests_to_keys)
        self.requests_to_values = nn.ModuleList(self.requests_to_values)

        # Cortex Module
        self.seg_embed1 = nn.Embedding(8, embedding_size)
        self.seg_embed2 = nn.Embedding(8, embedding_size)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)
        self.attn4 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)

        self.embed_to_target_position = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, num_traces_out-1))

        self.embed_to_displacement = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, num_traces_out-1))

        self.embed_to_ee_pos = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, num_traces_out-1))

        self.controller = Controller(num_traces=num_traces_out, num_weight_points=num_weight_points, embedding_size=embedding_size)

        self.fixed = nn.ModuleList([
            self.visual_encoder,
            self.visual_encoder_narrower,
            self.img_embed_merge_pos_embed,
            self.task_id_encoder,
            self.controller.layer2,
            self.controller.layer3,
            self.controller.layer4,
        ])

    def _embed_to_qkv_(self, embed, qkv_list):
        q = qkv_list[0](embed)
        k = qkv_list[1](embed)
        v = qkv_list[2](embed)
        return q, k, v

    def _img_pathway_(self, img):
        # Comprehensive Visual Encoder. img_embedding is the square token list
        img_embedding = self.visual_encoder(img)

        # Merge H and W dimensions
        _, _, H, W = img_embedding.shape
        img_embedding = img_embedding.reshape(img_embedding.shape[0], img_embedding.shape[1], -1).permute(0, 2, 1)

        # Narrow the embedding size
        img_embedding = self.visual_encoder_narrower(img_embedding)

        # Prepare the pos embedding for attention
        batch_size, H_W, _ = img_embedding.shape
        pos_embed_w = torch.tensor(np.arange(W), dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(batch_size, H, 1).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        pos_embed_h = torch.tensor(np.arange(H), dtype=torch.float32).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, W).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        
        # Concatenate pos embedding with the img embedding
        img_embedding = torch.cat((img_embedding, pos_embed_w, pos_embed_h), dim=2)
        img_embedding = self.img_embed_merge_pos_embed(img_embedding)
        img_embed_query, img_embed_key, img_embed_value = self._embed_to_qkv_(img_embedding, self.img_embed_to_qkv)

        return img_embed_query, img_embed_key, img_embed_value

    def _task_id_pathway_(self, lang):
        with torch.no_grad():
            task_embedding = self.task_id_encoder.encode_text(lang)
        task_embedding = task_embedding.float()
        task_embedding = self.task_id_embedding_narrower(task_embedding)
        task_embedding = task_embedding.unsqueeze(1)
        task_embed_query, task_embed_key, task_embed_value = self._embed_to_qkv_(task_embedding, self.task_embed_to_qkv)
        return task_embed_query, task_embed_key, task_embed_value

    def _requests_pathway_(self, batch_size):
        requests_qs = []
        requests_ks = []
        requests_vs = []
        for i in range(len(self.requests)):
            requests_qs.append(self.requests_to_queries[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
            requests_ks.append(self.requests_to_keys[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
            requests_vs.append(self.requests_to_values[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
        requests_qs = torch.cat(requests_qs, dim=1)
        requests_ks = torch.cat(requests_ks, dim=1)
        requests_vs = torch.cat(requests_vs, dim=1)
        return requests_qs, requests_ks, requests_vs

    def _joints_pathway_(self, joints):
        joints_embedding = self.joints_encoder(joints).unsqueeze(1)
        joints_embed_query, joints_embed_key, joints_embed_value = self._embed_to_qkv_(joints_embedding, self.joints_embed_to_qkv)
        return joints_embed_query, joints_embed_key, joints_embed_value

    def _status_embed_to_qkv_(self, status_embed):
        last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = self._embed_to_qkv_(status_embed, self.status_embed_to_qkv)
        return last_subjective_part_query, last_subjective_part_key, last_subjective_part_value

    def _update_cortex_status_(self, last_state_embed, perception_query, perception_key, perception_value):
        last_subjective_part = last_state_embed[:, :5, :]
        # last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = self._status_embed_to_qkv_(last_subjective_part)
        last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = last_subjective_part, last_subjective_part, last_subjective_part
        cortex_query = torch.cat((last_subjective_part_query, perception_query), dim=1)
        cortex_key = torch.cat((last_subjective_part_key, perception_key), dim=1)
        cortex_value = torch.cat((last_subjective_part_value, perception_value), dim=1)
        return cortex_query, cortex_key, cortex_value

    def _get_segment_embed_(self, batch_size, embed_shape, layer):
        segment_ids = [0, 1, 2, 3, 4] + embed_shape * [5] + [6, 7]
        segment_ids = np.array(segment_ids)
        segment_ids = torch.tensor(segment_ids, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        if layer == 1:
            segment_ids = self.seg_embed1(segment_ids)
        elif layer == 2:
            segment_ids = self.seg_embed2(segment_ids)
        elif layer == 3:
            segment_ids = self.seg_embed3(segment_ids)
        elif layer == 4:
            segment_ids = self.seg_embed4(segment_ids)
        return segment_ids

    def attn_forward(self, batch_size, embed_shape, cortex_query, cortex_key, cortex_value, attn_layer):        
        segment_embed = self._get_segment_embed_(batch_size=batch_size, embed_shape=embed_shape, layer=1)
        cortex_query = cortex_query + segment_embed
        cortex_key = cortex_key + segment_embed
        cortex_value = cortex_value + segment_embed
        state_embedding, attn_map = attn_layer(cortex_query[:, :5, :], cortex_key, cortex_value, need_weights=True, attn_mask=None)
        return state_embedding, attn_map

    def forward(self, img, joints, sentence, phis, stage=0):

        # Image Pathway
        img_embed_query, img_embed_key, img_embed_value = self._img_pathway_(img)

        # Task ID Pathway
        task_embed_query, task_embed_key, task_embed_value = self._task_id_pathway_(sentence)

        # EE POS Pathway
        joints_embed_query, joints_embed_key, joints_embed_value = self._joints_pathway_(joints)

        # Concatenate All Perception Pathways
        perception_query = torch.cat((img_embed_query, joints_embed_query, task_embed_query), dim=1)
        perception_key = torch.cat((img_embed_key, joints_embed_key, task_embed_key), dim=1)
        perception_value = torch.cat((img_embed_value, joints_embed_value, task_embed_value), dim=1)

        # Requests
        requests_qs, requests_ks, requests_vs = self._requests_pathway_(batch_size=img.shape[0])

        # Concatenate All Subjective and Objective Parts
        cortex_query = torch.cat((requests_qs, perception_query), dim=1)
        cortex_key = torch.cat((requests_ks, perception_key), dim=1)
        cortex_value = torch.cat((requests_vs, perception_value), dim=1)

        # Attention Layer1
        state_embedding, attn_map = self.attn_forward(
            batch_size=img.shape[0], 
            embed_shape=img_embed_query.shape[1], 
            cortex_query=cortex_query, 
            cortex_key=cortex_key, 
            cortex_value=cortex_value, 
            attn_layer=self.attn)

        # Attention Layer2
        cortex_query2, cortex_key2, cortex_value2 = self._update_cortex_status_(state_embedding, perception_query, perception_key, perception_value)
        state_embedding2, attn_map2 = self.attn_forward(
            batch_size=img.shape[0], 
            embed_shape=img_embed_query.shape[1], 
            cortex_query=cortex_query2, 
            cortex_key=cortex_key2, 
            cortex_value=cortex_value2, 
            attn_layer=self.attn2)
        # Post-attn operations. Predict the results from the state embedding
        if stage == 0:
            return attn_map, attn_map2

        # Attention Layer3
        cortex_query3, cortex_key3, cortex_value3 = self._update_cortex_status_(state_embedding2, perception_query, perception_key, perception_value)
        state_embedding3, attn_map3 = self.attn_forward(
            batch_size=img.shape[0], 
            embed_shape=img_embed_query.shape[1], 
            cortex_query=cortex_query3, 
            cortex_key=cortex_key3, 
            cortex_value=cortex_value3, 
            attn_layer=self.attn3)
        # Post-attn operations. Predict the results from the state embedding
        target_position_pred = self.embed_to_target_position(state_embedding3[:, 0, :])
        displacement_pred = self.embed_to_displacement(state_embedding3[:, 1, :])
        ee_pos_pred = self.embed_to_ee_pos(state_embedding3[:, 2, :])
        if stage == 1:
            return target_position_pred, ee_pos_pred, displacement_pred, attn_map, attn_map2, attn_map3

        # Attention Layer4
        cortex_query4, cortex_key4, cortex_value4 = self._update_cortex_status_(state_embedding3, perception_query, perception_key, perception_value)
        state_embedding4, attn_map4 = self.attn_forward(
            batch_size=img.shape[0], 
            embed_shape=img_embed_query.shape[1], 
            cortex_query=cortex_query4, 
            cortex_key=cortex_key4, 
            cortex_value=cortex_value4, 
            attn_layer=self.attn4)

        dmp_weights = self.controller(state_embedding4[:, 4, :])
        dmp_weights = dmp_weights.reshape(img.shape[0], self.num_traces_out, self.num_weight_points)
        
        centers = torch.tensor(np.linspace(0.0, 1.0, self.num_weight_points, dtype=np.float32)).to(self.device)
        centers = centers.unsqueeze(0).unsqueeze(1).repeat(img.shape[0], self.num_traces_out, 1).reshape(img.shape[0] * self.num_traces_out, self.num_weight_points)
        dmp_weights = dmp_weights.reshape(img.shape[0] * self.num_traces_out, self.num_weight_points)
        phis = phis.reshape(img.shape[0] * self.num_traces_out, phis.shape[-1])
        trajectory = Interp1d()(centers, dmp_weights, phis)
        trajectory = trajectory.reshape(img.shape[0], self.num_traces_out, phis.shape[-1])
        dmp_weights = dmp_weights.reshape(img.shape[0], self.num_traces_out, self.num_weight_points)

        return target_position_pred, ee_pos_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory
