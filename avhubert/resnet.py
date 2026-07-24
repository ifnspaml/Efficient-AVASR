# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import re
import logging
import math
from typing import Optional
import torch
import torch.nn as nn
import pdb

DBG=True if len(sys.argv) == 1 or re.match(r"(.*)(prune.py|save_final_ckpt.py|merge.py)", sys.argv[0]) else False

if DBG:
    from hardconcrete import HardConcrete
    from pruning_utils import prune_conv2d_layer, prune_layer_norm, prune_prelu
else:
    from .hardconcrete import HardConcrete
    from .pruning_utils import prune_conv2d_layer, prune_layer_norm, prune_prelu


logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )


def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='relu', block_detailed=None, prune_conv_channels=False, hard_concrete=None):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu']
        self.relu_type = relu_type

        self.conv1_inplanes = block_detailed[0][0] if block_detailed else inplanes
        self.conv1_planes = block_detailed[0][-1] if block_detailed else planes

        self.conv2_inplanes = block_detailed[-1][0] if block_detailed else planes
        self.conv2_planes = block_detailed[-1][-1] if block_detailed else planes

        self.conv1 = conv3x3(self.conv1_inplanes, self.conv1_planes, stride)
        self.bn1 = nn.BatchNorm2d(self.conv1_planes)

        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=self.conv1_planes)
            self.relu2 = nn.PReLU(num_parameters=self.conv2_planes)
        else:
            raise Exception('relu type not implemented')

        self.conv2 = conv3x3(self.conv2_inplanes, self.conv2_planes)
        self.bn2 = nn.BatchNorm2d(self.conv2_planes)
        
        self.downsample = downsample
        self.stride = stride

        if prune_conv_channels:
            self.hard_concrete_conv1 = HardConcrete(n_in=self.conv1_planes, init_mean=0.01)
            self.hard_concrete_conv2 = HardConcrete(n_in=self.conv2_planes, init_mean=0.01) if hard_concrete is None else hard_concrete
        else:
            self.hard_concrete_conv1 = None
            self.hard_concrete_conv2 = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        if self.hard_concrete_conv1 is not None:
            channel_mask = self.hard_concrete_conv1() # hard concrete mask, (out_channels,)
            out = out * channel_mask.unsqueeze(-1).unsqueeze(-1)    
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        if self.hard_concrete_conv2 is not None:
            channel_mask = self.hard_concrete_conv2() # hard concrete mask, (out_channels,)
            out = out * channel_mask.unsqueeze(-1).unsqueeze(-1)

        return out
    
    def prune(self, prev_index: Optional[torch.LongTensor]=None):
        if prev_index is not None:
            prune_conv2d_layer(self.conv1, prev_index, "input")
            if self.downsample is not None:
                prune_conv2d_layer(self.downsample[-2], prev_index, dim="input")

        if self.hard_concrete_conv1 is not None:
            assert not self.hard_concrete_conv1.training
            mask = self.hard_concrete_conv1()    # (out_features,)
            conv1_index = mask.nonzero().squeeze(-1)    # 2D -> 1D
            assert len(conv1_index) > 0, f"Conv channels pruned to zero at index {self}"
            prune_conv2d_layer(self.conv1, conv1_index, "output")
            prune_layer_norm(self.bn1, conv1_index)
            if self.relu_type == "prelu":
                prune_prelu(self.relu1, conv1_index)
            prune_conv2d_layer(self.conv2, conv1_index, "input")
            self.hard_concrete_conv1 = None
        else:
            conv1_index = None

        if self.hard_concrete_conv2 is not None:
            assert not self.hard_concrete_conv2.training
            mask = self.hard_concrete_conv2()    # (out_features,)
            conv2_index = mask.nonzero().squeeze(-1)    # 2D -> 1D
            assert len(conv2_index) > 0, f"Conv channels pruned to zero at index {self}"
            prune_conv2d_layer(self.conv2, conv2_index, "output")
            prune_layer_norm(self.bn2, conv2_index)
            if self.relu_type == "prelu":
                prune_prelu(self.relu2, conv2_index)
            if self.downsample is not None:
                prune_conv2d_layer(self.downsample[-2], conv2_index, dim="output")
                prune_layer_norm(self.downsample[-1], conv2_index)
            self.hard_concrete_conv2 = None
        else:
            conv2_index = None
        
        self.conv1_inplanes = len(prev_index) if prev_index is not None else self.conv1_inplanes
        self.conv1_planes = len(conv1_index) if conv1_index is not None else self.conv1_planes
        self.conv2_planes = len(conv2_index) if conv2_index is not None else self.conv2_planes

        return conv2_index, [(self.conv1_inplanes, self.conv1_planes), (self.conv1_planes, self.conv2_planes)]
    
    def get_num_params_and_out_channels(self, in_channels):
        if self.hard_concrete_conv1 is not None:
            conv1_out_channels = self.hard_concrete_conv1.l0_norm()
        else:
            conv1_out_channels = self.conv1_planes
        num_params = in_channels * conv1_out_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] # conv
        num_params += 2 * conv1_out_channels # batch norm
        if self.relu_type == 'prelu':
            num_params += conv1_out_channels # prelu
        if self.hard_concrete_conv2 is not None:
            conv2_out_channels = self.hard_concrete_conv2.l0_norm()
        else:
            conv2_out_channels = self.conv2_planes
        num_params += conv1_out_channels * conv2_out_channels * self.conv2.kernel_size[0] * self.conv2.kernel_size[1] # conv
        num_params += 2 * conv2_out_channels # batch norm
        if self.relu_type == 'prelu':
            num_params += conv2_out_channels # prelu
        if self.downsample is not None:
            num_params += conv2_out_channels * in_channels # conv
            num_params += conv2_out_channels * 2 # batch norm
        return num_params, conv2_out_channels


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, relu_type='relu', gamma_zero=False, avg_pool_downsample=False, layer_detailed=None, prune_conv_channels=False):
        self.inplanes = layer_detailed[0][0][0][0] if layer_detailed else 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], layer_detailed=layer_detailed[0] if layer_detailed else None, prune_conv_channels=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_detailed=layer_detailed[1] if layer_detailed else None, prune_conv_channels=prune_conv_channels)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_detailed=layer_detailed[2] if layer_detailed else None, prune_conv_channels=prune_conv_channels)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_detailed=layer_detailed[3] if layer_detailed else None, prune_conv_channels=prune_conv_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, layer_detailed=None, prune_conv_channels=False):
        downsample = None
        planes = layer_detailed[0][-1][-1] if layer_detailed else planes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, 
                                                 outplanes = planes * block.expansion, 
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type, block_detailed=layer_detailed[0] if layer_detailed else None, prune_conv_channels=prune_conv_channels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if prune_conv_channels:
                hard_concrete = layers[0].hard_concrete_conv2 #Todo: rethink this 0 (still 0 if more than 2 blocks?)
            else:
                hard_concrete = None
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type, block_detailed=layer_detailed[i] if layer_detailed else None, prune_conv_channels=prune_conv_channels, hard_concrete=hard_concrete))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def prune(self):
        new_config = []
        _, layer1_0_config = self.layer1[0].prune()
        _, layer1_1_config = self.layer1[1].prune()
        new_config.append([layer1_0_config, layer1_1_config])
        layer2_0_index, layer2_0_config = self.layer2[0].prune()
        layer2_1_index, layer2_1_config = self.layer2[1].prune(layer2_0_index)
        new_config.append([layer2_0_config, layer2_1_config])
        layer3_0_index, layer3_0_config = self.layer3[0].prune(layer2_1_index)
        layer3_1_index, layer3_1_config = self.layer3[1].prune(layer3_0_index)
        new_config.append([layer3_0_config, layer3_1_config])
        layer4_0_index, layer4_0_config = self.layer4[0].prune(layer3_1_index)
        layer4_1_index, layer4_1_config = self.layer4[1].prune(layer4_0_index)
        new_config.append([layer4_0_config, layer4_1_config])
        return layer4_1_index, new_config

    def get_num_params_and_out_channels(self):
        num_params, out_channels = self.layer1[0].get_num_params_and_out_channels(64)
        layer_params, out_channels = self.layer1[1].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer2[0].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer2[1].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer3[0].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer3[1].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer4[0].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        layer_params, out_channels = self.layer4[1].get_num_params_and_out_channels(out_channels)
        num_params += layer_params
        return num_params, out_channels


class ResEncoder(nn.Module):
    def __init__(self, relu_type, weights, layer_detailed=None, prune_conv_channels=False):
        super(ResEncoder, self).__init__()
        self.frontend_nout = layer_detailed[0][0][0][0] if layer_detailed else 64
        self.backend_out = layer_detailed[-1][-1][-1][-1] if layer_detailed else 512
        self.relu_type = relu_type
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type, layer_detailed=layer_detailed, prune_conv_channels=prune_conv_channels)
        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device('cpu'), weights_only=False)['model_state_dict']
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = '.'.join(key.split('.')[1:])
                if 'frontend3D' in key:
                    frontend_std[new_key] = val
                if 'trunk' in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch*s_time, n_channels, sx, sy)
    
    def prune(self):
        out_channel_index, new_config = self.trunk.prune()
        return out_channel_index, new_config
    
    def get_num_params_and_out_channels(self):
        num_params = self.frontend_nout * 5 * 7 * 7 # frontend3D: Conv3d
        num_params += self.frontend_nout * 2 # frontend3D: BatchNorm3d
        if self.relu_type == 'prelu':
            num_params += self.frontend_nout # frontend3D: PRELU
        num_params_trunk, out_channels = self.trunk.get_num_params_and_out_channels() # ResNet
        num_params += num_params_trunk
        return num_params, out_channels
