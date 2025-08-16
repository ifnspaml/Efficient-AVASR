"""Utility functions for pruning."""
from typing import Union

import torch
import torch.nn as nn


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: str):
    "Prune linear layer in place."
    # NOTE: weight: (out_features, in_features), bias: (out_features,)
    if dim == "input":
        dim = 1
        layer.in_features = len(index)
    elif dim == "output":
        dim = 0
        layer.out_features = len(index)
    else:
        raise ValueError

    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_conv2d_layer(layer: nn.Conv2d, index: torch.LongTensor, dim: str):
    """Prune conv2d in place."""
    # NOTE: weight: (out_channels, in_channels, kernel_size), bias: (out_channels,)
    if dim == "input":
        dim = 1
        layer.in_channels = len(index)
    elif dim == "output":
        dim = 0
        layer.out_channels = len(index)
    else:
        raise ValueError
    
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_layer_norm(layernorm: Union[nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d], index: torch.LongTensor):
    """Prune layer norm, group norm or batch norm 2d in place."""
    layernorm.weight = nn.Parameter(layernorm.weight.index_select(0, index).clone().detach())
    layernorm.bias = nn.Parameter(layernorm.bias.index_select(0, index).clone().detach())
    if isinstance(layernorm, nn.LayerNorm):
        layernorm.normalized_shape = (len(index),)
    elif isinstance(layernorm, nn.GroupNorm):
        layernorm.num_groups = len(index)
        layernorm.num_channels = len(index)
    elif isinstance(layernorm, nn.BatchNorm2d):
        layernorm.num_features = len(index)
        layernorm.running_mean = layernorm.running_mean.index_select(0, index).clone()
        layernorm.running_var = layernorm.running_var.index_select(0, index).clone()


def prune_prelu(prelu: nn.PReLU, index: torch.LongTensor):
    prelu.num_parameters = len(index)
    prelu.weight = nn.Parameter(prelu.weight.index_select(0, index).clone().detach())
