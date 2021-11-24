# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as functional

from inplace_abn import ABN


class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions"""

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class Interpolate(nn.Module):
    """nn.Module wrapper to nn.functional.interpolate"""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class ActivatedAffine(ABN):
    """Drop-in replacement for ABN which performs inference-mode BN + activation"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super(ActivatedAffine, self).__init__(num_features, eps, momentum, affine, activation, activation_param)

    @staticmethod
    def _broadcast_shape(x):
        out_size = []
        for i, s in enumerate(x.size()):
            if i != 1:
                out_size.append(1)
            else:
                out_size.append(s)
        return out_size

    def forward(self, x):
        inv_var = torch.rsqrt(self.running_var + self.eps)
        if self.affine:
            alpha = self.weight * inv_var
            beta = self.bias - self.running_mean * alpha
        else:
            alpha = inv_var
            beta = - self.running_mean * alpha

        x.mul_(alpha.view(self._broadcast_shape(x)))
        x.add_(beta.view(self._broadcast_shape(x)))

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class ActivatedGroupNorm(ABN):
    """GroupNorm + activation function compatible with the ABN interface"""

    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True, activation="leaky_relu", activation_param=0.01):
        super(ActivatedGroupNorm, self).__init__(num_channels, eps, affine=affine, activation=activation,
                                                 activation_param=activation_param)
        self.num_groups = num_groups

        # Delete running mean and var since they are not used here
        delattr(self, "running_mean")
        delattr(self, "running_var")

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))
