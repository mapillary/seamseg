# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN

from seamseg.utils.misc import try_index


class FPNROIHead(nn.Module):
    """ROI head module for FPN

    Parameters
    ----------
    in_channels : int
        Number of input channels
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    roi_size : tuple of int
        `(height, width)` of the ROIs extracted from the input feature map, these will be average-pooled 2x2 before
        feeding to the rest of the head
    hidden_channels : int
        Number of channels in the hidden layers
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, classes, roi_size, hidden_channels=1024, norm_act=ABN):
        super(FPNROIHead, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), hidden_channels, bias=False)),
            ("bn1", norm_act(hidden_channels)),
            ("fc2", nn.Linear(hidden_channels, hidden_channels, bias=False)),
            ("bn2", norm_act(hidden_channels))
        ]))
        self.roi_cls = nn.Linear(hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(hidden_channels, classes["thing"] * 4)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.fc.bn1.activation, self.fc.bn1.activation_param)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "roi_cls" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x):
        """ROI head module for FPN

        Parameters
        ----------
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W

        Returns
        -------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape S x (num_thing + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape S x num_thing x 4
        """
        x = functional.avg_pool2d(x, 2)

        # Run head
        x = self.fc(x.view(x.size(0), -1))
        return self.roi_cls(x), self.roi_bbx(x).view(x.size(0), -1, 4)


class FPNMaskHead(nn.Module):
    """ROI head module for FPN

    Parameters
    ----------
    in_channels : int
        Number of input channels
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    roi_size : tuple of int
        `(height, width)` of the ROIs extracted from the input feature map, these will be average-pooled 2x2 before
        feeding to the fully-connected branch
    fc_hidden_channels : int
        Number of channels in the hidden layers of the fully-connected branch
    conv_hidden_channels : int
        Number of channels in the hidden layers of the convolutional branch
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, classes, roi_size, fc_hidden_channels=1024, conv_hidden_channels=256, norm_act=ABN):
        super(FPNMaskHead, self).__init__()

        # ROI section
        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), fc_hidden_channels, bias=False)),
            ("bn1", norm_act(fc_hidden_channels)),
            ("fc2", nn.Linear(fc_hidden_channels, fc_hidden_channels, bias=False)),
            ("bn2", norm_act(fc_hidden_channels))
        ]))
        self.roi_cls = nn.Linear(fc_hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(fc_hidden_channels, classes["thing"] * 4)

        # Mask section
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn1", norm_act(conv_hidden_channels)),
            ("conv2", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn2", norm_act(conv_hidden_channels)),
            ("conv3", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn3", norm_act(conv_hidden_channels)),
            ("conv4", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn4", norm_act(conv_hidden_channels)),
            ("conv_up", nn.ConvTranspose2d(conv_hidden_channels, conv_hidden_channels, 2, stride=2, bias=False)),
            ("bn_up", norm_act(conv_hidden_channels))
        ]))
        self.roi_msk = nn.Conv2d(conv_hidden_channels, classes["thing"], 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.fc.bn1.activation, self.fc.bn1.activation_param)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
                if "roi_cls" in name or "roi_msk" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x, do_cls_bbx=True, do_msk=True):
        """ROI head module for FPN

        Parameters
        ----------
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W
        do_cls_bbx : bool
            Whether to compute or not the class and bounding box regression predictions
        do_msk : bool
            Whether to compute or not the mask predictions

        Returns
        -------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape S x (num_thing + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape S x num_thing x 4
        msk_logits : torch.Tensor
            A tensor of class-specific mask logits with shape S x num_thing x (H_roi * 2) x (W_roi * 2)
        """
        # Run fully-connected head
        if do_cls_bbx:
            x_fc = functional.avg_pool2d(x, 2)
            x_fc = self.fc(x_fc.view(x_fc.size(0), -1))

            cls_logits = self.roi_cls(x_fc)
            bbx_logits = self.roi_bbx(x_fc).view(x_fc.size(0), -1, 4)
        else:
            cls_logits = None
            bbx_logits = None

        # Run convolutional head
        if do_msk:
            x = self.conv(x)
            msk_logits = self.roi_msk(x)
        else:
            msk_logits = None

        return cls_logits, bbx_logits, msk_logits


class FPNSemanticHeadDeeplab(nn.Module):
    """Semantic segmentation head for FPN-style networks, extending Deeplab v3 for FPN bodies"""

    class _MiniDL(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, pooling_size, norm_act):
            super(FPNSemanticHeadDeeplab._MiniDL, self).__init__()
            self.pooling_size = pooling_size

            self.conv1_3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.conv1_dil = nn.Conv2d(in_channels, out_channels, 3, dilation=dilation, padding=dilation, bias=False)
            self.conv1_glb = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn1 = norm_act(out_channels * 3)

            self.conv2 = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
            self.bn2 = norm_act(out_channels)

        def _global_pooling(self, x):
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
            return pool

        def forward(self, x):
            x = torch.cat([
                self.conv1_3x3(x),
                self.conv1_dil(x),
                self.conv1_glb(self._global_pooling(x)),
            ], dim=1)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def __init__(self,
                 in_channels,
                 min_level,
                 levels,
                 num_classes,
                 hidden_channels=128,
                 dilation=6,
                 pooling_size=(64, 64),
                 norm_act=ABN,
                 interpolation="bilinear"):
        super(FPNSemanticHeadDeeplab, self).__init__()
        self.min_level = min_level
        self.levels = levels
        self.interpolation = interpolation

        self.output = nn.ModuleList([
            self._MiniDL(in_channels, hidden_channels, dilation, pooling_size, norm_act) for _ in range(levels)
        ])
        self.conv_sem = nn.Conv2d(hidden_channels * levels, num_classes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.output[0].bn1.activation, self.output[0].bn1.activation_param)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Conv2d):
                if "conv_sem" not in name:
                    nn.init.xavier_normal_(mod.weight, gain)
                else:
                    nn.init.xavier_normal_(mod.weight, .1)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, xs):
        xs = xs[self.min_level:self.min_level + self.levels]

        ref_size = xs[0].shape[-2:]
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        for i, output in enumerate(self.output):
            xs[i] = output(xs[i])
            if i > 0:
                xs[i] = functional.interpolate(xs[i], size=ref_size, **interp_params)

        xs = torch.cat(xs, dim=1)
        xs = self.conv_sem(xs)

        return xs
