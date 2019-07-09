from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN


class FPN(nn.Module):
    """Feature Pyramid Network module

    Parameters
    ----------
    in_channels : sequence of int
        Number of feature channels in each of the input feature levels
    out_channels : int
        Number of output feature channels (same for each level)
    extra_scales : int
        Number of extra low-resolution scales
    norm_act : callable
        Function to create normalization + activation modules
    interpolation : str
        Interpolation mode to use when up-sampling, see `torch.nn.functional.interpolate`
    """

    def __init__(self, in_channels, out_channels=256, extra_scales=0, norm_act=ABN, interpolation="nearest"):
        super(FPN, self).__init__()
        self.interpolation = interpolation

        # Lateral connections and output convolutions
        self.lateral = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])
        self.output = nn.ModuleList([
            self._make_output(out_channels, norm_act) for _ in in_channels
        ])

        if extra_scales > 0:
            self.extra = nn.ModuleList([
                self._make_extra(in_channels[-1] if i == 0 else out_channels, out_channels, norm_act)
                for i in range(extra_scales)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.lateral[0].bn.activation, self.lateral[0].bn.activation_param)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    @staticmethod
    def _make_lateral(input_channels, hidden_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 1, bias=False)),
            ("bn", norm_act(hidden_channels))
        ]))

    @staticmethod
    def _make_output(channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ("bn", norm_act(channels))
        ]))

    @staticmethod
    def _make_extra(input_channels, out_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, out_channels, 3, stride=2, padding=1, bias=False)),
            ("bn", norm_act(out_channels))
        ]))

    def forward(self, xs):
        """Feature Pyramid Network module

        Parameters
        ----------
        xs : sequence of torch.Tensor
            The input feature maps, tensors with shapes N x C_i x H_i x W_i

        Returns
        -------
        ys : sequence of torch.Tensor
            The output feature maps, tensors with shapes N x K x H_i x W_i
        """
        ys = []
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        # Build pyramid
        for x_i, lateral_i in zip(xs[::-1], self.lateral[::-1]):
            x_i = lateral_i(x_i)
            if len(ys) > 0:
                x_i = x_i + functional.interpolate(ys[0], size=x_i.shape[-2:], **interp_params)
            ys.insert(0, x_i)

        # Compute outputs
        ys = [output_i(y_i) for y_i, output_i in zip(ys, self.output)]

        # Compute extra outputs if necessary
        if hasattr(self, "extra"):
            y = xs[-1]
            for extra_i in self.extra:
                y = extra_i(y)
                ys.append(y)

        return ys


class FPNBody(nn.Module):
    """Wrapper for a backbone network and an FPN module

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone network, which takes a batch of images and produces a dictionary of intermediate features
    fpn : torch.nn.Module
        FPN module, which takes a list of intermediate features and produces a list of outputs
    fpn_inputs : iterable
        An iterable producing the names of the intermediate features to take from the backbone's output and pass
        to the FPN
    """

    def __init__(self, backbone, fpn, fpn_inputs=()):
        super(FPNBody, self).__init__()
        self.fpn_inputs = fpn_inputs

        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x):
        x = self.backbone(x)
        xs = [x[fpn_input] for fpn_input in self.fpn_inputs]
        return self.fpn(xs)
