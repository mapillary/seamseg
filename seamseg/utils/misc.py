# Copyright (c) Facebook, Inc. and its affiliates.

import io
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from inplace_abn import InPlaceABN, InPlaceABNSync, ABN

from seamseg.modules.misc import ActivatedAffine, ActivatedGroupNorm
from . import scheduler as lr_scheduler

NORM_LAYERS = [ABN, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]
OTHER_LAYERS = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]


class Empty(Exception):
    """Exception to facilitate handling of empty predictions, annotations etc."""
    pass


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list


def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str


def scheduler_from_config(scheduler_config, optimizer, epoch_length):
    assert scheduler_config["type"] in ("linear", "step", "poly", "multistep")

    params = scheduler_config.getstruct("params")

    if scheduler_config["type"] == "linear":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")

        beta = float(params["from"])
        alpha = float(params["to"] - beta) / count

        scheduler = lr_scheduler.LambdaLR(optimizer, lambda it: it * alpha + beta)
    elif scheduler_config["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer, params["step_size"], params["gamma"])
    elif scheduler_config["type"] == "poly":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda it: (1 - float(it) / count) ** params["gamma"])
    elif scheduler_config["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, params["milestones"], params["gamma"])
    else:
        raise ValueError("Unrecognized scheduler type {}, valid options: 'linear', 'step', 'poly', 'multistep'"
                         .format(scheduler_config["type"]))

    if scheduler_config.getint("burn_in_steps") != 0:
        scheduler = lr_scheduler.BurnInLR(scheduler,
                                          scheduler_config.getint("burn_in_steps"),
                                          scheduler_config.getfloat("burn_in_start"))

    return scheduler


def norm_act_from_config(body_config):
    """Make normalization + activation function from configuration

    Available normalization modes are:
      - `bn`: Standard In-Place Batch Normalization
      - `syncbn`: Synchronized In-Place Batch Normalization
      - `syncbn+bn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Standard In-Place
        Batch Normalization in the "dynamic" parts
      - `gn`: Group Normalization
      - `syncbn+gn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Group Normalization
        in the "dynamic" parts
      - `off`: No normalization (preserve scale and bias parameters)

    The "static" part of the network includes the backbone, FPN and semantic segmentation components, while the
    "dynamic" part of the network includes the RPN, detection and instance segmentation components. Note that this
    distinction is due to historical reasons and for back-compatibility with the CVPR2019 pre-trained models.

    Parameters
    ----------
    body_config
        Configuration object containing the following fields: `normalization_mode`, `activation`, `activation_slope`
        and `gn_groups`

    Returns
    -------
    norm_act_static : callable
        Function that returns norm_act modules for the static parts of the network
    norm_act_dynamic : callable
        Function that returns norm_act modules for the dynamic parts of the network
    """
    mode = body_config["normalization_mode"]
    activation = body_config["activation"]
    slope = body_config.getfloat("activation_slope")
    groups = body_config.getint("gn_groups")

    if mode == "bn":
        norm_act_static = norm_act_dynamic = partial(InPlaceABN, activation=activation, activation_param=slope)
    elif mode == "syncbn":
        norm_act_static = norm_act_dynamic = partial(InPlaceABNSync, activation=activation, activation_param=slope)
    elif mode == "syncbn+bn":
        norm_act_static = partial(InPlaceABNSync, activation=activation, activation_param=slope)
        norm_act_dynamic = partial(InPlaceABN, activation=activation, activation_param=slope)
    elif mode == "gn":
        norm_act_static = norm_act_dynamic = partial(
            ActivatedGroupNorm, num_groups=groups, activation=activation, activation_param=slope)
    elif mode == "syncbn+gn":
        norm_act_static = partial(InPlaceABNSync, activation=activation, activation_param=slope)
        norm_act_dynamic = partial(ActivatedGroupNorm, num_groups=groups, activation=activation, activation_param=slope)
    elif mode == "off":
        norm_act_static = norm_act_dynamic = partial(ActivatedAffine, activation=activation, activation_param=slope)
    else:
        raise ValueError("Unrecognized normalization_mode {}, valid options: 'bn', 'syncbn', 'syncbn+bn', 'gn', "
                         "'syncbn+gn', 'off'".format(mode))

    return norm_act_static, norm_act_dynamic


def freeze_params(module):
    """Freeze all parameters of the given module"""
    for p in module.parameters():
        p.requires_grad_(False)


def all_reduce_losses(losses):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in losses.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.cat([v.view(1) for v in values], dim=0)
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.view(())) for k, v in zip(names, values))
