# Copyright (c) Facebook, Inc. and its affiliates.

import argparse

import torch
import torch.utils.model_zoo as model_zoo

MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

NETS = {
    "resnet18": {"structure": [2, 2, 2, 2], "bottleneck": False},
    "resnet34": {"structure": [3, 4, 6, 3], "bottleneck": False},
    "resnet50": {"structure": [3, 4, 6, 3], "bottleneck": True},
    "resnet101": {"structure": [3, 4, 23, 3], "bottleneck": True},
    "resnet152": {"structure": [3, 8, 36, 3], "bottleneck": True},
}

CONV_PARAMS = ["weight"]
BN_PARAMS = ["weight", "bias", "running_mean", "running_var"]

parser = argparse.ArgumentParser(description="Convert pre-trained ResNet from Pytorch to our format")
parser.add_argument("net", metavar="NET", type=str, choices=list(MODEL_URLS.keys()), help="Network name")
parser.add_argument("out_file", metavar="OUT", type=str, help="Output file")


def copy_layer(inm, outm, name_in, name_out, params):
    for param_name in params:
        outm[name_out + "." + param_name] = inm[name_in + "." + param_name]


def convert(model, structure, bottleneck):
    out = dict()
    num_convs = 3 if bottleneck else 2

    # Initial module
    copy_layer(model, out, "conv1", "mod1.conv1", CONV_PARAMS)
    copy_layer(model, out, "bn1", "mod1.bn1", BN_PARAMS)

    # Other modules
    for mod_id, num in enumerate(structure):
        for block_id in range(num):
            for conv_id in range(num_convs):
                copy_layer(model, out,
                           "layer{}.{}.conv{}".format(mod_id + 1, block_id, conv_id + 1),
                           "mod{}.block{}.convs.conv{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                           CONV_PARAMS)
                copy_layer(model, out,
                           "layer{}.{}.bn{}".format(mod_id + 1, block_id, conv_id + 1),
                           "mod{}.block{}.convs.bn{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                           BN_PARAMS)

            # Try copying projection module
            try:
                copy_layer(model, out,
                           "layer{}.{}.downsample.0".format(mod_id + 1, block_id),
                           "mod{}.block{}.proj_conv".format(mod_id + 2, block_id + 1),
                           CONV_PARAMS)
                copy_layer(model, out,
                           "layer{}.{}.downsample.1".format(mod_id + 1, block_id),
                           "mod{}.block{}.proj_bn".format(mod_id + 2, block_id + 1),
                           BN_PARAMS)
            except KeyError:
                pass

    return out


if __name__ == '__main__':
    args = parser.parse_args()

    original = model_zoo.load_url(MODEL_URLS[args.net])
    converted = convert(original, **NETS[args.net])
    torch.save(converted, args.out_file)
