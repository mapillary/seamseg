# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from seamseg.utils.parallel import PackedSequence


def iss_collate_fn(items):
    """Collate function for ISS batches"""
    out = {}
    if len(items) > 0:
        for key in items[0]:
            out[key] = [item[key] for item in items]
            if isinstance(items[0][key], torch.Tensor):
                out[key] = PackedSequence(out[key])
    return out
