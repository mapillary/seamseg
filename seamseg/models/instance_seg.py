# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict

import torch
import torch.nn as nn

from seamseg.utils.sequence import pad_packed_images

NETWORK_INPUTS = ["img", "msk", "cat", "iscrowd", "bbx"]


class InstanceSegNet(nn.Module):
    def __init__(self,
                 body,
                 rpn_head,
                 roi_head,
                 rpn_algo,
                 instance_seg_algo,
                 classes):
        super(InstanceSegNet, self).__init__()
        self.num_stuff = classes["stuff"]

        # Modules
        self.body = body
        self.rpn_head = rpn_head
        self.roi_head = roi_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.instance_seg_algo = instance_seg_algo

    def _prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out = [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~iscrowd_i

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = iscrowd_i & thing
                iscrowd_out.append(iscrowd_i[msk_i].any(dim=0))
            else:
                iscrowd_out.append(None)

        return cat_out, iscrowd_out, bbx_out, ids_out

    def forward(self, img, msk=None, cat=None, iscrowd=None, bbx=None, do_loss=False, do_prediction=True):
        # Pad the input images
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]

        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx, ids = self._prepare_inputs(msk, cat, iscrowd, bbx)

        # Run network body
        x = self.body(img)

        # RPN part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(
                self.rpn_head, x, bbx, iscrowd, valid_size, training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, x, valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = self.instance_seg_algo.training(
                self.roi_head, x, proposals, bbx, cat, iscrowd, ids, msk, img_size)
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred = self.instance_seg_algo.inference(
                self.roi_head, x, proposals, valid_size, img_size)
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Prepare outputs
        loss = OrderedDict([
            ("obj_loss", obj_loss),
            ("bbx_loss", bbx_loss),
            ("roi_cls_loss", roi_cls_loss),
            ("roi_bbx_loss", roi_bbx_loss),
            ("roi_msk_loss", roi_msk_loss)
        ])
        pred = OrderedDict([
            ("bbx_pred", bbx_pred),
            ("cls_pred", cls_pred),
            ("obj_pred", obj_pred),
            ("msk_pred", msk_pred)
        ])
        return loss, pred
