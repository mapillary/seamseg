# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict

import torch.nn as nn

from seamseg.utils.sequence import pad_packed_images

NETWORK_INPUTS = ["img", "cat", "iscrowd", "bbx"]


class DetectionNet(nn.Module):
    def __init__(self,
                 body,
                 rpn_head,
                 roi_head,
                 rpn_algo,
                 detection_algo,
                 classes):
        super(DetectionNet, self).__init__()
        self.num_stuff = classes["stuff"]

        # Modules
        self.body = body
        self.rpn_head = rpn_head
        self.roi_head = roi_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.detection_algo = detection_algo

    def _prepare_inputs(self, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out = [], [], []
        for cat_i, iscrowd_i, bbx_i in zip(cat, iscrowd, bbx):
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~iscrowd_i

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
            else:
                cat_out.append(None)
                bbx_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = iscrowd_i & thing
                iscrowd_out.append(bbx_i[iscrowd_i])
            else:
                iscrowd_out.append(None)

        return cat_out, iscrowd_out, bbx_out

    def forward(self, img, cat=None, iscrowd=None, bbx=None, do_loss=False, do_prediction=True):
        # Pad the input images
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]

        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx = self._prepare_inputs(cat, iscrowd, bbx)

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
            roi_cls_loss, roi_bbx_loss = self.detection_algo.training(
                self.roi_head, x, proposals, bbx, cat, iscrowd, img_size)
        else:
            roi_cls_loss, roi_bbx_loss = None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred = self.detection_algo.inference(
                self.roi_head, x, proposals, valid_size, img_size)
        else:
            bbx_pred, cls_pred, obj_pred = None, None, None

        # Prepare outputs
        loss = OrderedDict([
            ("obj_loss", obj_loss),
            ("bbx_loss", bbx_loss),
            ("roi_cls_loss", roi_cls_loss),
            ("roi_bbx_loss", roi_bbx_loss)
        ])
        pred = OrderedDict([
            ("bbx_pred", bbx_pred),
            ("cls_pred", cls_pred),
            ("obj_pred", obj_pred)
        ])
        return loss, pred
