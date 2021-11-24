# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn.functional as functional

from seamseg.modules.losses import smooth_l1
from seamseg.utils.bbx import ious, calculate_shift, bbx_overlap, mask_overlap
from seamseg.utils.misc import Empty
from seamseg.utils.nms import nms
from seamseg.utils.parallel import PackedSequence


class PredictionGenerator:
    """Perform NMS-based selection of detections

    Parameters
    ----------
    nms_threshold : float
        IoU threshold for the class-specific NMS
    score_threshold : float
        Minimum class probability for a detection to be kept
    max_predictions : int
        Maximum number of detections to keep for each image
    """

    def __init__(self,
                 nms_threshold,
                 score_threshold,
                 max_predictions):
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_predictions = max_predictions

    @staticmethod
    def _proposals_for_img(proposals, proposals_idx, roi_cls_logits, roi_bbx_logits, img_it):
        relevant = proposals_idx == img_it
        if relevant.any():
            return proposals[relevant], roi_cls_logits[relevant], roi_bbx_logits[relevant]
        else:
            return None, None, None

    def __call__(self, boxes, scores):
        """Perform NMS-based selection of detections

        Parameters
        ----------
        boxes : sequence of torch.Tensor
            Sequence of N tensors of class-specific bounding boxes with shapes M_i x C x 4, entries can be None
        scores : sequence of torch.Tensor
            Sequence of N tensors of class probabilities with shapes M_i x (C + 1), entries can be None

        Returns
        -------
        bbx_pred : PackedSequence
            A sequence of N tensors of bounding boxes with shapes S_i x 4, entries are None for images in which no
            detection can be kept according to the selection parameters
        cls_pred : PackedSequence
            A sequence of N tensors of thing class predictions with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        obj_pred : PackedSequence
            A sequence of N tensors of detection confidences with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        """
        bbx_pred, cls_pred, obj_pred = [], [], []
        for bbx_i, obj_i in zip(boxes, scores):
            try:
                if bbx_i is None or obj_i is None:
                    raise Empty

                # Do NMS separately for each class
                bbx_pred_i, cls_pred_i, obj_pred_i = [], [], []
                for cls_id, (bbx_cls_i, obj_cls_i) in enumerate(zip(torch.unbind(bbx_i, dim=1),
                                                                    torch.unbind(obj_i, dim=1)[1:])):
                    # Filter out low-scoring predictions
                    idx = obj_cls_i > self.score_threshold
                    if not idx.any().item():
                        continue
                    bbx_cls_i = bbx_cls_i[idx]
                    obj_cls_i = obj_cls_i[idx]

                    # Filter out empty predictions
                    idx = (bbx_cls_i[:, 2] > bbx_cls_i[:, 0]) & (bbx_cls_i[:, 3] > bbx_cls_i[:, 1])
                    if not idx.any().item():
                        continue
                    bbx_cls_i = bbx_cls_i[idx]
                    obj_cls_i = obj_cls_i[idx]

                    # Do NMS
                    idx = nms(bbx_cls_i.contiguous(), obj_cls_i.contiguous(), threshold=self.nms_threshold, n_max=-1)
                    if idx.numel() == 0:
                        continue
                    bbx_cls_i = bbx_cls_i[idx]
                    obj_cls_i = obj_cls_i[idx]

                    # Save remaining outputs
                    bbx_pred_i.append(bbx_cls_i)
                    cls_pred_i.append(bbx_cls_i.new_full((bbx_cls_i.size(0),), cls_id, dtype=torch.long))
                    obj_pred_i.append(obj_cls_i)

                # Compact predictions from the classes
                if len(bbx_pred_i) == 0:
                    raise Empty
                bbx_pred_i = torch.cat(bbx_pred_i, dim=0)
                cls_pred_i = torch.cat(cls_pred_i, dim=0)
                obj_pred_i = torch.cat(obj_pred_i, dim=0)

                # Do post-NMS selection (if needed)
                if bbx_pred_i.size(0) > self.max_predictions:
                    _, idx = obj_pred_i.topk(self.max_predictions)
                    bbx_pred_i = bbx_pred_i[idx]
                    cls_pred_i = cls_pred_i[idx]
                    obj_pred_i = obj_pred_i[idx]

                # Save results
                bbx_pred.append(bbx_pred_i)
                cls_pred.append(cls_pred_i)
                obj_pred.append(obj_pred_i)
            except Empty:
                bbx_pred.append(None)
                cls_pred.append(None)
                obj_pred.append(None)

        return PackedSequence(bbx_pred), PackedSequence(cls_pred), PackedSequence(obj_pred)


class ProposalMatcher:
    """Match proposals to ground truth boxes

    Parameters
    ----------
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    num_samples : int
        Maximum number of non-void proposals to keep for each image
    pos_ratio : float
        Fraction of `num_samples` reserved for positive proposals
    pos_threshold : float
        Minimum IoU threshold to mark a proposal as positive
    neg_threshold_hi : float
        Maximum IoU threshold to mark a proposal as negative / background
    neg_threshold_lo : float
        Minimum IoU threshold to mark a proposal as negative / background
    void_threshold : float
        If not zero, minimum overlap threshold with void regions to mark a proposal as void
    """

    def __init__(self,
                 classes,
                 num_samples=128,
                 pos_ratio=0.25,
                 pos_threshold=0.5,
                 neg_threshold_hi=0.5,
                 neg_threshold_lo=0.0,
                 void_threshold=0.):
        self.num_stuff = classes["stuff"]
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio
        self.pos_threshold = pos_threshold
        self.neg_threshold_hi = neg_threshold_hi
        self.neg_threshold_lo = neg_threshold_lo
        self.void_threshold = void_threshold

    def _subsample(self, pos_idx, neg_idx):
        num_pos = int(self.num_samples * self.pos_ratio)
        pos_idx = torch.nonzero(pos_idx).view(-1)
        if pos_idx.numel() > 0:
            rand_selection = np.random.permutation(pos_idx.numel()).astype(np.int64)
            rand_selection = torch.from_numpy(rand_selection).to(pos_idx.device)

            num_pos = min(num_pos, pos_idx.numel())
            pos_idx = pos_idx[rand_selection[:num_pos]]
        else:
            num_pos = 0
            pos_idx = torch.tensor((), dtype=torch.long, device=pos_idx.device)

        num_neg = self.num_samples - num_pos
        neg_idx = torch.nonzero(neg_idx).view(-1)
        if neg_idx.numel() > 0:
            rand_selection = np.random.permutation(neg_idx.numel()).astype(np.int64)
            rand_selection = torch.from_numpy(rand_selection).to(neg_idx.device)

            num_neg = min(num_neg, neg_idx.numel())
            neg_idx = neg_idx[rand_selection[:num_neg]]
        else:
            neg_idx = torch.tensor((), dtype=torch.long, device=neg_idx.device)

        return pos_idx, neg_idx

    def __call__(self,
                 proposals,
                 bbx,
                 cat,
                 iscrowd):
        """Match proposals to ground truth boxes

        Parameters
        ----------
        proposals : PackedSequence
            A sequence of N tensors with shapes P_i x 4 containing bounding box proposals, entries can be None
        bbx : sequence of torch.Tensor
            A sequence of N tensors with shapes K_i x 4 containing ground truth bounding boxes, entries can be None
        cat : sequence of torch.Tensor
            A sequence of N tensors with shapes K_i containing ground truth instance -> category mappings, entries can
            be None
        iscrowd : sequence of torch.Tensor
            Sequence of N tensors of ground truth crowd regions (shapes H_i x W_i), or ground truth crowd bounding boxes
            (shapes K_i x 4), entries can be None

        Returns
        -------
        out_proposals : PackedSequence
            A sequence of N tensors with shapes S_i x 4 containing the non-void bounding box proposals, entries are None
            for images that do not contain any non-void proposal
        match : PackedSequence
            A sequence of matching results with shape S_i, with the following semantic:
              - match[i, j] == -1: the j-th anchor in image i is negative
              - match[i, j] == k, k >= 0: the j-th anchor in image i is matched to bbx[i][k]
        """
        out_proposals = []
        match = []

        for proposals_i, bbx_i, cat_i, iscrowd_i in zip(proposals, bbx, cat, iscrowd):
            try:
                # Append proposals to ground truth bounding boxes before proceeding
                if bbx_i is not None and proposals_i is not None:
                    proposals_i = torch.cat([bbx_i, proposals_i], dim=0)
                elif bbx_i is not None:
                    proposals_i = bbx_i
                else:
                    raise Empty

                # Optionally check overlap with void
                if self.void_threshold != 0 and iscrowd_i is not None:
                    if iscrowd_i.dtype == torch.uint8:
                        overlap = mask_overlap(proposals_i, iscrowd_i)
                    else:
                        overlap = bbx_overlap(proposals_i, iscrowd_i)
                        overlap, _ = overlap.max(dim=1)

                    valid = overlap < self.void_threshold
                    proposals_i = proposals_i[valid]

                if proposals_i.size(0) == 0:
                    raise Empty

                # Find positives and negatives based on IoU
                if bbx_i is not None:
                    iou = ious(proposals_i, bbx_i)
                    best_iou, best_gt = iou.max(dim=1)

                    pos_idx = best_iou >= self.pos_threshold
                    neg_idx = (best_iou >= self.neg_threshold_lo) & (best_iou < self.neg_threshold_hi)
                else:
                    # No ground truth boxes: all proposals that are non-void are negative
                    pos_idx = proposals_i.new_zeros(proposals_i.size(0), dtype=torch.uint8)
                    neg_idx = proposals_i.new_ones(proposals_i.size(0), dtype=torch.uint8)

                # Check that there are still some non-voids and do sub-sampling
                if not pos_idx.any().item() and not neg_idx.any().item():
                    raise Empty
                pos_idx, neg_idx = self._subsample(pos_idx, neg_idx)

                # Gather selected proposals
                out_proposals_i = proposals_i[torch.cat([pos_idx, neg_idx])]

                # Save matching
                match_i = out_proposals_i.new_full((out_proposals_i.size(0),), -1, dtype=torch.long)
                match_i[:pos_idx.numel()] = best_gt[pos_idx]

                # Save to output
                out_proposals.append(out_proposals_i)
                match.append(match_i)
            except Empty:
                out_proposals.append(None)
                match.append(None)

        return PackedSequence(out_proposals), PackedSequence(match)


class DetectionLoss:
    """Detection loss"""

    def __init__(self, sigma):
        self.sigma = sigma

    def bbx_loss(self, bbx_logits, bbx_lbl, num_non_void):
        bbx_logits = bbx_logits.view(-1, 4)
        bbx_lbl = bbx_lbl.view(-1, 4)

        bbx_loss = smooth_l1(bbx_logits, bbx_lbl, self.sigma).sum(dim=-1).sum()
        bbx_loss /= num_non_void
        return bbx_loss

    def __call__(self, cls_logits, bbx_logits, cls_lbl, bbx_lbl):
        """Detection loss

        Parameters
        ----------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape M x (C + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape M x C x 4
        cls_lbl : PackedSequence
            A sequence of N tensors of classification labels with shapes M_i, entries can be None
        bbx_lbl : PackedSequence
            A sequence of N tensors of bounding box regression labels with shapes P_i x 4, entries can be None.

        Returns
        -------
        cls_loss : torch.Tensor
            A scalar tensor with the classification loss
        bbx_loss : torch.Tensor
            A scalar tensor with the bounding box regression loss
        """
        # Get contiguous view of the labels
        cls_lbl, _ = cls_lbl.contiguous
        bbx_lbl, _ = bbx_lbl.contiguous

        # Classification loss
        cls_loss = functional.cross_entropy(cls_logits, cls_lbl)

        # Regression loss
        positives = cls_lbl > 0
        num_non_void = cls_lbl.numel()
        if positives.any().item():
            cls_lbl = cls_lbl[positives]
            bbx_logits = bbx_logits[positives]

            idx = torch.arange(0, bbx_logits.size(0), dtype=torch.long, device=bbx_logits.device)
            bbx_loss = self.bbx_loss(bbx_logits[idx, cls_lbl - 1], bbx_lbl, num_non_void)
        else:
            bbx_loss = bbx_logits.sum() * 0

        return cls_loss, bbx_loss


class DetectionAlgo:
    """Base class for detection algorithms

    Parameters
    ----------
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    bbx_reg_weights : sequence of float
        Weights assigned to the bbx regression coordinates
    """

    def __init__(self, classes, bbx_reg_weights):
        self.num_stuff = classes["stuff"]
        self.bbx_reg_weights = bbx_reg_weights

    @staticmethod
    def _split_and_clip(boxes, scores, index, valid_size):
        boxes_out, scores_out = [], []
        for img_id, valid_size_i in enumerate(valid_size):
            idx = index == img_id
            if idx.any().item():
                boxes_i = boxes[idx]
                boxes_i[:, :, [0, 2]] = torch.clamp(boxes_i[:, :, [0, 2]], min=0, max=valid_size_i[0])
                boxes_i[:, :, [1, 3]] = torch.clamp(boxes_i[:, :, [1, 3]], min=0, max=valid_size_i[1])

                boxes_out.append(boxes_i)
                scores_out.append(scores[idx])
            else:
                boxes_out.append(None)
                scores_out.append(None)

        return boxes_out, scores_out

    def _match_to_lbl(self, proposals, bbx, cat, match):
        cls_lbl = []
        bbx_lbl = []
        for i, (proposals_i, bbx_i, cat_i, match_i) in enumerate(zip(proposals, bbx, cat, match)):
            if match_i is not None:
                pos = match_i >= 0

                # Objectness labels
                cls_lbl_i = proposals_i.new_zeros(proposals_i.size(0), dtype=torch.long)
                cls_lbl_i[pos] = cat_i[match_i[pos]] + 1 - self.num_stuff

                # Bounding box regression labels
                if pos.any().item():
                    bbx_lbl_i = calculate_shift(proposals_i[pos], bbx_i[match_i[pos]])
                    bbx_lbl_i *= bbx_lbl_i.new(self.bbx_reg_weights)
                else:
                    bbx_lbl_i = None

                cls_lbl.append(cls_lbl_i)
                bbx_lbl.append(bbx_lbl_i)
            else:
                cls_lbl.append(None)
                bbx_lbl.append(None)

        return PackedSequence(cls_lbl), PackedSequence(bbx_lbl)

    def training(self, head, x, proposals, bbx, cat, iscrowd, img_size):
        """Given input features, proposals and ground truth compute detection losses

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute classification and bounding box regression logits given an input feature map and a set
            of proposal bounding boxes. Must be callable as `head(x, proposals, proposals_idx, img_size)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        proposals : PackedSequence
            A sequence of N tensors of bounding box proposals with shapes P_i x 4, entries can be None
        bbx : sequence of torch.Tensor
            A sequence of N tensors of ground truth bounding boxes with shapes M_i x 4, entries can be None
        cat : sequence of torch.Tensor
            A sequence of N tensors of ground truth instance -> category mappings with shapes K_i, entries can
            be None
        iscrowd : sequence of torch.Tensor
            A sequence of N tensors of ground truth crowd bounding boxes with shapes M_i x 4, entries can be None
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        cls_loss : torch.Tensor
            A scalar tensor with the classification loss
        bbx_loss : torch.Tensor
            A scalar tensor with the bounding box regression loss
        """
        raise NotImplementedError()

    def inference(self, head, x, proposals, valid_size, img_size):
        """Given input features compute detection predictions

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute classification and bounding box regression logits given an input feature map and a set
            of proposal bounding boxes. Must be callable as `head(x, proposals, proposals_idx, img_size)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        proposals : PackedSequence
            A sequence of N tensors of bounding box proposals with shapes P_i x 4, entries can be None
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        bbx_preds : PackedSequence
            A sequence of N tensors of bounding boxes with shapes S_i x 4, entries are None for images in which no
            detection can be kept according to the selection parameters
        cls_preds : PackedSequence
            A sequence of N tensors of thing class predictions with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        obj_preds : PackedSequence
            A sequence of N tensors of detection confidences with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        """
        raise NotImplementedError()
