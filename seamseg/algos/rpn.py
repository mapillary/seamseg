from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as functional

from seamseg.modules.losses import smooth_l1
from seamseg.utils.bbx import ious, calculate_shift, mask_overlap, bbx_overlap
from seamseg.utils.misc import Empty
from seamseg.utils.nms import nms
from seamseg.utils.parallel import PackedSequence

CHUNK_SIZE = 16


class ProposalGenerator:
    """Perform NMS-based selection of proposals

    Parameters
    ----------
    nms_threshold : float
        Intersection over union threshold for the NMS
    num_pre_nms_train : int
        Number of top-scoring proposals to feed to NMS, training mode
    num_post_nms_train : int
        Number of top-scoring proposal to keep after NMS, training mode
    num_pre_nms_val : int
        Number of top-scoring proposals to feed to NMS, validation mode
    num_post_nms_val : int
        Number of top-scoring proposal to keep after NMS, validation mode
    min_size : int
        Minimum size for proposals, discard anything with a side smaller than this
    """

    def __init__(self,
                 nms_threshold=0.7,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_val=6000,
                 num_post_nms_val=300,
                 min_size=0):
        super(ProposalGenerator, self).__init__()
        self.nms_threshold = nms_threshold
        self.num_pre_nms_train = num_pre_nms_train
        self.num_post_nms_train = num_post_nms_train
        self.num_pre_nms_val = num_pre_nms_val
        self.num_post_nms_val = num_post_nms_val
        self.min_size = min_size

    def __call__(self, boxes, scores, training):
        """Perform NMS-based selection of proposals

        Parameters
        ----------
        boxes : torch.Tensor
            Tensor of bounding boxes with shape N x M
        scores : torch.Tensor
            Tensor of bounding box scores with shape N x M x 4
        training : bool
            Switch between training and validation modes

        Returns
        -------
        proposals : PackedSequence
            Sequence of N tensors of selected bounding boxes with shape M_i x 4, entries can be None
        """
        if training:
            num_pre_nms = self.num_pre_nms_train
            num_post_nms = self.num_post_nms_train
        else:
            num_pre_nms = self.num_pre_nms_val
            num_post_nms = self.num_post_nms_val

        proposals = []
        for bbx_i, obj_i in zip(boxes, scores):
            try:
                # Optional size pre-selection
                if self.min_size > 0:
                    bbx_size = bbx_i[:, 2:] - bbx_i[:, :2]
                    valid = (bbx_size[:, 0] >= self.min_size) & (bbx_size[:, 1] >= self.min_size)

                    if valid.any().item():
                        bbx_i, obj_i = bbx_i[valid], obj_i[valid]
                    else:
                        raise Empty

                # Score pre-selection
                obj_i, idx = obj_i.topk(min(obj_i.size(0), num_pre_nms))
                bbx_i = bbx_i[idx]

                # NMS
                idx = nms(bbx_i, obj_i, self.nms_threshold, num_post_nms)
                if idx.numel() == 0:
                    raise Empty
                bbx_i = bbx_i[idx]

                proposals.append(bbx_i)
            except Empty:
                proposals.append(None)

        return PackedSequence(proposals)


class AnchorMatcher:
    """Match anchors to ground truth boxes

    Parameters
    ----------
    num_samples : int
        Maximum number of training samples to keep for each image
    pos_ratio : float
        Maximum ratio of positives to keep for each image
    pos_threshold : float
        Intersection over union threshold to assign positive labels
    neg_threshold : float
        Intersection over union threshold to assign negative labels
    void_threshold : float
        Overlap threshold to exclude anchors that overlap too much with group / crowd regions
    """

    def __init__(self,
                 num_samples=256,
                 pos_ratio=.5,
                 pos_threshold=.7,
                 neg_threshold=.3,
                 void_threshold=0.):
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.void_threshold = void_threshold

    def _subsample(self, match):
        num_pos = int(self.num_samples * self.pos_ratio)
        pos_idx = torch.nonzero(match >= 0).view(-1)
        if pos_idx.numel() > num_pos:
            rand_selection = torch.randperm(pos_idx.numel(), dtype=torch.long, device=match.device)[num_pos:]
            match[pos_idx[rand_selection]] = -2
        else:
            num_pos = pos_idx.numel()

        num_neg = self.num_samples - num_pos
        neg_idx = torch.nonzero(match == -1).view(-1)
        if neg_idx.numel() > num_neg:
            rand_selection = torch.randperm(neg_idx.numel(), dtype=torch.long, device=match.device)[num_neg:]
            match[neg_idx[rand_selection]] = -2

    @staticmethod
    def _is_inside(bbx, valid_size):
        p0y, p0x, p1y, p1x = bbx[:, 0], bbx[:, 1], bbx[:, 2], bbx[:, 3]
        return (p0y >= 0) & (p0x >= 0) & (p1y <= valid_size[0]) & (p1x <= valid_size[1])

    def __call__(self, anchors, bbx, iscrowd, valid_size):
        """Match anchors to ground truth boxes

        Parameters
        ----------
        anchors : torch.Tensor
            Tensors of anchor bounding boxes with shapes M x 4
        bbx : sequence of torch.Tensor
            Sequence of N tensors of ground truth bounding boxes with shapes M_i x 4, entries can be None
        iscrowd : sequence of torch.Tensor
            Sequence of N tensors of ground truth crowd regions (shapes H_i x W_i), or ground truth crowd bounding boxes
            (shapes K_i x 4), entries can be None
        valid_size : list of tuple of int
            List of N valid image sizes in input coordinates

        Returns
        -------
        match : torch.Tensor
            Tensor of matching results with shape N x M, with the following semantic:
              - match[i, j] == -2: the j-th anchor in image i is void
              - match[i, j] == -1: the j-th anchor in image i is negative
              - match[i, j] == k, k >= 0: the j-th anchor in image i is matched to bbx[i][k]
        """
        match = []
        for bbx_i, iscrowd_i, valid_size_i in zip(bbx, iscrowd, valid_size):
            # Default labels: everything is void
            match_i = anchors.new_full((anchors.size(0),), -2, dtype=torch.long)

            try:
                # Find anchors that are entirely within the original image area
                valid = self._is_inside(anchors, valid_size_i)

                # Check overlap with crowd
                if self.void_threshold != 0 and iscrowd_i is not None:
                    if iscrowd_i.dtype == torch.uint8:
                        overlap = mask_overlap(anchors, iscrowd_i)
                    else:
                        overlap = bbx_overlap(anchors, iscrowd_i)
                        overlap, _ = overlap.max(dim=1)

                    valid = valid & (overlap < self.void_threshold)

                if not valid.any().item():
                    raise Empty

                valid_anchors = anchors[valid]

                if bbx_i is not None:
                    max_a2g_iou = bbx_i.new_zeros(valid_anchors.size(0))
                    max_a2g_idx = bbx_i.new_full((valid_anchors.size(0),), -1, dtype=torch.long)
                    max_g2a_iou = []
                    max_g2a_idx = []

                    # Calculate assignments iteratively to save memory
                    for j, bbx_i_j in enumerate(torch.split(bbx_i, CHUNK_SIZE, dim=0)):
                        iou = ious(valid_anchors, bbx_i_j)

                        # Anchor -> GT
                        iou_max, iou_idx = iou.max(dim=1)
                        replace_idx = iou_max > max_a2g_iou

                        max_a2g_idx[replace_idx] = iou_idx[replace_idx] + j * CHUNK_SIZE
                        max_a2g_iou[replace_idx] = iou_max[replace_idx]

                        # GT -> Anchor
                        max_g2a_iou_j, max_g2a_idx_j = iou.max(dim=0)
                        max_g2a_iou.append(max_g2a_iou_j)
                        max_g2a_idx.append(max_g2a_idx_j)

                        del iou

                    max_g2a_iou = torch.cat(max_g2a_iou, dim=0)
                    max_g2a_idx = torch.cat(max_g2a_idx, dim=0)

                    a2g_pos = max_a2g_iou >= self.pos_threshold
                    a2g_neg = max_a2g_iou < self.neg_threshold
                    g2a_pos = max_g2a_iou > 0

                    valid_match = valid_anchors.new_full((valid_anchors.size(0),), -2, dtype=torch.long)
                    valid_match[a2g_pos] = max_a2g_idx[a2g_pos]
                    valid_match[a2g_neg] = -1
                    valid_match[max_g2a_idx[g2a_pos]] = g2a_pos.nonzero().squeeze()
                else:
                    # No ground truth boxes for this image: everything that is not void is negative
                    valid_match = valid_anchors.new_full((valid_anchors.size(0),), -1, dtype=torch.long)

                # Subsample positives and negatives
                self._subsample(valid_match)

                match_i[valid] = valid_match
            except Empty:
                pass

            match.append(match_i)

        return torch.stack(match, dim=0)


class RPNLoss:
    """RPN loss function

    Parameters
    ----------
    sigma : float
        "bandwidth" parameter of the smooth-L1 loss used for bounding box regression
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def bbx_loss(self, bbx_logits, bbx_lbl, num_non_void):
        bbx_logits = bbx_logits.view(-1, 4)
        bbx_lbl = bbx_lbl.view(-1, 4)

        bbx_loss = smooth_l1(bbx_logits, bbx_lbl, self.sigma).sum(dim=-1).sum()
        bbx_loss *= torch.clamp(1 / num_non_void, max=1.)
        return bbx_loss

    def __call__(self, obj_logits, bbx_logits, obj_lbl, bbx_lbl):
        """RPN loss function

        Parameters
        ----------
        obj_logits : torch.Tensor
            Tensor of objectness logits with shape N x M
        bbx_logits : torch.Tensor
            Tensor of bounding box regression logits with shape N x M x 4
        obj_lbl : torch.Tensor
            Tensor of objectness labels with shape N x M
        bbx_lbl : torch.Tensor
            Tensor of bounding box regression labels with shape N x M x 4.

        Returns
        -------
        obj_loss : torch.Tensor
            Scalar tensor containing the objectness loss
        bbx_loss : torch.Tensor
            Scalar tensor containing the objectness loss
        """
        # Get contiguous view of the labels
        positives = obj_lbl == 1
        non_void = obj_lbl != -1
        num_non_void = non_void.float().sum()

        # Objectness loss
        obj_loss = functional.binary_cross_entropy_with_logits(
            obj_logits, positives.float(), non_void.float(), reduction="sum")
        obj_loss *= torch.clamp(1. / num_non_void, max=1.)

        # Bounding box regression loss
        if positives.any().item():
            bbx_logits = bbx_logits[positives.unsqueeze(-1).expand_as(bbx_logits)]
            bbx_lbl = bbx_lbl[positives.unsqueeze(-1).expand_as(bbx_lbl)]
            bbx_loss = self.bbx_loss(bbx_logits, bbx_lbl, num_non_void)
        else:
            bbx_loss = bbx_logits.sum() * 0

        return obj_loss.mean(), bbx_loss.mean()


class RPNAlgo:
    """Base class for RPN algorithms

    Parameters
    ----------
    anchor_scales : sequence of float
        Anchor scale factors, these will be multiplied by the RPN stride to determine the actual anchor sizes
    anchor_ratios : sequence of float
        Anchor aspect ratios
    """

    def __init__(self, anchor_scales, anchor_ratios):
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios

    def _base_anchors(self, stride):
        # Pre-generate per-cell anchors
        anchors = []
        center = stride / 2.
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                h = stride * scale * sqrt(ratio)
                w = stride * scale * sqrt(1. / ratio)

                anchor = (
                    center - h / 2.,
                    center - w / 2.,
                    center + h / 2.,
                    center + w / 2.
                )
                anchors.append(anchor)

        return anchors

    @staticmethod
    def _shifted_anchors(anchors, stride, height, width, dtype=torch.float32, device="cpu"):
        grid_y = torch.arange(0, stride * height, stride, dtype=dtype, device=device)
        grid_x = torch.arange(0, stride * width, stride, dtype=dtype, device=device)
        grid = torch.stack([grid_y.view(-1, 1).repeat(1, width), grid_x.view(1, -1).repeat(height, 1)], dim=-1)

        anchors = torch.tensor(anchors, dtype=dtype, device=device)
        shifted_anchors = anchors.view(1, 1, -1, 4) + grid.repeat(1, 1, 2).unsqueeze(2)
        return shifted_anchors.view(-1, 4)

    @staticmethod
    def _match_to_lbl(anchors, bbx, match):
        pos, neg = match >= 0, match == -1

        # Objectness labels from matching tensor
        obj_lbl = torch.full_like(match, -1)
        obj_lbl[neg] = 0
        obj_lbl[pos] = 1

        # Bounding box regression labels from matching tensor
        bbx_lbl = anchors.new_zeros(len(bbx), anchors.size(0), anchors.size(1))
        for i, (pos_i, bbx_i, match_i) in enumerate(zip(pos, bbx, match)):
            if pos_i.any():
                bbx_lbl[i, pos_i] = calculate_shift(anchors[pos_i], bbx_i[match_i[pos_i]])

        return obj_lbl, bbx_lbl

    def training(self, head, x, bbx, iscrowd, valid_size, training=True, do_inference=False):
        """Given input features and ground truth compute losses and, optionally, predictions

        Parameters
        ----------
        head : nn.Module
            Module that computes objectness and bounding box regression logits given the input features
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W
        bbx : sequence of torch.Tensor
            A sequence of N tensors of ground truth bounding boxes with shapes M_i x 4, entries can be None
        iscrowd : sequence of torch.Tensor
            Sequence of tensors of ground truth crowd regions (shapes H_i x W_i), or ground truth crowd bounding boxes
            (shapes K_i x 4), entries can be None
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        training : bool
            Switch between training- and test-mode inference
        do_inference : bool
            If True, also perform inference at training time

        Returns
        -------
        obj_loss : torch.Tensor
            Scalar tensor containing the objectness loss
        bbx_loss : torch.Tensor
            Scalar tensor containing the regression loss
        proposals : PackedSequence
            Sequence of tensors of proposal boxes with shape M_i x 4, entries can be None. This is None if do_inference
            is False
        """
        raise NotImplementedError()

    def inference(self, head, x, valid_size, training):
        """Given input features compute object proposals

        Parameters
        ----------
        head : nn.Module
            Module that computes objectness and bounding box regression logits given the input features
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        training : bool
            Switch between training- and test-mode inference

        Returns
        -------
        proposals : PackedSequence
            Sequence of tensors of proposal boxes with shape M_i x 4, entries can be None
        """
        raise NotImplementedError()
