# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from .bbx import invert_roi_bbx
from .misc import Empty
from .roi_sampling import roi_sampling


class PanopticPreprocessing:
    def __init__(self,
                 score_threshold=0.5,
                 overlap_threshold=0.5,
                 min_stuff_area=64 * 64):
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        self.min_stuff_area = min_stuff_area

    def __call__(self, sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, num_stuff):
        img_size = [sem_pred.size(0), sem_pred.size(1)]

        # Initialize outputs
        occupied = torch.zeros_like(sem_pred, dtype=torch.uint8)
        msk = torch.zeros_like(sem_pred)
        cat = [255]
        obj = [0]
        iscrowd = [0]

        # Process thing
        try:
            if bbx_pred is None or cls_pred is None or obj_pred is None or msk_pred is None:
                raise Empty

            # Remove low-confidence instances
            keep = obj_pred > self.score_threshold
            if not keep.any():
                raise Empty
            obj_pred, bbx_pred, cls_pred, msk_pred = obj_pred[keep], bbx_pred[keep], cls_pred[keep], msk_pred[keep]

            # Up-sample masks
            bbx_inv = invert_roi_bbx(bbx_pred, list(msk_pred.shape[-2:]), img_size)
            bbx_idx = torch.arange(0, msk_pred.size(0), dtype=torch.long, device=msk_pred.device)
            msk_pred = roi_sampling(msk_pred.unsqueeze(1).sigmoid(), bbx_inv, bbx_idx, tuple(img_size), padding="zero")
            msk_pred = msk_pred.squeeze(1) > 0.5

            # Sort by score
            idx = torch.argsort(obj_pred, descending=True)

            # Process instances
            for msk_i, cls_i, obj_i in zip(msk_pred[idx], cls_pred[idx], obj_pred[idx]):
                # Check intersection
                intersection = occupied & msk_i
                if intersection.float().sum() / msk_i.float().sum() > self.overlap_threshold:
                    continue

                # Add non-intersecting part to output
                msk_i = msk_i - intersection
                msk[msk_i] = len(cat)
                cat.append(cls_i.item() + num_stuff)
                obj.append(obj_i.item())
                iscrowd.append(0)

                # Update occupancy mask
                occupied += msk_i
        except Empty:
            pass

        # Process stuff
        for cls_i in range(sem_pred.max().item() + 1):
            msk_i = sem_pred == cls_i

            # Remove occupied part and check remaining area
            msk_i = msk_i & ~occupied
            if msk_i.float().sum() < self.min_stuff_area:
                continue

            # Add non-intersecting part to output
            msk[msk_i] = len(cat)
            cat.append(cls_i)
            obj.append(1)
            iscrowd.append(cls_i >= num_stuff)

            # Update occupancy mask
            occupied += msk_i

        # Wrap in tensors
        cat = torch.tensor(cat, dtype=torch.long)
        obj = torch.tensor(obj, dtype=torch.float)
        iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)

        return msk.cpu(), cat, obj, iscrowd


def panoptic_stats(msk_gt, cat_gt, panoptic_pred, num_classes, _num_stuff):
    # Move gt to CPU
    msk_gt, cat_gt = msk_gt.cpu(), cat_gt.cpu()
    msk_pred, cat_pred, _, iscrowd_pred = panoptic_pred

    # Convert crowd predictions to void
    msk_remap = msk_pred.new_zeros(cat_pred.numel())
    msk_remap[~iscrowd_pred] = torch.arange(
        0, (~iscrowd_pred).long().sum().item(), dtype=msk_remap.dtype, device=msk_remap.device)
    msk_pred = msk_remap[msk_pred]
    cat_pred = cat_pred[~iscrowd_pred]

    iou = msk_pred.new_zeros(num_classes, dtype=torch.double)
    tp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fn = msk_pred.new_zeros(num_classes, dtype=torch.double)

    if cat_gt.numel() > 1:
        msk_gt = msk_gt.view(-1)
        msk_pred = msk_pred.view(-1)

        # Compute confusion matrix
        confmat = msk_pred.new_zeros(cat_gt.numel(), cat_pred.numel(), dtype=torch.double)
        confmat.view(-1).index_add_(0, msk_gt * cat_pred.numel() + msk_pred,
                                    confmat.new_ones(msk_gt.numel()))

        # track potentially valid FP, i.e. those that overlap with void_gt <= 0.5
        num_pred_pixels = confmat.sum(0)
        valid_fp = (confmat[0] / num_pred_pixels) <= 0.5

        # compute IoU without counting void pixels (both in gt and pred)
        _iou = confmat / ((num_pred_pixels - confmat[0]).unsqueeze(0) + confmat.sum(1).unsqueeze(1) - confmat)

        # flag TP matches, i.e. same class and iou > 0.5
        matches = ((cat_gt.unsqueeze(1) == cat_pred.unsqueeze(0)) & (_iou > 0.5))

        # remove potential match of void_gt against void_pred
        matches[0, 0] = 0

        _iou = _iou[matches]
        tp_i, _ = matches.max(1)
        fn_i = ~tp_i
        fn_i[0] = 0  # remove potential fn match due to void against void
        fp_i = ~matches.max(0)[0] & valid_fp
        fp_i[0] = 0  # remove potential fp match due to void against void

        # Compute per instance classes for each tp, fp, fn
        tp_cat = cat_gt[tp_i]
        fn_cat = cat_gt[fn_i]
        fp_cat = cat_pred[fp_i]

        # Accumulate per class counts
        if tp_cat.numel() > 0:
            tp.index_add_(0, tp_cat, tp.new_ones(tp_cat.numel()))
        if fp_cat.numel() > 0:
            fp.index_add_(0, fp_cat, fp.new_ones(fp_cat.numel()))
        if fn_cat.numel() > 0:
            fn.index_add_(0, fn_cat, fn.new_ones(fn_cat.numel()))
        if tp_cat.numel() > 0:
            iou.index_add_(0, tp_cat, _iou)

    # note else branch is not needed because if cat_gt has only void we don't penalize predictions
    return iou, tp, fp, fn
