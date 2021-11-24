# Copyright (c) Facebook, Inc. and its affiliates.

import json
import tempfile
import time
from collections import defaultdict
from os import path, remove

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode as mask_encode

from .bbx import invert_roi_bbx, extract_boxes
from .parallel import PackedSequence
from .roi_sampling import roi_sampling


def process_prediction(bbx_pred, cls_pred, obj_pred, msk_pred, img_size, idx, original_size):
    # Move everything to CPU
    bbx_pred, cls_pred, obj_pred = (t.cpu() for t in (bbx_pred, cls_pred, obj_pred))
    msk_pred = msk_pred.cpu() if msk_pred is not None else None

    if msk_pred is not None:
        if isinstance(msk_pred, torch.Tensor):
            # ROI-stile prediction
            bbx_inv = invert_roi_bbx(bbx_pred, list(msk_pred.shape[-2:]), list(img_size))
            bbx_idx = torch.arange(0, msk_pred.size(0), dtype=torch.long)
            msk_pred = roi_sampling(msk_pred.unsqueeze(1).sigmoid(), bbx_inv, bbx_idx, list(img_size), padding="zero")
            msk_pred = msk_pred.squeeze(1) > 0.5
        elif isinstance(msk_pred, PackedSequence):
            # Seeds-style prediction
            msk_pred.data = msk_pred.data > 0.5
            msk_pred_exp = msk_pred.data.new_zeros(len(msk_pred), img_size[0], img_size[1])

            for it, (msk_pred_i, bbx_pred_i) in enumerate(zip(msk_pred, bbx_pred)):
                i, j = int(bbx_pred_i[0].item()), int(bbx_pred_i[1].item())
                msk_pred_exp[it, i:i + msk_pred_i.size(0), j:j + msk_pred_i.size(1)] = msk_pred_i

            msk_pred = msk_pred_exp

    # Convert bbx and redo clamping
    bbx_pred[:, [0, 2]] = (bbx_pred[:, [0, 2]] / img_size[0] * original_size[0]).clamp(min=0, max=original_size[0])
    bbx_pred[:, [1, 3]] = (bbx_pred[:, [1, 3]] / img_size[1] * original_size[1]).clamp(min=0, max=original_size[1])
    bbx_pred_size = bbx_pred[:, 2:] - bbx_pred[:, :2]

    outs = []
    for i, (bbx_pred_i, bbx_pred_size_i, cls_pred_i, obj_pred_i) in \
            enumerate(zip(bbx_pred, bbx_pred_size, cls_pred, obj_pred)):
        out = dict(image_id=idx, category_id=int(cls_pred_i.item()), score=float(obj_pred_i.item()))

        out["bbox"] = [
            float(bbx_pred_i[1].item()),
            float(bbx_pred_i[0].item()),
            float(bbx_pred_size_i[1].item()),
            float(bbx_pred_size_i[0].item()),
        ]

        # Expand and convert mask if present
        if msk_pred is not None:
            segmentation = Image.fromarray(msk_pred[i].numpy()).resize(original_size[::-1], Image.NEAREST)

            out["segmentation"] = mask_encode(np.asfortranarray(np.array(segmentation)))
            out["segmentation"]["counts"] = str(out["segmentation"]["counts"], "utf-8")

        outs.append(out)

    return outs


def process_panoptic_prediction(panoptic_pred, num_stuff, idx, img_size, original_size):
    # Extract panoptic prediction
    msk_pred, cat_pred, obj_pred, iscrowd_pred = panoptic_pred

    bbx_pred = extract_boxes(msk_pred, cat_pred.numel())

    # Convert bbx and redo clamping
    bbx_pred[:, [0, 2]] = (bbx_pred[:, [0, 2]] / img_size[0] * original_size[0]).clamp(min=0, max=original_size[0])
    bbx_pred[:, [1, 3]] = (bbx_pred[:, [1, 3]] / img_size[1] * original_size[1]).clamp(min=0, max=original_size[1])
    bbx_pred_size = bbx_pred[:, 2:] - bbx_pred[:, :2]

    outs = []
    for i, (obj_i, cat_i, bbx_i, iscrowd_i, bbx_size_i) in enumerate(zip(
            obj_pred, cat_pred, bbx_pred, iscrowd_pred, bbx_pred_size)):
        if iscrowd_i.item() == 1 or cat_i.item() < num_stuff or cat_i.item() == 255:
            continue
        out = dict(image_id=idx, category_id=int(cat_i.item()), score=float(obj_i.item()))

        out["bbox"] = [
            float(bbx_i[1].item()),
            float(bbx_i[0].item()),
            float(bbx_size_i[1].item()),
            float(bbx_size_i[0].item()),
        ]

        segmentation = msk_pred == i
        segmentation = Image.fromarray(segmentation.numpy()).resize(original_size[::-1], Image.NEAREST)
        out["segmentation"] = mask_encode(np.asfortranarray(np.array(segmentation)))
        out["segmentation"]["counts"] = str(out["segmentation"]["counts"], "utf-8")

        outs.append(out)

    return outs


def summarize(predictions, annotations_file, img_list, mask=False):
    msk_map = 0
    with tempfile.NamedTemporaryFile("w") as fid:
        json.dump(predictions, fid)
        fid.flush()

        # Detection
        gt = COCO(annotations_file, img_list)
        pred = gt.loadRes(fid.name)
        pred_eval = COCOeval(gt, pred, "bbox")
        pred_eval.evaluate()
        pred_eval.accumulate()
        pred_eval.summarize()
        det_map = pred_eval.stats[0]

        if mask:
            pred_eval = COCOeval(gt, pred, "segm")
            pred_eval.evaluate()
            pred_eval.accumulate()
            pred_eval.summarize()
            msk_map = pred_eval.stats[0]

    return det_map, msk_map


def summarize_mp(predictions, annotations_file, img_list, log_dir, mask=False):
    # Write partial results to file (all workers)
    rank = dist.get_rank()
    with open(path.join(log_dir, "coco_ap_{:02d}.json".format(rank)), "w") as fid:
        json.dump(predictions, fid)
    with open(path.join(log_dir, "img_list_{:02d}.json".format(rank)), "w") as fid:
        json.dump(img_list, fid)

    dist.barrier()

    # Merge results from all workers and run evaluation (only rank 0)
    if rank == 0:
        predictions = []
        img_list = []

        for i in range(dist.get_world_size()):
            coco_ap_file = path.join(log_dir, "coco_ap_{:02d}.json".format(i))
            with open(coco_ap_file) as fid:
                predictions += json.load(fid)
            remove(coco_ap_file)

            img_list_file = path.join(log_dir, "img_list_{:02d}.json".format(i))
            with open(img_list_file) as fid:
                img_list += json.load(fid)
            remove(img_list_file)

        det_map, msk_map = summarize(predictions, annotations_file, img_list, mask)
    else:
        det_map, msk_map = 0, 0

    dist.barrier()

    return det_map, msk_map


class COCO(_COCO):
    """Modified COCO class that loads only a subset of"""

    def __init__(self, annotation_file, img_list):
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        # Clean-up dataset, removing all images and annotations that are not in img_list
        img_list = set(img_list)
        dataset["images"] = [img for img in dataset["images"] if img["id"] in img_list]
        dataset["annotations"] = [ann for ann in dataset["annotations"] if ann["image_id"] in img_list]

        self.dataset = dataset
        self.createIndex()
