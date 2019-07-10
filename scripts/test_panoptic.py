import argparse
import time
from collections import OrderedDict
from functools import partial
from os import path, mkdir

import numpy as np
import torch
import torch.utils.data as data
import umsgpack
from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from torch import distributed

import seamseg.models as models
from seamseg.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, \
    ProposalMatcher
from seamseg.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from seamseg.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from seamseg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from seamseg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from seamseg.config import load_config, DEFAULTS as DEFAULT_CONFIGS
from seamseg.data import ISSTestDataset, ISSTestTransform, iss_collate_fn
from seamseg.data.sampler import DistributedARBatchSampler
from seamseg.models.panoptic import PanopticNet
from seamseg.modules.fpn import FPN, FPNBody
from seamseg.modules.heads import FPNMaskHead, RPNHead, FPNSemanticHeadDeeplab
from seamseg.utils import logging
from seamseg.utils.meters import AverageMeter
from seamseg.utils.misc import config_to_string, norm_act_from_config
from seamseg.utils.panoptic import PanopticPreprocessing
from seamseg.utils.parallel import DistributedDataParallel
from seamseg.utils.snapshot import resume_from_snapshot

parser = argparse.ArgumentParser(description="Panoptic testing script")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("--meta", type=str, help="Path to metadata file of training dataset")
parser.add_argument("--score_threshold", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="Panoptic disambiguation IoU threshold")
parser.add_argument("--min_area", type=float, default=4096, help="Minimum pixel area for stuff predictions")
parser.add_argument("--raw", action="store_true", help="Save raw predictions instead of rendered images")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("model", metavar="FILE", type=str, help="Path to model file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")
parser.add_argument("out_dir", metavar="DIR", type=str, help="Path to output directory")


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_config(args):
    log_debug("Loading configuration from %s", args.config)

    conf = load_config(args.config, DEFAULT_CONFIGS["panoptic"])

    log_debug("\n%s", config_to_string(conf))
    return conf


def make_dataloader(args, config, rank, world_size):
    config = config["dataloader"]
    log_debug("Creating dataloaders for dataset in %s", args.data)

    # Validation dataloader
    test_tf = ISSTestTransform(config.getint("shortest_size"),
                               config.getstruct("rgb_mean"),
                               config.getstruct("rgb_std"))
    test_db = ISSTestDataset(args.data, test_tf)
    test_sampler = DistributedARBatchSampler(test_db, config.getint("val_batch_size"), world_size, rank, False)
    test_dl = data.DataLoader(test_db,
                              batch_sampler=test_sampler,
                              collate_fn=iss_collate_fn,
                              pin_memory=True,
                              num_workers=config.getint("num_workers"))

    return test_dl


def load_meta(meta_file):
    with open(meta_file, "rb") as fid:
        data = umsgpack.load(fid, encoding="utf-8")
        meta = data["meta"]
    return meta


def make_model(config, num_thing, num_stuff):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN([body_channels[inp] for inp in fpn_inputs],
              fpn_config.getint("out_channels"),
              fpn_config.getint("extra_scales"),
              norm_act_static,
              fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scales"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create instance segmentation network
    bbx_prediction_generator = BbxPredictionGenerator(roi_config.getfloat("nms_threshold"),
                                                      roi_config.getfloat("score_threshold"),
                                                      roi_config.getint("max_predictions"))
    msk_prediction_generator = MskPredictionGenerator()
    roi_size = roi_config.getstruct("roi_size")
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    bbx_loss = DetectionLoss(roi_config.getfloat("sigma"))
    msk_loss = InstanceSegLoss()
    lbl_roi_size = tuple(s * 2 for s in roi_size)
    roi_algo = InstanceSegAlgoFPN(
        bbx_prediction_generator, msk_prediction_generator, proposal_matcher, bbx_loss, msk_loss, classes,
        roi_config.getstruct("bbx_reg_weights"), roi_config.getint("fpn_canonical_scale"),
        roi_config.getint("fpn_canonical_level"), roi_size, roi_config.getint("fpn_min_level"),
        roi_config.getint("fpn_levels"), lbl_roi_size, roi_config.getboolean("void_is_background"))
    roi_head = FPNMaskHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["total"])
    sem_head = FPNSemanticHeadDeeplab(fpn_config.getint("out_channels"),
                                      sem_config.getint("fpn_min_level"),
                                      sem_config.getint("fpn_levels"),
                                      classes["total"],
                                      pooling_size=sem_config.getstruct("pooling_size"),
                                      norm_act=norm_act_static)

    # Create final network
    return PanopticNet(body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, classes)


def test(model, dataloader, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(0)

    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    make_panoptic = varargs["make_panoptic"]
    num_stuff = varargs["num_stuff"]
    save_function = varargs["save_function"]

    data_time = time.time()
    for it, batch in enumerate(dataloader):
        with torch.no_grad():
            # Extract data
            img = batch["img"].cuda(device=varargs["device"], non_blocking=True)

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            _, pred, _ = model(img=img, do_loss=False, do_prediction=True)

            # Update meters
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            for i, (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred) in enumerate(zip(
                    pred["sem_pred"], pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"], pred["msk_pred"])):
                img_info = {
                    "batch_size": batch["img"][i].shape[-2:],
                    "original_size": batch["size"][i],
                    "rel_path": batch["rel_path"][i],
                    "abs_path": batch["abs_path"][i]
                }

                # Compute panoptic output
                panoptic_pred = make_panoptic(sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, num_stuff)

                # Save prediction
                raw_pred = (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred)
                save_function(raw_pred, panoptic_pred, img_info)

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", 0, 1, 1,
                    it + 1, len(dataloader),
                    OrderedDict([
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()


def ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


def save_prediction_image(_, panoptic_pred, img_info, out_dir, colors, num_stuff):
    msk, cat, obj, iscrowd = panoptic_pred

    img = Image.open(img_info["abs_path"])

    # Prepare folders and paths
    folder, img_name = path.split(img_info["rel_path"])
    img_name, _ = path.splitext(img_name)
    out_dir = path.join(out_dir, folder)
    ensure_dir(out_dir)
    out_path = path.join(out_dir, img_name + ".jpg")

    # Render semantic
    sem = cat[msk].numpy()
    crowd = iscrowd[msk].numpy()
    sem[crowd == 1] = 255

    sem_img = Image.fromarray(colors[sem])
    sem_img = sem_img.resize(img_info["original_size"][::-1])

    # Render contours
    is_background = (sem < num_stuff) | (sem == 255)
    msk = msk.numpy()
    msk[is_background] = 0

    contours = find_boundaries(msk, mode="outer", background=0).astype(np.uint8) * 255
    contours = dilation(contours)

    contours = np.expand_dims(contours, -1).repeat(4, -1)
    contours_img = Image.fromarray(contours, mode="RGBA")
    contours_img = contours_img.resize(img_info["original_size"][::-1])

    # Compose final image and save
    out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
    out = Image.alpha_composite(out, contours_img)
    out.convert(mode="RGB").save(out_path)


def save_prediction_raw(raw_pred, _, img_info, out_dir):
    # Prepare folders and paths
    folder, img_name = path.split(img_info["rel_path"])
    img_name, _ = path.splitext(img_name)
    out_dir = path.join(out_dir, folder)
    ensure_dir(out_dir)
    out_path = path.join(out_dir, img_name + ".pth.tar")

    out_data = {
        "sem_pred": raw_pred[0],
        "bbx_pred": raw_pred[1],
        "cls_pred": raw_pred[2],
        "obj_pred": raw_pred[3],
        "msk_pred": raw_pred[4]
    }
    torch.save(out_data, out_path)


def main(args):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    if rank == 0:
        logging.init(args.log_dir, "test")

    # Load configuration
    config = make_config(args)

    # Create dataloader
    test_dataloader = make_dataloader(args, config, rank, world_size)
    meta = load_meta(args.meta)

    # Create model
    model = make_model(config, meta["num_thing"], meta["num_stuff"])

    # Load snapshot
    log_debug("Loading snapshot from %s", args.model)
    resume_from_snapshot(model, args.model, ["body", "rpn_head", "roi_head", "sem_head"])

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id)

    # Panoptic processing parameters
    panoptic_preprocessing = PanopticPreprocessing(args.score_threshold, args.iou_threshold, args.min_area)

    if args.raw:
        save_function = partial(save_prediction_raw, out_dir=args.out_dir)
    else:
        palette = []
        for i in range(256):
            if i < len(meta["palette"]):
                palette.append(meta["palette"][i])
            else:
                palette.append((0, 0, 0))
        palette = np.array(palette, dtype=np.uint8)

        save_function = partial(
            save_prediction_image, out_dir=args.out_dir, colors=palette, num_stuff=meta["num_stuff"])
    test(model, test_dataloader, device=device, summary=None,
         log_interval=config["general"].getint("log_interval"), save_function=save_function,
         make_panoptic=panoptic_preprocessing, num_stuff=meta["num_stuff"])


if __name__ == "__main__":
    main(parser.parse_args())
