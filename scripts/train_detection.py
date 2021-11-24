# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import shutil
import time
from collections import OrderedDict
from os import path

import tensorboardX as tensorboard
import torch
import torch.optim as optim
import torch.utils.data as data
from torch import distributed

import seamseg.models as models
import seamseg.utils.coco_ap as coco_ap
from seamseg.algos.detection import PredictionGenerator, ProposalMatcher, DetectionLoss
from seamseg.algos.fpn import DetectionAlgoFPN, RPNAlgoFPN
from seamseg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from seamseg.config import load_config, DEFAULTS as DEFAULT_CONFIGS
from seamseg.data import ISSTransform, ISSDataset, iss_collate_fn
from seamseg.data.sampler import DistributedARBatchSampler
from seamseg.models.detection import DetectionNet, NETWORK_INPUTS
from seamseg.modules.fpn import FPN, FPNBody
from seamseg.modules.heads import RPNHead, FPNROIHead
from seamseg.utils import logging
from seamseg.utils.meters import AverageMeter
from seamseg.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from seamseg.utils.parallel import DistributedDataParallel
from seamseg.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots

parser = argparse.ArgumentParser(description="Detection training script")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file")
parser.add_argument("--eval", action="store_true", help="Do a single validation run")
parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                    help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                         "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                         "'body', 'rpn_head' or 'roi_head'. In that case only that part of the network "
                         "will be loaded from the snapshot")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_config(args):
    log_debug("Loading configuration from %s", args.config)

    conf = load_config(args.config, DEFAULT_CONFIGS["detection"])

    log_debug("\n%s", config_to_string(conf))
    return conf


def make_dataloader(args, config, rank, world_size):
    config = config["dataloader"]
    log_debug("Creating dataloaders for dataset in %s", args.data)

    # Training dataloader
    train_tf = ISSTransform(config.getint("shortest_size"),
                            config.getint("longest_max_size"),
                            config.getstruct("rgb_mean"),
                            config.getstruct("rgb_std"),
                            config.getboolean("random_flip"),
                            config.getstruct("random_scale"))
    train_db = ISSDataset(args.data, config["train_set"], train_tf)
    train_sampler = DistributedARBatchSampler(
        train_db, config.getint("train_batch_size"), world_size, rank, True)
    train_dl = data.DataLoader(train_db,
                               batch_sampler=train_sampler,
                               collate_fn=iss_collate_fn,
                               pin_memory=True,
                               num_workers=config.getint("num_workers"))

    # Validation dataloader
    val_tf = ISSTransform(config.getint("shortest_size"),
                          config.getint("longest_max_size"),
                          config.getstruct("rgb_mean"),
                          config.getstruct("rgb_std"))
    val_db = ISSDataset(args.data, config["val_set"], val_tf)
    val_sampler = DistributedARBatchSampler(
        val_db, config.getint("val_batch_size"), world_size, rank, False)
    val_dl = data.DataLoader(val_db,
                             batch_sampler=val_sampler,
                             collate_fn=iss_collate_fn,
                             pin_memory=True,
                             num_workers=config.getint("num_workers"))

    return train_dl, val_dl


def make_model(config, num_thing, num_stuff):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

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
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create detection network
    prediction_generator = PredictionGenerator(roi_config.getfloat("nms_threshold"),
                                               roi_config.getfloat("score_threshold"),
                                               roi_config.getint("max_predictions"))
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    roi_loss = DetectionLoss(roi_config.getfloat("sigma"))
    roi_size = roi_config.getstruct("roi_size")
    roi_algo = DetectionAlgoFPN(
        prediction_generator, proposal_matcher, roi_loss, classes, roi_config.getstruct("bbx_reg_weights"),
        roi_config.getint("fpn_canonical_scale"), roi_config.getint("fpn_canonical_level"), roi_size,
        roi_config.getint("fpn_min_level"), roi_config.getint("fpn_levels"))
    roi_head = FPNROIHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create final network
    return DetectionNet(body, rpn_head, roi_head, rpn_algo, roi_algo, classes)


def make_optimizer(config, model, epoch_length):
    body_config = config["body"]
    opt_config = config["optimizer"]
    sch_config = config["scheduler"]

    # Gather parameters from the network
    norm_parameters = []
    other_parameters = []
    for m in model.modules():
        if any(isinstance(m, layer) for layer in NORM_LAYERS):
            norm_parameters += [p for p in m.parameters() if p.requires_grad]
        elif any(isinstance(m, layer) for layer in OTHER_LAYERS):
            other_parameters += [p for p in m.parameters() if p.requires_grad]
    assert len(norm_parameters) + len(other_parameters) == len([p for p in model.parameters() if p.requires_grad]), \
        "Not all parameters that require grad are accounted for in the optimizer"

    # Set-up optimizer hyper-parameters
    parameters = [
        {
            "params": norm_parameters,
            "lr": opt_config.getfloat("lr") if not body_config.getboolean("bn_frozen") else 0.,
            "weight_decay": opt_config.getfloat("weight_decay") if opt_config.getboolean("weight_decay_norm") else 0.
        },
        {
            "params": other_parameters,
            "lr": opt_config.getfloat("lr"),
            "weight_decay": opt_config.getfloat("weight_decay")
        }
    ]

    optimizer = optim.SGD(
        parameters, momentum=opt_config.getfloat("momentum"), nesterov=opt_config.getboolean("nesterov"))

    scheduler = scheduler_from_config(sch_config, optimizer, epoch_length)

    assert sch_config["update_mode"] in ("batch", "epoch")
    batch_update = sch_config["update_mode"] == "batch"
    total_epochs = sch_config.getint("epochs")

    return optimizer, scheduler, batch_update, total_epochs


def train(model, optimizer, scheduler, dataloader, meters, **varargs):
    model.train()
    dataloader.batch_sampler.set_epoch(varargs["epoch"])
    optimizer.zero_grad()
    global_step = varargs["global_step"]
    loss_weights = varargs["loss_weights"]

    data_time_meter = AverageMeter((), meters["loss"].momentum)
    batch_time_meter = AverageMeter((), meters["loss"].momentum)

    data_time = time.time()
    for it, batch in enumerate(dataloader):
        # Upload batch
        batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_INPUTS}

        data_time_meter.update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        # Run network
        losses, _ = model(**batch, do_loss=True, do_prediction=False)
        distributed.barrier()

        losses = OrderedDict((k, v.mean()) for k, v in losses.items())
        losses["loss"] = sum(w * l for w, l in zip(loss_weights, losses.values()))

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        # Gather stats from all workers
        losses = all_reduce_losses(losses)

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())
        batch_time_meter.update(torch.tensor(time.time() - batch_time))

        # Clean-up
        del batch, losses

        # Log
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                varargs["summary"], "train", global_step,
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(dataloader),
                OrderedDict([
                    ("lr", scheduler.get_lr()[0]),
                    ("loss", meters["loss"]),
                    ("obj_loss", meters["obj_loss"]),
                    ("bbx_loss", meters["bbx_loss"]),
                    ("roi_cls_loss", meters["roi_cls_loss"]),
                    ("roi_bbx_loss", meters["roi_bbx_loss"]),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)
                ])
            )

        data_time = time.time()

    return global_step


def validate(model, dataloader, loss_weights, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(varargs["epoch"])

    num_stuff = dataloader.dataset.num_stuff

    loss_meter = AverageMeter(())
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    # Accumulators for ap and panoptic computation
    coco_struct = []
    img_list = []

    data_time = time.time()
    for it, batch in enumerate(dataloader):
        with torch.no_grad():
            idxs = batch["idx"]
            batch_sizes = [img.shape[-2:] for img in batch["img"]]
            original_sizes = batch["size"]

            # Upload batch
            batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_INPUTS}
            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            losses, pred = model(**batch, do_loss=True, do_prediction=True)
            losses = OrderedDict((k, v.mean()) for k, v in losses.items())
            losses = all_reduce_losses(losses)
            loss = sum(w * l for w, l in zip(loss_weights, losses.values()))

            # Update meters
            loss_meter.update(loss.cpu())
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            del loss, losses

            # Accumulate COCO AP and panoptic predictions
            for i, (bbx_pred, cls_pred, obj_pred) in enumerate(
                    zip(pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"])):
                # If there are no detections skip this image
                if bbx_pred is None:
                    continue

                # COCO AP
                coco_struct += coco_ap.process_prediction(
                    bbx_pred, cls_pred + num_stuff, obj_pred, None, batch_sizes[i], idxs[i], original_sizes[i])
                img_list.append(idxs[i])

            del pred, batch

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", varargs["global_step"],
                    varargs["epoch"] + 1, varargs["num_epochs"],
                    it + 1, len(dataloader),
                    OrderedDict([
                        ("loss", loss_meter),
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()

    # Finalize AP computation
    det_map, _ = coco_ap.summarize_mp(coco_struct, varargs["coco_gt"], img_list, varargs["log_dir"], False)

    # Log results
    log_info("Validation done")
    if varargs["summary"] is not None:
        logging.iteration(
            varargs["summary"], "val", varargs["global_step"],
            varargs["epoch"] + 1, varargs["num_epochs"],
            len(dataloader), len(dataloader),
            OrderedDict([
                ("loss", loss_meter.mean.item()),
                ("det_map", det_map),
                ("data_time", data_time_meter.mean.item()),
                ("batch_time", batch_time_meter.mean.item())
            ])
        )

    return det_map


def main(args):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    if rank == 0:
        logging.init(args.log_dir, "training" if not args.eval else "eval")
        summary = tensorboard.SummaryWriter(args.log_dir)
    else:
        summary = None

    # Load configuration
    config = make_config(args)

    # Create dataloaders
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(config, train_dataloader.dataset.num_thing, train_dataloader.dataset.num_stuff)
    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        log_debug("Loading snapshot from %s", args.resume)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "rpn_head", "roi_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_debug("Loading pre-trained model from %s", args.pre_train)
        pre_train_from_snapshots(model, args.pre_train, ["body", "rpn_head", "roi_head"])
    else:
        assert not args.eval, "--resume is needed in eval mode"
        snapshot = None

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id,
                                    find_unused_parameters=True)

    # Create optimizer
    optimizer, scheduler, batch_update, total_epochs = make_optimizer(config, model, len(train_dataloader))
    if args.resume:
        optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])

    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    meters = {
        "loss": AverageMeter((), momentum),
        "obj_loss": AverageMeter((), momentum),
        "bbx_loss": AverageMeter((), momentum),
        "roi_cls_loss": AverageMeter((), momentum),
        "roi_bbx_loss": AverageMeter((), momentum)
    }

    if args.resume:
        starting_epoch = snapshot["training_meta"]["epoch"] + 1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]
        for name, meter in meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        starting_epoch = 0
        best_score = 0
        global_step = 0

    # Optional: evaluation only:
    if args.eval:
        log_info("Validating epoch %d", starting_epoch - 1)
        validate(model, val_dataloader, config["optimizer"].getstruct("loss_weights"),
                 device=device, summary=summary, global_step=global_step,
                 epoch=starting_epoch - 1, num_epochs=total_epochs,
                 log_interval=config["general"].getint("log_interval"),
                 coco_gt=config["dataloader"]["coco_gt"], log_dir=args.log_dir)
        exit(0)

    for epoch in range(starting_epoch, total_epochs):
        log_info("Starting epoch %d", epoch + 1)
        if not batch_update:
            scheduler.step(epoch)

        # Run training epoch
        global_step = train(model, optimizer, scheduler, train_dataloader, meters,
                            batch_update=batch_update, epoch=epoch, summary=summary, device=device,
                            log_interval=config["general"].getint("log_interval"), num_epochs=total_epochs,
                            global_step=global_step, loss_weights=config["optimizer"].getstruct("loss_weights"))

        # Save snapshot (only on rank 0)
        if rank == 0:
            snapshot_file = path.join(args.log_dir, "model_last.pth.tar")
            log_debug("Saving snapshot to %s", snapshot_file)
            meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}
            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                          body=model.module.body.state_dict(),
                          rpn_head=model.module.rpn_head.state_dict(),
                          roi_head=model.module.roi_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          **meters_out_dict)

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            log_info("Validating epoch %d", epoch + 1)
            score = validate(model, val_dataloader, config["optimizer"].getstruct("loss_weights"),
                             device=device, summary=summary, global_step=global_step,
                             epoch=epoch, num_epochs=total_epochs,
                             log_interval=config["general"].getint("log_interval"),
                             coco_gt=config["dataloader"]["coco_gt"], log_dir=args.log_dir)

            # Update the score on the last saved snapshot
            if rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot

            if score > best_score:
                best_score = score
                if rank == 0:
                    shutil.copy(snapshot_file, path.join(args.log_dir, "model_best.pth.tar"))


if __name__ == "__main__":
    main(parser.parse_args())
