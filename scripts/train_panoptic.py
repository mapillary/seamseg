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
from seamseg.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, \
    ProposalMatcher
from seamseg.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from seamseg.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from seamseg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from seamseg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from seamseg.config import load_config, DEFAULTS as DEFAULT_CONFIGS
from seamseg.data import ISSTransform, ISSDataset, iss_collate_fn
from seamseg.data.sampler import DistributedARBatchSampler
from seamseg.models.panoptic import PanopticNet, NETWORK_INPUTS
from seamseg.modules.fpn import FPN, FPNBody
from seamseg.modules.heads import FPNMaskHead, RPNHead, FPNSemanticHeadDeeplab
from seamseg.utils import logging
from seamseg.utils.meters import AverageMeter, ConfusionMatrixMeter
from seamseg.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from seamseg.utils.panoptic import panoptic_stats, PanopticPreprocessing
from seamseg.utils.parallel import DistributedDataParallel
from seamseg.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots

parser = argparse.ArgumentParser(description="Panoptic training script")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file")
parser.add_argument("--eval", action="store_true", help="Do a single validation run")
parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                    help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                         "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                         "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                         "will be loaded from the snapshot")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def log_miou(miou, classes):
    logger = logging.get_logger()
    padding = max(len(cls) for cls in classes)

    for miou_i, class_i in zip(miou, classes):
        logger.info(("{:>" + str(padding) + "} : {:.3f}").format(class_i, miou_i.item()))


def make_config(args):
    log_debug("Loading configuration from %s", args.config)

    conf = load_config(args.config, DEFAULT_CONFIGS["panoptic"])

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
    sem_config = config["sem"]
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


def get_panoptic_scores(panoptic_buffer, device, num_stuff):
    # Gather from all workers
    panoptic_buffer = panoptic_buffer.to(device)
    distributed.all_reduce(panoptic_buffer, distributed.ReduceOp.SUM)

    # From buffers to scores
    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.
    scores = panoptic_buffer[0] / denom

    return scores.mean().item(), scores[:num_stuff].mean().item(), scores[num_stuff:].mean().item()


def confusion_matrix(gt, pred):
    conf_mat = gt.new_zeros(256 * 256, dtype=torch.float)
    conf_mat.index_add_(0, gt.view(-1) * 256 + pred.view(-1), conf_mat.new_ones(gt.numel()))
    return conf_mat.view(256, 256)


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
        assert all(msk.size(0) == 1 for msk in batch["msk"]), "Mask R-CNN + segmentation requires panoptic ground truth"

        data_time_meter.update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        # Run network
        losses, _, conf = model(**batch, do_loss=True, do_prediction=False)
        distributed.barrier()

        losses = OrderedDict((k, v.mean()) for k, v in losses.items())
        losses["loss"] = sum(w * l for w, l in zip(loss_weights, losses.values()))

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        # Gather stats from all workers
        losses = all_reduce_losses(losses)
        for k in conf.keys():
            distributed.all_reduce(conf[k], distributed.ReduceOp.SUM)

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())
            meters["sem_conf"].update(conf["sem_conf"].cpu())
        batch_time_meter.update(torch.tensor(time.time() - batch_time))

        # Clean-up
        del batch, losses, conf

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
                    ("roi_msk_loss", meters["roi_msk_loss"]),
                    ("sem_loss", meters["sem_loss"]),
                    ("miou", meters["sem_conf"].iou.mean().item()),
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
    num_classes = dataloader.dataset.num_categories

    loss_meter = AverageMeter(())
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    # Accumulators for ap, mIoU and panoptic computation
    panoptic_buffer = torch.zeros(4, num_classes, dtype=torch.double)
    conf_mat = torch.zeros(256, 256, dtype=torch.double)
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
            assert all(msk.size(0) == 1 for msk in batch["msk"]), \
                "Mask R-CNN + segmentation requires panoptic ground truth"
            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            losses, pred, conf = model(**batch, do_loss=True, do_prediction=True)
            losses = OrderedDict((k, v.mean()) for k, v in losses.items())
            losses = all_reduce_losses(losses)
            loss = sum(w * l for w, l in zip(loss_weights, losses.values()))

            if varargs["eval_mode"] == "separate":
                # Directly accumulate confusion matrix from the network
                conf_mat[:num_classes, :num_classes] += conf["sem_conf"].to(conf_mat)

            # Update meters
            loss_meter.update(loss.cpu())
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            del loss, losses, conf

            # Accumulate COCO AP and panoptic predictions
            for i, (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, msk_gt, cat_gt, iscrowd) in enumerate(zip(
                    pred["sem_pred"], pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"], pred["msk_pred"],
                    batch["msk"], batch["cat"], batch["iscrowd"])):
                msk_gt = msk_gt.squeeze(0)
                sem_gt = cat_gt[msk_gt]

                # Remove crowd from gt
                cmap = msk_gt.new_zeros(cat_gt.numel())
                cmap[~iscrowd] = torch.arange(0, (~iscrowd).long().sum().item(), dtype=cmap.dtype, device=cmap.device)
                msk_gt = cmap[msk_gt]
                cat_gt = cat_gt[~iscrowd]

                # Compute panoptic output
                panoptic_pred = varargs["make_panoptic"](sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, num_stuff)

                # Panoptic evaluation
                panoptic_buffer += torch.stack(
                    panoptic_stats(msk_gt, cat_gt, panoptic_pred, num_classes, num_stuff), dim=0)

                if varargs["eval_mode"] == "panoptic":
                    # Calculate confusion matrix on panoptic output
                    sem_pred = panoptic_pred[1][panoptic_pred[0]]

                    conf_mat_i = confusion_matrix(sem_gt.cpu(), sem_pred)
                    conf_mat += conf_mat_i.to(conf_mat)

                    # Update coco AP from panoptic output
                    if varargs["eval_coco"] and ((panoptic_pred[1] >= num_stuff) & (panoptic_pred[1] != 255)).any():
                        coco_struct += coco_ap.process_panoptic_prediction(
                            panoptic_pred, num_stuff, idxs[i], batch_sizes[i], original_sizes[i])
                        img_list.append(idxs[i])
                elif varargs["eval_mode"] == "separate":
                    # Update coco AP from detection output
                    if varargs["eval_coco"] and bbx_pred is not None:
                        coco_struct += coco_ap.process_prediction(
                            bbx_pred, cls_pred + num_stuff, obj_pred, msk_pred, batch_sizes[i], idxs[i],
                            original_sizes[i])
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

    # Finalize mIoU computation
    conf_mat = conf_mat.to(device=varargs["device"])
    distributed.all_reduce(conf_mat, distributed.ReduceOp.SUM)
    conf_mat = conf_mat.cpu()[:num_classes, :]
    miou = conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0)[:num_classes] - conf_mat.diag())

    # Finalize AP computation
    if varargs["eval_coco"]:
        det_map, msk_map = coco_ap.summarize_mp(coco_struct, varargs["coco_gt"], img_list, varargs["log_dir"], True)

    # Finalize panoptic computation
    panoptic_score, stuff_pq, thing_pq = get_panoptic_scores(panoptic_buffer, varargs["device"], num_stuff)

    # Log results
    log_info("Validation done")
    if varargs["summary"] is not None:
        metrics = OrderedDict()
        metrics["loss"] = loss_meter.mean.item()
        if varargs["eval_coco"]:
            metrics["det_map"] = det_map
            metrics["msk_map"] = msk_map
        metrics["miou"] = miou.mean().item()
        metrics["panoptic"] = panoptic_score
        metrics["stuff_pq"] = stuff_pq
        metrics["thing_pq"] = thing_pq
        metrics["data_time"] = data_time_meter.mean.item()
        metrics["batch_time"] = batch_time_meter.mean.item()

        logging.iteration(
            varargs["summary"], "val", varargs["global_step"],
            varargs["epoch"] + 1, varargs["num_epochs"],
            len(dataloader), len(dataloader), metrics
        )

    log_miou(miou, dataloader.dataset.categories)

    return panoptic_score


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
        snapshot = resume_from_snapshot(model, args.resume, ["body", "rpn_head", "roi_head", "sem_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_debug("Loading pre-trained model from %s", args.pre_train)
        pre_train_from_snapshots(model, args.pre_train, ["body", "rpn_head", "roi_head", "sem_head"])
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
        "roi_bbx_loss": AverageMeter((), momentum),
        "roi_msk_loss": AverageMeter((), momentum),
        "sem_loss": AverageMeter((), momentum),
        "sem_conf": ConfusionMatrixMeter(train_dataloader.dataset.num_categories, momentum)
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

    # Panoptic aggregation strategy
    panoptic_preprocessing = PanopticPreprocessing(config["general"].getfloat("score_threshold"),
                                                   config["general"].getfloat("overlap_threshold"),
                                                   config["general"].getint("min_stuff_area"))
    eval_mode = config["general"]["eval_mode"]
    eval_coco = config["general"].getboolean("eval_coco")
    assert eval_mode in ["panoptic", "separate"], "eval_mode must be one of 'panoptic', 'separate'"

    # Optional: evaluation only:
    if args.eval:
        log_info("Validating epoch %d", starting_epoch - 1)
        validate(model, val_dataloader, config["optimizer"].getstruct("loss_weights"),
                 device=device, summary=summary, global_step=global_step,
                 epoch=starting_epoch - 1, num_epochs=total_epochs,
                 log_interval=config["general"].getint("log_interval"),
                 coco_gt=config["dataloader"]["coco_gt"], make_panoptic=panoptic_preprocessing,
                 eval_mode=eval_mode, eval_coco=eval_coco, log_dir=args.log_dir)
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
                          sem_head=model.module.sem_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          **meters_out_dict)

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            log_info("Validating epoch %d", epoch + 1)
            score = validate(model, val_dataloader, config["optimizer"].getstruct("loss_weights"),
                             device=device, summary=summary, global_step=global_step,
                             epoch=epoch, num_epochs=total_epochs,
                             log_interval=config["general"].getint("log_interval"),
                             coco_gt=config["dataloader"]["coco_gt"],
                             make_panoptic=panoptic_preprocessing, eval_mode=eval_mode, eval_coco=eval_coco,
                             log_dir=args.log_dir)

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
