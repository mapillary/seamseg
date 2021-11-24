# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from math import log10
from os import path

from .meters import AverageMeter

_NAME = "seamseg"


def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


def init(log_dir, name):
    logger = logging.getLogger(_NAME)
    logger.setLevel(logging.DEBUG)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Setup file logging
    file_handler = logging.FileHandler(path.join(log_dir, name + ".log"), mode="w")
    file_formatter = logging.Formatter(fmt="%(levelname).1s %(asctime)s %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def get_logger():
    return logging.getLogger(_NAME)


def iteration(summary, phase, global_step, epoch, num_epochs, step, num_steps, values, multiple_lines=False):
    logger = get_logger()

    # Build message and write summary
    msg = _current_total_formatter(epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)
    for k, v in values.items():
        if isinstance(v, AverageMeter):
            msg += "\n" if multiple_lines else "" + "\t{}={:.3f} ({:.3f})".format(k, v.value.item(), v.mean.item())
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v.value.item(), global_step)
        else:
            msg += "\n" if multiple_lines else "" + "\t{}={:.3f}".format(k, v)
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v, global_step)

    # Write log
    logger.info(msg)
