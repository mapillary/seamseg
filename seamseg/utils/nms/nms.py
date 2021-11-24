# Copyright (c) Facebook, Inc. and its affiliates.

from . import _backend


def nms(bbx, scores, threshold=0.5, n_max=-1):
    """Perform non-maxima suppression

    Select up to n_max bounding boxes from bbx, giving priorities to bounding boxes with greater scores. Each selected
    bounding box suppresses all other not yet selected boxes that intersect it by more than the given threshold.

    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes with shape N x 4
    scores : torch.Tensor
        A tensor of bounding box scores with shape N
    threshold : float
        The minimum iou value for a pair of bounding boxes to be considered a match
    n_max : int
        Maximum number of bounding boxes to select. If n_max <= 0, keep all surviving boxes

    Returns
    -------
    selection : torch.Tensor
        A tensor with the indices of the selected boxes

    """
    selection = _backend.nms(bbx, scores, threshold, n_max)
    return selection.to(device=bbx.device)
