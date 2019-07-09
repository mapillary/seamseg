import torch


def nms(bbx: torch.Tensor, scores: torch.Tensor, threshold: float, n_max: int) -> torch.Tensor: ...
