import torch


def extract_boxes(mask: torch.Tensor, n_instances: int) -> torch.Tensor: ...


def mask_count(bbx: torch.Tensor, int_mask: torch.Tensor) -> torch.Tensor: ...
