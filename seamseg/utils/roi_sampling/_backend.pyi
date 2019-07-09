from typing import Tuple

import torch


class PaddingMode:
    Zero = ...
    Border = ...


class Interpolation:
    Bilinear = ...
    Nearest = ...


def roi_sampling_forward(
        x: torch.Tensor, bbx: torch.Tensor, idx: torch.Tensor, out_size: Tuple[int, int],
        interpolation: Interpolation, padding: PaddingMode, valid_mask: bool) -> Tuple[torch.Tensor, torch.Tensor]: ...


def roi_sampling_backward(
        dy: torch.Tensor, bbx: torch.Tensor, idx: torch.Tensor, in_size: Tuple[int, int, int],
        interpolation: Interpolation, padding: PaddingMode) -> torch.Tensor: ...
