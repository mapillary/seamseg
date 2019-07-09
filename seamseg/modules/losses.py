import torch

from seamseg.utils.parallel import PackedSequence


def smooth_l1(x1, x2, sigma):
    """Smooth L1 loss"""
    sigma2 = sigma ** 2

    diff = x1 - x2
    abs_diff = diff.abs()

    mask = (abs_diff.detach() < (1. / sigma2)).float()
    return mask * (sigma2 / 2.) * diff ** 2 + (1 - mask) * (abs_diff - 0.5 / sigma2)


def ohem_loss(loss, ohem=None):
    if isinstance(loss, torch.Tensor):
        loss = loss.view(loss.size(0), -1)
        if ohem is None:
            return loss.mean()

        top_k = min(max(int(ohem * loss.size(1)), 1), loss.size(1))
        if top_k != loss.size(1):
            loss, _ = loss.topk(top_k, dim=1)

        return loss.mean()
    elif isinstance(loss, PackedSequence):
        if ohem is None:
            return sum(loss_i.mean() for loss_i in loss) / len(loss)

        loss_out = loss.data.new_zeros(())
        for loss_i in loss:
            loss_i = loss_i.view(-1)

            top_k = min(max(int(ohem * loss_i.numel()), 1), loss_i.numel())
            if top_k != loss_i.numel():
                loss_i, _ = loss_i.topk(top_k, dim=0)

            loss_out += loss_i.mean()

        return loss_out / len(loss)
