from collections import OrderedDict

import torch


class Meter:
    def __init__(self):
        self._states = OrderedDict()

    def register_state(self, name, tensor):
        if name not in self._states and isinstance(tensor, torch.Tensor):
            self._states[name] = tensor

    def __getattr__(self, item):
        if "_states" in self.__dict__:
            _states = self.__dict__["_states"]
            if item in _states:
                return _states[item]
        return self.__dict__[item]

    def reset(self):
        for state in self._states.values():
            state.zero_()

    def state_dict(self):
        return dict(self._states)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self._states:
                self._states[k].copy_(v)
            else:
                raise KeyError("Unexpected key {} in state dict when loading {} from state dict"
                               .format(k, self.__class__.__name__))


class ConstantMeter(Meter):
    def __init__(self, shape):
        super(ConstantMeter, self).__init__()
        self.register_state("last", torch.zeros(shape, dtype=torch.float32))

    def update(self, value):
        self.last.copy_(value)

    @property
    def value(self):
        return self.last


class AverageMeter(ConstantMeter):
    def __init__(self, shape, momentum=1.):
        super(AverageMeter, self).__init__(shape)
        self.register_state("sum", torch.zeros(shape, dtype=torch.float32))
        self.register_state("count", torch.tensor(0, dtype=torch.float32))
        self.momentum = momentum

    def update(self, value):
        super(AverageMeter, self).update(value)
        self.sum.mul_(self.momentum).add_(value)
        self.count.mul_(self.momentum).add_(1.)

    @property
    def mean(self):
        if self.count.item() == 0:
            return torch.tensor(0.)
        else:
            return self.sum / self.count.clamp(min=1)


class ConfusionMatrixMeter(AverageMeter):
    def __init__(self, num_classes, momentum=1.):
        super(ConfusionMatrixMeter, self).__init__((num_classes, num_classes), momentum)

    @property
    def iou(self):
        mean_conf = self.mean
        return mean_conf.diag() / (mean_conf.sum(dim=0) + mean_conf.sum(dim=1) - mean_conf.diag())

    @property
    def precision(self):
        return self.mean.diag() * torch.clamp(1. / self.mean.sum(dim=0), max=1.)

    @property
    def recall(self):
        return self.mean.diag() * torch.clamp(1. / self.mean.sum(dim=1), max=1.)


class PanopticMeter(AverageMeter):
    def panoptic(self):
        return None if self.sum is None else \
            self.sum[0] / (self.sum[1] + 0.5 * self.sum[2] + 0.5 * self.sum[3])

    @property
    def avg(self):
        panoptic = self.panoptic()
        return 0 if panoptic is None else panoptic.mean()
