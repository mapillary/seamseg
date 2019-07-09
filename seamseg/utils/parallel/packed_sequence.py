import torch


def _all_same(lst):
    return not lst or lst.count(lst[0]) == len(lst)


class PackedSequence:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            tensors = args[0]
        else:
            tensors = args

        # Check if all input are tensors of the same type and device
        for tensor in tensors:
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise TypeError("All args must be tensors")
        if not _all_same([tensor.dtype for tensor in tensors if tensor is not None]):
            raise TypeError("All tensors must have the same type")
        if not _all_same([tensor.device for tensor in tensors if tensor is not None]):
            raise TypeError("All tensors must reside on the same device")
        self._tensors = tensors

        # Check useful properties of the sequence
        self._compatible = _all_same([tensor.shape[1:] for tensor in self._tensors if tensor is not None])
        self._all_none = all([tensor is None for tensor in self._tensors])

    def __add__(self, other):
        if not isinstance(other, PackedSequence):
            raise TypeError("other must be a PackedSequence")
        return PackedSequence(self._tensors + other._tensors)

    def __iadd__(self, other):
        if not isinstance(other, PackedSequence):
            raise TypeError("other must be a PackedSequence")
        self._tensors += other._tensors
        return self

    def __len__(self):
        return self._tensors.__len__()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PackedSequence(*self._tensors.__getitem__(item))
        else:
            return self._tensors.__getitem__(item)

    def __iter__(self):
        return self._tensors.__iter__()

    def cuda(self, device=None, non_blocking=False):
        self._tensors = [
            tensor.cuda(device, non_blocking) if tensor is not None else None
            for tensor in self._tensors
        ]
        return self

    def cpu(self):
        self._tensors = [
            tensor.cpu() if tensor is not None else None
            for tensor in self._tensors
        ]
        return self

    @property
    def all_none(self):
        return self._all_none

    @property
    def dtype(self):
        if self.all_none:
            return None
        return next(tensor.dtype for tensor in self._tensors if tensor is not None)

    @property
    def device(self):
        if self.all_none:
            return None
        return next(tensor.device for tensor in self._tensors if tensor is not None)

    @property
    def contiguous(self):
        if not self._compatible:
            raise ValueError("The tensors in the sequence are not compatible for contiguous view")
        if self.all_none:
            return None, None

        packed_tensors = []
        packed_idx = []
        for i, tensor in enumerate(self._tensors):
            if tensor is not None:
                packed_tensors.append(tensor)
                packed_idx.append(tensor.new_full((tensor.size(0),), i, dtype=torch.long))

        return torch.cat(packed_tensors, dim=0), torch.cat(packed_idx, dim=0)
