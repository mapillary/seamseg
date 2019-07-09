from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel

from .scatter_gather import scatter_kwargs, gather


class DistributedDataParallel(TorchDistributedDataParallel):
    """`nn.parallel.DistributedDataParallel` extension which can handle `PackedSequence`s"""

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
