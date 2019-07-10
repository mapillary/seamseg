import math

import torch
from torch import distributed
from torch.utils.data.sampler import Sampler


class ARBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False, epoch=0):
        super(ARBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._epoch = epoch

        # Split images by orientation
        self.img_sets = self._split_images(range(len(data_source)))

    def _split_images(self, indices):
        img_sizes = self.data_source.img_sizes
        img_sets = [[], []]
        for img_id in indices:
            aspect_ratio = img_sizes[img_id][0] / img_sizes[img_id][1]
            if aspect_ratio < 1:
                img_sets[0].append({"id": img_id, "ar": aspect_ratio})
            else:
                img_sets[1].append({"id": img_id, "ar": aspect_ratio})

        return img_sets

    def _generate_batches(self):
        g = torch.Generator()
        g.manual_seed(self._epoch)

        # Shuffle the two sets separately
        self.img_sets[0] = [self.img_sets[0][i] for i in torch.randperm(len(self.img_sets[0]), generator=g)]
        self.img_sets[1] = [self.img_sets[1][i] for i in torch.randperm(len(self.img_sets[1]), generator=g)]

        batches = []
        leftover = []
        for img_set in self.img_sets:
            batch = []
            for img in img_set:
                batch.append(img)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            leftover += batch

        if not self.drop_last:
            batch = []
            for img in leftover:
                batch.append(img)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []

            if len(batch) != 0:
                batches.append(batch)

        return batches

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        if self.drop_last:
            return len(self.img_sets[0]) // self.batch_size + len(self.img_sets[1]) // self.batch_size
        else:
            return (len(self.img_sets[0]) + len(self.img_sets[1]) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = self._generate_batches()
        for batch in batches:
            batch = sorted(batch, key=lambda i: i["ar"])
            batch = [i["id"] for i in batch]
            yield batch


class DistributedARBatchSampler(ARBatchSampler):
    def __init__(self, data_source, batch_size, num_replicas=None, rank=None, drop_last=False, epoch=0):
        super(DistributedARBatchSampler, self).__init__(data_source, batch_size, drop_last, epoch)

        # Automatically get world size and rank if not provided
        if num_replicas is None:
            num_replicas = distributed.get_world_size()
        if rank is None:
            rank = distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        tot_batches = super(DistributedARBatchSampler, self).__len__()
        self.num_batches = int(math.ceil(tot_batches / self.num_replicas))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batches = self._generate_batches()

        g = torch.Generator()
        g.manual_seed(self._epoch)
        indices = list(torch.randperm(len(batches), generator=g))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.num_batches * self.num_replicas - len(indices))]
        assert len(indices) == self.num_batches * self.num_replicas

        # subsample
        offset = self.num_batches * self.rank
        indices = indices[offset:offset + self.num_batches]
        assert len(indices) == self.num_batches

        for idx in indices:
            batch = sorted(batches[idx], key=lambda i: i["ar"])
            batch = [i["id"] for i in batch]
            yield batch
