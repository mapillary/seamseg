# Copyright (c) Facebook, Inc. and its affiliates.

from math import ceil

import torch
import torch.nn.functional as functional

from seamseg.utils.parallel import PackedSequence
from seamseg.utils.sequence import pack_padded_images


class SemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, ignore_index=255):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index

    def __call__(self, sem_logits, sem):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        for sem_logits_i, sem_i in zip(sem_logits, sem):
            sem_loss_i = functional.cross_entropy(
                sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0), ignore_index=self.ignore_index, reduction="none")
            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)


class SemanticSegAlgo:
    """Semantic segmentation algorithm

    Parameters
    ----------
    loss : SemanticSegLoss
    num_classes : int
        Number of classes
    """

    def __init__(self, loss, num_classes, ignore_index=255):
        self.loss = loss
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    @staticmethod
    def _pack_logits(sem_logits, valid_size, img_size):
        sem_logits = functional.interpolate(sem_logits, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(sem_logits, valid_size)

    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                confmat.index_add_(
                    0, sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1), confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    @staticmethod
    def _logits(head, x, valid_size, img_size):
        sem_logits = head(x)
        return SemanticSegAlgo._pack_logits(sem_logits, valid_size, img_size)

    def training(self, head, x, sem, valid_size, img_size):
        """Given input features and ground truth compute semantic segmentation loss, confusion matrix and prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        conf_mat : torch.Tensor
            A confusion matrix tensor with shape M x M, where M is the number of classes
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        # Compute logits and prediction
        sem_logits = self._logits(head, x, valid_size, img_size)
        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])

        # Compute loss and confusion matrix
        sem_loss = self.loss(sem_logits, sem)
        conf_mat = self._confusion_matrix(sem_pred, sem)

        return sem_loss, conf_mat, sem_pred

    def inference(self, head, x, valid_size, img_size):
        """Given input features compute semantic segmentation prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        sem_logits = self._logits(head, x, valid_size, img_size)
        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        return sem_pred


def confusion_matrix(sem_pred, sem, num_classes, ignore_index=255):
    confmat = sem_pred.new_zeros(num_classes * num_classes, dtype=torch.float)

    valid = sem != ignore_index
    if valid.any():
        sem_pred = sem_pred[valid]
        sem = sem[valid]

        confmat.index_add_(0, sem.view(-1) * num_classes + sem_pred.view(-1), confmat.new_ones(sem.numel()))

    return confmat.view(num_classes, num_classes)
