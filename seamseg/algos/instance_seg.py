import torch
import torch.nn.functional as functional

from seamseg.utils.bbx import calculate_shift
from seamseg.utils.parallel import PackedSequence
from seamseg.utils.roi_sampling import roi_sampling


class PredictionGenerator:
    """Instance mask prediction algorithm

    Given mask logits and bounding box / class predictions computes the final mask predictions
    """

    def __call__(self, cls_pred, msk_logits):
        """Compute mask predictions given mask logits and bounding box / class predictions

        Parameters
        ----------
        cls_pred : sequence of torch.Tensor
            A sequence of N tensors with shape S_i, each containing the predicted classes of the detections selected in
            each image, entries can be None.
        msk_logits : torch.Tensor
            A tensor with shape S x C x H x W containing the class-specific mask logits predicted for the instances
            in bbx_preds. Note that S = sum_i S_i.

        Returns
        -------
        msk_pred : PackedSequence
            A sequence of N tensors with shape S_i x H x W containing the mask logits for the detections in each
            image. Entries of `msk_preds` are None for images with no instances.
        """
        # Prepare output lists
        msk_pred = []

        last_it = 0
        for cls_pred_i in cls_pred:
            if cls_pred_i is not None:
                msk_pred_i = msk_logits[last_it:last_it + cls_pred_i.numel()]
                idx = torch.arange(0, cls_pred_i.numel(), dtype=torch.long, device=msk_pred_i.device)
                msk_pred_i = msk_pred_i[idx, cls_pred_i, ...]

                msk_pred.append(msk_pred_i)
                last_it += cls_pred_i.numel()
            else:
                msk_pred.append(None)

        return PackedSequence(msk_pred)


class InstanceSegLoss:
    def __call__(self, msk_logits, cls_lbl, msk_lbl):
        """Instance segmentation loss

        Parameters
        ----------
        msk_logits : torch.Tensor
            A tensor with shape P x C x H x W containing class-specific instance segmentation logits
        cls_lbl : PackedSequence
            A sequence of N tensors of classification labels with shapes M_i, entries can be None
        msk_lbl : PackedSequence
            A sequence of N tensors of instance segmentation labels with shapes P_i x H x W, entries can be None.
            Note that P = sum_i P_i

        Returns
        -------
        msk_loss : torch.Tensor
            A scalar tensor with the loss
        """
        # Get contiguous view of the labels
        cls_lbl, _ = cls_lbl.contiguous

        positives = cls_lbl > 0
        if positives.any().item():
            msk_lbl, _ = msk_lbl.contiguous
            cls_lbl = cls_lbl[positives]
            msk_logits = msk_logits[positives]

            weights = (msk_lbl != -1).to(msk_logits.dtype)
            msk_lbl = (msk_lbl == 1).to(msk_logits.dtype)

            idx = torch.arange(0, msk_logits.size(0), dtype=torch.long, device=msk_logits.device)
            msk_loss = functional.binary_cross_entropy_with_logits(
                msk_logits[idx, cls_lbl - 1], msk_lbl, weight=weights, reduction="sum")
            msk_loss = msk_loss / weights.sum()
        else:
            msk_loss = msk_logits.sum() * 0

        return msk_loss


class InstanceSegAlgo:
    """Base class for instance segmentation algorithms

    Parameters
    ----------
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    bbx_reg_weights : sequence of float
        Weights assigned to the bbx regression coordinates
    lbl_roi_size : tuple of int
        Spatial size of the ROI mask labels as `(height, width)`
    void_is_background : bool
        If True treat void areas as background in the instance mask loss instead of void
    """

    def __init__(self,
                 classes,
                 bbx_reg_weights,
                 lbl_roi_size=(14, 14),
                 void_is_background=False):
        self.num_stuff = classes["stuff"]
        self.bbx_reg_weights = bbx_reg_weights
        self.lbl_roi_size = lbl_roi_size
        self.void_is_background = void_is_background

    @staticmethod
    def _split_and_clip(boxes, scores, index, valid_size):
        boxes_out, scores_out = [], []
        for img_id, valid_size_i in enumerate(valid_size):
            idx = index == img_id
            if idx.any().item():
                boxes_i = boxes[idx]
                boxes_i[:, :, [0, 2]] = torch.clamp(boxes_i[:, :, [0, 2]], min=0, max=valid_size_i[0])
                boxes_i[:, :, [1, 3]] = torch.clamp(boxes_i[:, :, [1, 3]], min=0, max=valid_size_i[1])

                boxes_out.append(boxes_i)
                scores_out.append(scores[idx])
            else:
                boxes.append(None)
                scores.append(None)

        return boxes_out, scores_out

    def _match_to_lbl(self, proposals, bbx, cat, ids, msk, match):
        cls_lbl = []
        bbx_lbl = []
        msk_lbl = []
        for i, (proposals_i, bbx_i, cat_i, ids_i, msk_i, match_i) in enumerate(zip(
                proposals, bbx, cat, ids, msk, match)):
            if match_i is not None:
                pos = match_i >= 0

                # Objectness labels
                cls_lbl_i = proposals_i.new_zeros(proposals_i.size(0), dtype=torch.long)
                cls_lbl_i[pos] = cat_i[match_i[pos]] + 1 - self.num_stuff

                # Bounding box regression labels
                if pos.any().item():
                    bbx_lbl_i = calculate_shift(proposals_i[pos], bbx_i[match_i[pos]])
                    bbx_lbl_i *= bbx_lbl_i.new(self.bbx_reg_weights)

                    iis_lbl_i = ids_i[match_i[pos]]
                    # Compute instance segmentation masks
                    msk_i = roi_sampling(
                        msk_i.unsqueeze(0), proposals_i[pos], msk_i.new_zeros(pos.long().sum().item()),
                        self.lbl_roi_size, interpolation="nearest")

                    # Calculate mask segmentation labels
                    msk_lbl_i = (msk_i == iis_lbl_i.view(-1, 1, 1, 1)).any(dim=1).to(torch.long)
                    if not self.void_is_background:
                        msk_lbl_i[(msk_i == 0).all(dim=1)] = -1
                else:
                    bbx_lbl_i = None
                    msk_lbl_i = None

                cls_lbl.append(cls_lbl_i)
                bbx_lbl.append(bbx_lbl_i)
                msk_lbl.append(msk_lbl_i)
            else:
                cls_lbl.append(None)
                bbx_lbl.append(None)
                msk_lbl.append(None)

        return PackedSequence(cls_lbl), PackedSequence(bbx_lbl), PackedSequence(msk_lbl)

    def training(self, head, x, proposals, bbx, cat, iscrowd, ids, msk, img_size):
        """Given input features, proposals and ground truth compute detection and instance segmentation losses

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute classification and bounding box regression logits given an input feature map and a set
            of proposal bounding boxes. Must be callable as `head(x, proposals, proposals_idx, img_size)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        proposals : PackedSequence
            A sequence of N tensors of bounding box proposals with shapes P_i x 4, entries can be None
        bbx : sequence of torch.Tensor
            A sequence of N tensors of ground truth bounding boxes with shapes M_i x 4, entries can be None
        cat : sequence of torch.Tensor
            A sequence of N tensors of ground truth instance -> category mappings with shapes K_i, entries can
            be None
        iscrowd : sequence of torch.Tensor
            A sequence of N tensors of ground truth crowd bounding boxes with shapes M_i x 4, entries can be None
        ids : sequence of torch.Tensor
            A sequence of N tensors with shapes K_i with the ids of the instances in each image, entries can be None
        msk : sequence of torch.Tensor
            A sequence of N tensors with shapes L_i x H_i x W_i with the ground truth instance segmentations
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        cls_loss : torch.Tensor
            A scalar tensor with the classification loss
        bbx_loss : torch.Tensor
            A scalar tensor with the bounding box regression loss
        msk_loss : torch.Tensor
            A scalar tensor with the instance segmentation loss
        """
        raise NotImplementedError()

    def inference(self, head, x, proposals, valid_size, img_size):
        """Given input features compute detection and instance segmentation predictions

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute classification and bounding box regression logits given an input feature map and a set
            of proposal bounding boxes. Must be callable as `head(x, proposals, proposals_idx, img_size)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        proposals : PackedSequence
            A sequence of N tensors of bounding box proposals with shapes P_i x 4, entries can be None
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        bbx_preds : PackedSequence
            A sequence of N tensors of bounding boxes with shapes S_i x 4, entries are None for images in which no
            detection can be kept according to the selection parameters
        cls_preds : PackedSequence
            A sequence of N tensors of thing class predictions with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        obj_preds : PackedSequence
            A sequence of N tensors of detection confidences with shapes S_i, entries are None for images in which no
            detection can be kept according to the selection parameters
        msk_preds : PackedSequence
            A sequence of N tensors of instance segmentation logits with shapes S_i x H_roi x W_roi, entries are None
            for images in which no detection can be kept accordint to the selection parameters
        """
        raise NotImplementedError()
