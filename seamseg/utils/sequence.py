# Copyright (c) Facebook, Inc. and its affiliates.

from .parallel import PackedSequence


def pad_packed_images(packed_images, pad_value=0., snap_size_to=None):
    """Assemble a padded tensor for a `PackedSequence` of images with different spatial sizes

    This method allows any standard convnet to operate on a `PackedSequence` of images as a batch

    Parameters
    ----------
    packed_images : PackedSequence
        A PackedSequence containing N tensors with different spatial sizes H_i, W_i. The tensors can be either 2D or 3D.
        If they are 3D, they must all have the same number of channels C.
    pad_value : float or int
        Value used to fill the padded areas
    snap_size_to : int or None
        If not None, chose the spatial sizes of the padded tensor to be multiples of this

    Returns
    -------
    padded_images : torch.Tensor
        A tensor with shape N x C x H x W or N x H x W, where `H = max_i H_i` and `W = max_i W_i` containing the images
        of the sequence aligned to the top left corner and padded with `pad_value`
    sizes : list of tuple of int
        A list with the original spatial sizes of the input images
    """
    if packed_images.all_none:
        raise ValueError("at least one image in packed_images should be non-None")

    reference_img = next(img for img in packed_images if img is not None)
    max_size = reference_img.shape[-2:]
    ndims = len(reference_img.shape)
    chn = reference_img.shape[0] if ndims == 3 else 0

    # Check the shapes and find maximum spatial size
    for img in packed_images:
        if img is not None:
            if len(img.shape) != 3 and len(img.shape) != 2:
                raise ValueError("The input sequence must contain 2D or 3D tensors")
            if len(img.shape) != ndims:
                raise ValueError("All tensors in the input sequence must have the same number of dimensions")
            if ndims == 3 and img.shape[0] != chn:
                raise ValueError("3D tensors must all have the same number of channels")
            max_size = [max(s1, s2) for s1, s2 in zip(max_size, img.shape[-2:])]

    # Optional size snapping
    if snap_size_to is not None:
        max_size = [(s + snap_size_to - 1) // snap_size_to * snap_size_to for s in max_size]

    if ndims == 3:
        padded_images = reference_img.new_full([len(packed_images), chn] + max_size, pad_value)
    else:
        padded_images = reference_img.new_full([len(packed_images)] + max_size, pad_value)

    sizes = []
    for i, tensor in enumerate(packed_images):
        if tensor is not None:
            if ndims == 3:
                padded_images[i, :, :tensor.shape[1], :tensor.shape[2]] = tensor
                sizes.append(tensor.shape[1:])
            else:
                padded_images[i, :tensor.shape[0], :tensor.shape[1]] = tensor
                sizes.append(tensor.shape)
        else:
            sizes.append((0, 0))

    return padded_images, sizes


def pack_padded_images(padded_images, sizes):
    """Inverse function of `pad_packed_images`, refer to that for details"""
    images = []
    for img, size in zip(padded_images, sizes):
        if img.dim() == 2:
            images.append(img[:int(size[0]), :int(size[1])])
        else:
            images.append(img[:, :int(size[0]), :int(size[1])])

    return PackedSequence([img.contiguous() for img in images])
