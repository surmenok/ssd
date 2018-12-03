"""
Based on torchvision code:
https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch import tensor as T
from typing import Collection, Sized


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor."""
    if not(_is_numpy_image(pic)):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def to_numpy_image(pic):
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor) or isinstance(pic, torch.cuda.FloatTensor):
        pic = pic.mul(255).byte()

    if torch.is_tensor(pic):
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    return npimg


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def resize_image(img, size):
    return F.interpolate(img.unsqueeze(0), size=size, mode='bilinear')[0, :, :, :]


def resize_boxes(boxes, original_size, new_size):
    # Size is (height, width)
    # Boxes are of shape (N, 4). Coordinates: (top, left, bottom, right)
    m = (new_size[0] / original_size[0], new_size[1] / original_size[1])
    boxes = boxes * T(np.asarray([m[0], m[1], m[0], m[1]]), dtype=torch.float32)
    return boxes


class ToTensor(object):
    """Convert a``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        image, y = sample
        image = to_tensor(image)
        return image, y

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, y = sample
        image = normalize(image, self.mean, self.std)
        return image, y

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToNumpyImage(object):
    """Convert a tensor of shape C x H x W to ndarray image while preserving the value range."""
    def __call__(self, pic):
        """
        Args:
            pic (Tensor): Image to be converted to ndarray.
        Returns:
            ndarray
        """
        return to_numpy_image(pic)

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class Resize(object):
    def __init__(self, size: Collection):
        assert isinstance(size, Collection) and len(size) == 2
        self.size = size

    def __call__(self, sample):
        image, (boxes, classes) = sample
        original_size = (image.shape[1], image.shape[2])
        image = resize_image(image, self.size)
        boxes = resize_boxes(boxes, original_size, self.size)
        return image, (boxes, classes)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
