import math
import numpy as np
import torch
import voxynth

from typing import List
from torch import Tensor

from .filter import gaussian_blur


def smooth_gaussian(shape, sigma, magnitude=1.0, device=None, method='blur'):
    """
    Generates a smooth Gaussian noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    sigma : float
        The spatial smoothing sigma in voxel coordinates.
    magnitude : float
        The standard deviation of the noise.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.
    method : 'blur' or 'upsample'
        Method for noise generation. Upsampling is much faster and more memory efficient
        for larger sigma values, but at the cost of quality.

    Returns
    -------
    Tensor
        A smooth Gaussian noise image of shape `shape`.
    """
    if method == 'blur':
        noise = torch.normal(0, 1, size=shape, device=device)
        noise = gaussian_blur(noise.unsqueeze(0), sigma).squeeze(0)
    elif method == 'upsample':
        downshape = tuple([int(s // sigma) for s in shape])
        noise = torch.normal(0, 1, size=(1, 1, *downshape), device=device)
        mode = 'trilinear' if len(shape) == 3 else 'bilinear'
        noise = torch.nn.functional.interpolate(noise, shape, mode=mode).view(shape)
    else:
        raise ValueError(f'unknown smooth gaussian method `{method}`')

    # in-place normalize
    noise -= noise.mean()
    noise *= magnitude / noise.std()
    return noise


def perlin(shape, smoothing=None, magnitude=1.0, weights=None, device=None, method='blur'):
    """
    Generates a perlin noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    smoothing : float or List[float]
        The spatial smoothing sigma(s) in voxel coordinates.
    magnitude : float
        The standard deviation of the noise.
    weights : float or List[float]
        The weights of the smoothing components (scales). If None, defaults
        to monotonically increasing weights.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.
    method : 'blur' or 'upsample'
        Method for noise generation. Upsampling is much faster and more memory efficient
        for larger sigma values, but at the cost of quality.
        
    Returns
    -------
    Tensor
        A Perlin noise image of shape `shape`.
    """
    if smoothing is None:
        smoothing = 2 ** np.arange(np.log2(max(shape)))[1:]
    elif np.isscalar(smoothing):
        return smooth_gaussian(shape, smoothing, magnitude, device=device)

    if len(smoothing) == 1:
        weights = [None]
    elif weights is None:
        weights = np.arange(len(smoothing)) + 1

    noise = None
    for s, w in zip(smoothing, weights):

        # generate smooth field
        sample = smooth_gaussian(shape, s, device=device, method=method)
        if w is not None:
            sample *= w

        # merge the noise at this scale with the rest
        if noise is None:
            noise = sample
        else:
            noise += sample

    # in-place normalize
    noise -= noise.mean()
    noise *= magnitude / noise.std()
    return noise
