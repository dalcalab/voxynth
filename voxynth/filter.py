import torch

from typing import List
from torch import Tensor


def gaussian_kernel(
    sigma: List[float],
    truncate: int = 4,
    device: torch.device = None) -> Tensor:
    """
    Generate a Gaussian kernel with the specified standard deviations.

    Parameters
    ----------
    sigma : List[float]
        A list of standard deviations for each dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.
    device : torch.device, optional
        The device on which to create the kernel.

    Returns
    -------
    Tensor
        A kernel of shape `(2 * truncate * sigma + 1,) * ndim`.

    Notes
    -----
    The kernel is truncated when its values drop below `1e-5` of the maximum value.
    """
    ndim = len(sigma)

    # compute the radii of the kernel for each dimension
    radii = [int(truncate * s + 0.5) for s in sigma]

    # generate a range of indices for each dimension
    ranges = [torch.arange(-r, r + 1, device=device) for r in radii]

    # create a meshgrid of indices for all dimensions and determine shape of the kernel
    coords = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1)
    kernel_shape = coords.shape[:-1]
    
    # convert the standard deviations to a tensor and compute the inverse squares
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    sigma2 = 1 / torch.clip(sigma, min=1e-5).pow(2)

    # reshape the coordinates and compute the pdf
    coords = coords.view(-1, ndim)
    pdf = torch.exp(-0.5 * (coords.pow(2) * sigma2).sum(-1)).view(kernel_shape)

    # normalize the kernel
    pdf /= pdf.sum()
    return pdf


def gaussian_blur(
    image: Tensor,
    sigma: List[float],
    truncate: int = 4) -> Tensor:
    """
    Apply Gaussian blurring to an image.

    Parameters
    ----------
    image : Tensor
        An input tensor of shape `(C, W, H[, D])` to blur.
    sigma : float or List[float]
        Standard deviation(s) of the Gaussian filter along each dimension.
    truncate : int, optional
        The number of standard deviations to extend the kernel before truncating.

    Returns
    -------
    Tensor
        The blurred tensor with the same shape as the input tensor.

    Notes
    -----
    The Gaussian filter is applied using convolution. The size of the filter kernel is
    determined by the standard deviation and the truncation factor.
    """
    ndim = image.ndim - 1

    # generate the filter
    sigma = torch.as_tensor(sigma)
    if sigma.ndim == 0:
        sigma = sigma.repeat(ndim)
    kernel = gaussian_kernel(sigma, truncate, image.device)

    # pad the image
    padding = [s // 2 for s in reversed(kernel.shape) for _ in range(2)]
    image = torch.nn.functional.pad(image, padding, mode='reflect')

    # apply the convolution
    kernel = kernel.expand(image.shape[0], 1, *kernel.shape)
    conv = getattr(torch.nn.functional, f'conv{ndim}d')
    blurred = conv(image, kernel, groups=image.shape[0])
    return blurred
