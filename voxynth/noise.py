import math
import torch

from typing import List
from torch import Tensor


def perlin(
    shape : List[int],
    wavelength : List[int],
    device: torch.device = None) -> Tensor:
    """
    Generates a perlin noise image.

    Parameters
    ----------
    shape : List[int]
        The desired shape of the output tensor. Can be 2D or 3D.
    wavelength : List[int]
        The wavelength of the noise in voxel coordinates. If a single value
        is provided, it will be used for all dimensions.
    device : torch.device or None, optional
        The device on which the output tensor is allocated. If None, defaults to CPU.

    Returns
    -------
    Tensor
        A Perlin noise image of shape `shape`.
    """
    ndim = len(shape)
    shape = torch.as_tensor(shape, dtype=torch.int32, device=device)
    d = torch.as_tensor(wavelength, dtype=torch.int32, device=device)
    if d.ndim == 0:
        d = d.repeat(ndim)

    res = torch.ceil((shape + 2) / d).type(torch.int32)
    intermediate_shape = (d * res)

    grid = torch.meshgrid([torch.linspace(0, r, d, device=device) for r, d in zip(res, intermediate_shape)], indexing='ij')
    grid = torch.stack(grid, dim=-1) % 1

    if ndim == 2:

        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    elif ndim == 3:

        theta = 2 * math.pi * torch.rand(*(res + 1), device=device)
        phi = 2 * math.pi * torch.rand(*(res + 1), device=device)
        gradients = torch.stack((
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)),
            dim=-1,
        )

    for i, repeat in enumerate(d):
        gradients = gradients.repeat_interleave(repeat, dim=i)

    interp = lambda x : x * x * x * (x * (x * 6 - 15) + 10)

    if ndim == 2:

        g00 = gradients[    :-d[0],    :-d[1]]
        g10 = gradients[d[0]:     ,    :-d[1]]
        g01 = gradients[    :-d[0],d[1]:     ]
        g11 = gradients[d[0]:     ,d[1]:     ]

        n00 = torch.sum(torch.dstack((grid[:, :, 0]    , grid[:, :, 1]    )) * g00, 2)
        n10 = torch.sum(torch.dstack((grid[:, :, 0] - 1, grid[:, :, 1]    )) * g10, 2)
        n01 = torch.sum(torch.dstack((grid[:, :, 0]    , grid[:, :, 1] - 1)) * g01, 2)
        n11 = torch.sum(torch.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

        t = interp(grid)

        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        merged = math.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    elif ndim == 3:

        g000 = gradients[    :-d[0],    :-d[1],    :-d[2]]
        g100 = gradients[d[0]:     ,    :-d[1],    :-d[2]]
        g010 = gradients[    :-d[0],d[1]:     ,    :-d[2]]
        g110 = gradients[d[0]:     ,d[1]:     ,    :-d[2]]
        g001 = gradients[    :-d[0],    :-d[1],d[2]:     ]
        g101 = gradients[d[0]:     ,    :-d[1],d[2]:     ]
        g011 = gradients[    :-d[0],d[1]:     ,d[2]:     ]
        g111 = gradients[d[0]:     ,d[1]:     ,d[2]:     ]

        n000 = torch.sum(torch.stack((grid[:, :, :, 0]    , grid[:, :, :, 1]    , grid[:, :, :, 2]    ), axis=3) * g000, 3)
        n100 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1]    , grid[:, :, :, 2]    ), axis=3) * g100, 3)
        n010 = torch.sum(torch.stack((grid[:, :, :, 0]    , grid[:, :, :, 1] - 1, grid[:, :, :, 2]    ), axis=3) * g010, 3)
        n110 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]    ), axis=3) * g110, 3)
        n001 = torch.sum(torch.stack((grid[:, :, :, 0]    , grid[:, :, :, 1]    , grid[:, :, :, 2] - 1), axis=3) * g001, 3)
        n101 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1]    , grid[:, :, :, 2] - 1), axis=3) * g101, 3)
        n011 = torch.sum(torch.stack((grid[:, :, :, 0]    , grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
        n111 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)

        t = interp(grid)

        n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
        n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
        n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
        n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
        n0  = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
        n1  = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11
        merged = ((1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1)

    # 
    cropping = tuple([slice(0, s) for s in shape])
    merged = merged[cropping]

    return merged
