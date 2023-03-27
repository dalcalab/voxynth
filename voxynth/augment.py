import numpy as np
import torch

from torch import Tensor
from typing import List, Tuple

from .utility import chance
from .utility import grid_coordinates
from .filter import gaussian_blur
from .noise import perlin


def image_augment(
    image: Tensor,
    mask: Tensor = None,
    voxsize: List[float] = 1.0,
    smoothing_probability: float = 0.0,
    smoothing_max_sigma: float = 2.0,
    bias_field_probability: float = 0.0,
    bias_field_max_strength: float = 0.5,
    bias_field_wavelength_range: Tuple[float, float] = [20, 60],
    background_noise_probability: float = 0.0,
    background_blob_probability: float = 0.0,
    background_roll_probability: float = 0.0,
    background_roll_max_scale: float = 0.5,
    added_noise_probability: float = 0.0,
    added_noise_max_sigma: float = 0.05,
    wave_artifact_probability: float = 0.0,
    wave_artifact_max_strength: float = 0.05,
    line_corruption_probability: float = 0.0,
    gamma_scaling_probability: float = 0.0,
    gamma_scaling_max: float = 0.8,
    resized_probability: float = 0.0,
    resized_one_axis_probability: float = 0.5,
    resized_max_voxsize: float = 2):
    """
    TODOC
    """

    # parse some info about the input image
    device = image.device
    shape = image.shape[1:]
    channels = image.shape[0]
    ndim = len(shape)

    voxsize = torch.as_tensor(voxsize, device=device)
    if voxsize.ndim == 0:
        voxsize = voxsize.repeat(ndim)

    # 
    image = image.clone() if torch.is_floating_point(image) else image.type(torch.float32)

    # min/max normalize since everything below operates with the
    # assumption that intensities are between 0 and 1
    dims = tuple([i + 1 for i in range(ndim)])
    image -= image.amin(dim=dims, keepdim=True)
    image /= image.amax(dim=dims, keepdim=True)

    #
    if mask is None:
        mask = image.sum(0) > 0
    elif mask.ndim != (image.ndim - 1):
        raise ValueError(f'expected mask to have {ndim} dims, but got shape {mask.shape}')

    # 
    background = mask == 0
    bg_coords = background.nonzero(as_tuple=True)
    bg_size = (bg_coords[0].shape[0],)
    bg_channel = torch.zeros(bg_size, dtype=torch.int64, device=device)


    for channel in range(channels):

        # ---- intensity smoothing ----

        # apply a random gaussian smoothing kernel
        # NOTE: this is applied consistently across channels
        if chance(smoothing_probability):
            max_sigma = (smoothing_max_sigma / voxsize.min()).cpu().numpy()  # TODO: just use torch
            if chance(0.5):
                # half the time let's only smooth one axis to emulate thick slice data
                sigma = np.zeros(ndim)
                sigma[np.random.randint(ndim)] = np.random.uniform(0, max_sigma)
            else:
                # the other half of the time just smooth in all dimensions
                sigma = np.random.uniform(0, max_sigma, size=ndim)
            image[channel] = gaussian_blur(image[channel].unsqueeze(0), sigma)
    
        # ---- background synthesis ----

        # this is just a trick to modify the background of each channel efficiently.
        # bg_channel_coords is a list of the image coordinates corresponding
        # to the currrent channel
        bg_channel.fill_(channel)
        bg_channel_coords = (bg_channel, *bg_coords)

        # set the background as Gaussian noise with a random mean
        if chance(background_noise_probability):
            wavelength = torch.ceil(torch.tensor(np.random.uniform(1, 16)) / voxsize)
            bg_image = perlin(shape, wavelength, device=device)
            bg_image /= np.random.uniform(1, 10)
            bg_image += np.random.rand()
        else:
            bg_image = torch.zeros(shape, device=device)

        # 
        if chance(background_blob_probability):
            wavelength = torch.ceil(torch.tensor(np.random.uniform(32, 64)) / voxsize)
            noise = perlin(shape, wavelength, device=device)
            blobs = noise > np.random.uniform(-0.2, 0.2)
            bg_image[blobs] = np.random.rand() if chance(0.5) else noise[blobs] * np.random.rand()

        # now we copy-paste parts of the image around the background via axis rolling
        if chance(background_roll_probability):
            for i in range(np.random.randint(1, 4)):
                dims = tuple(np.random.permutation(ndim)[:np.random.choice((1, 2))])
                shifts = [int(np.random.uniform(shape[d] / 4, shape[d] / 2)) for d in dims]
                shifts = tuple(np.asarray(shifts) * np.random.choice([-1, 1], size=len(shifts)))
                scale = np.random.randn() * background_roll_max_scale
                bg_image += scale * torch.roll(image[channel], shifts, dims=dims)

        image[bg_channel_coords] = bg_image[background].clip(0, 1)

        # ---- corruptions ----

        # synthesize a bias field of varying degrees of size and
        # intensity with perlin noise
        if chance(bias_field_probability):
            wavelength = torch.ceil(torch.tensor(np.random.uniform(32, 128)) / voxsize)
            image[channel] *= random_bias_field(
                            shape=shape,
                            wavelength=wavelength,
                            strength=np.random.uniform(0, bias_field_max_strength),
                            device=device)

        # some small corruptions that fill random slices with random intensities
        if chance(line_corruption_probability):
            for i in range(np.random.randint(1, 4)):
                indices = [slice(0, s) for s in shape]
                axis = np.random.randint(ndim)
                indices[axis] = np.random.randint(shape[axis])
                indices = (slice(channel, channel + 1), *indices)
                image[indices] = np.random.rand()

        # add gaussian noise across the entire image
        if chance(added_noise_probability):
            std = np.random.uniform(0, added_noise_max_sigma)
            image[channel] += torch.normal(mean=0, std=std, size=shape, device=device)

        # generate linear or circular wave artifacts across the image
        if chance(wave_artifact_probability):
            meshgrid = grid_coordinates(shape, device=device)
            if chance(0.5):
                wavelength = np.random.uniform(2, 8)
                grating = random_linear_wave(meshgrid, wavelength)
                image[channel] += grating * np.random.rand() * wave_artifact_max_strength
            else:
                wavelength = np.random.uniform(1, 2)
                grating = random_spherical_wave(meshgrid, wavelength)
                image[channel] += grating * np.random.rand() * wave_artifact_max_strength

        # ---- resizing ----

        # here we account for low-resolution images that have been upsampled to the target resolution
        if chance(resized_probability):
            # there's no need to downsample if the target resolution is less
            # than the max ds voxsize allowed
            if torch.any(voxsize < resized_max_voxsize):
                # half the time only downsample one random axis to mimic thick slice acquisitions
                if chance(resized_one_axis_probability):
                    vsa = np.full(ndim, voxsize, dtype=np.float32)
                    vsa[np.random.randint(ndim)] = np.random.uniform(voxsize.min().cpu(), resized_max_voxsize)
                    scale = tuple(1 / vsa)
                else:
                    scale = tuple(1 / np.random.uniform(voxsize, resized_max_voxsize))
                # downsample then resample, always use nearest here because if we don't enable align_corners,
                # then the image will be moved around a lot
                linear = 'trilinear' if ndim == 3 else 'bilinear'
                ds = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=scale, mode=linear, align_corners=True)
                image = torch.nn.functional.interpolate(ds, shape, mode=linear, align_corners=True).squeeze(0)

        # ---- gamma exponentiation ----

        # one final min/max normalization across channels
        image[channel] -= image[channel].min()
        image[channel] /= image[channel].max()

        # gamma exponentiation of the intensities to shift signal
        # distribution in a random direction
        if chance(gamma_scaling_probability):
            gamma = np.random.uniform(-gamma_scaling_max, gamma_scaling_max)
            image[channel] = image[channel].pow(np.exp(gamma))

    return image


def random_bias_field(
    shape : List[int],
    wavelength : int,
    strength : float = 0.1,
    device: torch.device = None) -> Tensor:
    """
    Generate a random bias field with perlin noise. The bias field
    is generated by exponentiating the noise.

    Parameters
    ----------
    shape : List[int]
        Shape of the bias field.
    wavelength : int
        Wavelength of the bias field in voxels.
    strength : float, optional
        Magnitude of the field.
    device : torch.device, optional
        The device to create the field on.

    Returns
    -------
    Tensor
        Bias field image.
    """
    bias = perlin(shape=shape, wavelength=wavelength, device=device)
    bias *= strength / bias.std()
    return bias.exp()


def random_linear_wave(meshgrid : Tensor, wavelength : float) -> Tensor:
    """
    TODOC
    """
    # pick two random axes and generate an angled wave grating
    ndim = meshgrid.ndim - 1
    angle = 0 if wavelength < 4 else np.random.uniform(0, np.pi)
    if ndim == 3:
        a, b = [meshgrid[..., d] for d in np.random.permutation(3)[:2]]
    elif ndim == 2:
        a, b = meshgrid[..., 0], meshgrid[..., 1]
    grating = torch.sin(2 * np.pi * (a * np.cos(angle) + b * np.sin(angle)) / wavelength)
    return grating


def random_spherical_wave(meshgrid : Tensor, wavelength : float) -> Tensor:
    """
    TODOC
    """
    # generate a circular wave signal emanating from a random point in the image
    ndim = meshgrid.ndim - 1
    delta = [np.random.uniform(0, s) for s in meshgrid.shape[:-1]]
    if ndim == 3:
        x, y, z = [meshgrid[..., d] - delta[d]  for d in range(ndim)]
        grating = torch.sin(torch.sqrt(x ** 2 + y ** 2 + z ** 2) * wavelength)
    elif ndim == 2:
        # TODO: implement this for 2D, shouldn't be that hard, just lazy
        raise NotImplementedError('spherical waves not yet implemented for 2D images')
    return grating


def random_cropping_mask(mask: Tensor) -> Tensor:
    """
    TODOC
    """
    # this code isn't pretty but basically it computes a bounding box around the
    # pertinent tissue (determined by the mask or background label), randomly selects
    # an axis (or two) as a 'crop axis', and moves that axis towards the tissue by
    # some reasonable amount
    shape = mask.shape[1:]
    ndim = len(shape)
    crop_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    nonzeros = mask.nonzero()[:, 1:]
    mincoord = nonzeros.min(0)[0].cpu()
    maxcoord = nonzeros.max(0)[0].cpu() + 1
    bbox = tuple([slice(a, b) for a, b in zip(mincoord, maxcoord)])
    for _ in range(np.random.randint(1, 3)):
        axis = np.random.randint(ndim)
        s = bbox[axis]
        # don't displace the crop axis by more than 1/3 the tissue width
        displacement = int(np.random.uniform(0, (s.stop - s.start) / 3))
        cropping = [slice(0, d) for d in shape]
        # either move the low axis up and the high axis down... if that makes any sense
        if chance(0.5):
            cropping[axis] = slice(0, s.start + displacement)
        else:
            cropping[axis] = slice(s.stop - displacement, shape[axis])
        crop_mask[tuple(cropping)] = 1
    return crop_mask
