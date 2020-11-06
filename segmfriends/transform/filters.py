import numpy as np
import scipy.ndimage

# For more options check out http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
# https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#nearest
from ..utils import various as var_utils
try:
    import tinybrain
except ImportError:
    tinybrain = None

def apply_separable_filter(array, kernels):
    ndim = array.ndim
    dtype = array.dtype
    array = array.astype("float32")

    if not isinstance(kernels, (list, tuple)):
        kernels = [kernels for _ in range(ndim)]

    assert len(kernels) == ndim

    for d in range(ndim):
        ndim_kernel = np.copy(kernels[d])
        for _ in range(ndim-1):
            ndim_kernel = np.expand_dims(ndim_kernel, 0)
        ndim_kernel = np.rollaxis(ndim_kernel, -1, d)

        # Perform 1D convolution:
        array = scipy.ndimage.convolve(array, ndim_kernel)

    return array.astype(dtype)


def box_kernel(dws_factor=2):
    assert isinstance(dws_factor, int)
    step = 1. / dws_factor
    return np.ones((dws_factor)) * step


def lanczos(x, alpha=2.):
    assert isinstance(x, np.ndarray)

    x = np.abs(x)

    # Prevent division by zero:
    zero_mask = x < 1e-8
    x[zero_mask] = 1.

    # Compute lanczos function:
    output = np.sin(np.pi * x) / (np.pi * x) * np.sin(np.pi * x / alpha) / (np.pi * x / alpha)
    output[x >= alpha] = 0.
    output[zero_mask] = 1.

    # Normalize:
    output /= output.sum()
    return output


def lanczos_kernel(dws_factor=2, alpha=2.):
    assert isinstance(dws_factor, int)
    step = 1. / dws_factor
    x = np.arange(-alpha + step, alpha, step).astype('float32')
    kernel = lanczos(x, alpha=alpha)
    return kernel


def downscale(array, dws_factor, filter="box"):
    ndim = array.ndim
    if isinstance(dws_factor, int):
        dws_factor = tuple(dws_factor for _ in range(ndim))

    # Compute filter:
    if filter == "box":
        kernels = [box_kernel(dws) for dws in dws_factor]
    else:
        raise ValueError("Filter not implemented")

    # Compute downsample slice:
    ds_factor = [dws_factor for _ in range(ndim)] if isinstance(dws_factor, int) else dws_factor
    ds_slice = tuple(slice(None, None, ds_factor[d]) for d in range(ndim))

    # Smooth and downsample:
    array = apply_separable_filter(array, kernels)
    return array[ds_slice]


def countless3d(array):
    assert tinybrain is not None, "tinybrain module is necessary for 3D Countless downsampling algorithm"
    array = var_utils.make_dimensions_even(array)
    array = tinybrain.downsample.countless3d(array)
    return array
