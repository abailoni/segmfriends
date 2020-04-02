import numpy as np
import yaml
from itertools import repeat
import os
import h5py
import vigra

from scipy.ndimage import zoom

try:
    import cremi
    from cremi.evaluation import NeuronIds
    from cremi import Volume
except ImportError:
    cremi = None



def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """
    Wrapper around pool.starmap accepting args_iter and kwargs_iter. Example of usage:

        args_iter = zip(repeat(project_name), api_extensions)
        kwargs_iter = repeat(dict(payload={'a': 1}, key=True))
        branches = starmap_with_kwargs(pool, fetch_api, args_iter, kwargs_iter)
    """
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def search_sorted(array, keys_to_search):
    """
    Return the indices of the keys in array. If not found than the indices are masked.
    """
    index = np.argsort(array)
    sorted_x = array[index]
    sorted_index = np.searchsorted(sorted_x, keys_to_search)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = array[yindex] != keys_to_search

    return np.ma.array(yindex, mask=mask)

def cantor_pairing_fct(int1, int2):
    """
    Remarks:
        - int1 and int2 should be positive (or zero), otherwise use f(n) = n * 2 if n >= 0; f(n) = -n * 2 - 1 if n < 0
        - int1<=int2 to assure that cantor_pairing_fct(int1, int2)==cantor_pairing_fct(int2, int1)

    It returns an unique integer associated to (int1, int2).
    """
    return np.floor_divide((int1 + int2) * (int1 + int2 + 1), np.array(2, dtype='uint64')) + int2
    # return (int1 + int2) * (int1 + int2 + 1) / 2 + int2

# @njit
def find_first_index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None

def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return data_slice
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return tuple(slices)

# Yaml to dict reader
def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f, Loader=yaml.FullLoader)
    return readict

def check_dir_and_create(directory):
    '''
    if the directory does not exist, create it
    '''
    folder_exists = os.path.exists(directory)
    if not folder_exists:
        os.makedirs(directory)
    return folder_exists



def compute_output_size_transp_conv(input_size,
                                    padding=0,
                                    stride=1,
                                    dilation=1,
                                    kernel_size=3):
    return int((input_size-1)*stride -2*padding + dilation*(kernel_size-1) + 1)

def compute_output_size_conv(input_size,
                                    padding=0,
                                    stride=1,
                                    dilation=1,
                                    kernel_size=3):
    return int((input_size + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def readHDF5(path,
             inner_path,
             crop_slice=None,
             dtype=None,
             ds_factor=None,
             ds_order=3,
             run_connected_components=False,
             ):
    if isinstance(crop_slice, str):
        crop_slice = parse_data_slice(crop_slice)
    elif crop_slice is not None:
        assert isinstance(crop_slice, tuple), "Crop slice not recognized"
        assert all([isinstance(sl, slice) for sl in crop_slice]), "Crop slice not recognized"
    else:
        crop_slice = slice(None)
    with h5py.File(path, 'r') as f:
        output = f[inner_path][crop_slice]

    if run_connected_components:
        assert output.dtype in [np.dtype("uint32")]
        assert output.ndim == 3 or output.ndim == 2
        output = vigra.analysis.labelVolumeWithBackground(output.astype('uint32'))
    if dtype is not None:
        output = output.astype(dtype)

    if ds_factor is not None:
        assert isinstance(ds_factor, (list, tuple))
        assert output.ndim == len(ds_factor)
        output = zoom(output, tuple(1./fct for fct in ds_factor), order=ds_order)

    return output

def readHDF5_from_volume_config(
        sample,
        path,
        inner_path,
        crop_slice=None,
        dtype=None,
        ds_factor=None,
        ds_order=3,
        run_connected_components=False,
        ):
    if isinstance(path, dict):
        if sample not in path:
            sample = eval(sample)
            assert sample in path
    path = path[sample] if isinstance(path, dict) else path
    inner_path = inner_path[sample] if isinstance(inner_path, dict) else inner_path
    crop_slice = crop_slice[sample] if isinstance(crop_slice, dict) else crop_slice
    dtype = dtype[sample] if isinstance(dtype, dict) else dtype
    return readHDF5(path, inner_path, crop_slice, dtype, ds_factor, ds_order, run_connected_components)

def writeHDF5(data, path, inner_path, compression='gzip'):
    if os.path.exists(path):
        write_mode = 'r+'
    else:
        write_mode = 'w'
    with h5py.File(path, write_mode) as f:
        if inner_path in f:
            del f[inner_path]
        f.create_dataset(inner_path, data=data, compression=compression)

def getHDF5datasets(path):
    # TODO: expand to sub-levels
    with h5py.File(path, 'r') as f:
        datasets = [dt for dt in f]
    return datasets



def cremi_score(gt, seg, return_all_scores=False, border_threshold=None):
    if cremi is None:
        raise ImportError("The cremi package is necessary to run cremi_score()")

    # # the zeros must be kept in the gt since they are the ignore label
    gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
    # seg = vigra.analysis.labelVolume(seg.astype(np.uint32))

    seg = np.array(seg)
    seg = np.require(seg, requirements=['C'])
    # Make sure that all labels are strictly positive:
    seg = seg.astype('uint32')
    # FIXME: it seems to have some trouble with label 0 in the segmentation:
    seg += 1

    gt = np.array(gt)
    gt = np.require(gt, requirements=['C'])
    gt = (gt - 1).astype('uint32')
    # assert gt.min() >= -1

    gt_ = Volume(gt)
    seg_ = Volume(seg)

    metrics = NeuronIds(gt_, border_threshold=border_threshold)
    arand = metrics.adapted_rand(seg_)

    vi_s, vi_m = metrics.voi(seg_)
    cs = np.sqrt(arand * (vi_s + vi_m))
    # cs = (vi_s + vi_m + arand) / 3.
    if return_all_scores:
        return {'cremi-score': cs.item(), 'vi-merge': vi_m.item(), 'vi-split': vi_s.item(), 'adapted-rand': arand.item()}
    else:
        return cs
