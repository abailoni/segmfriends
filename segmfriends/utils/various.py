import numpy as np
import yaml
# from numba import njit
from itertools import repeat
import os
import h5py

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
  if not os.path.exists(directory):
    os.makedirs(directory)


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


def readHDF5(path, inner_path, crop_slice=None):
    if isinstance(crop_slice, str):
        crop_slice = parse_data_slice(crop_slice)
    elif crop_slice is not None:
        assert isinstance(crop_slice, tuple), "Crop slice not recognized"
        assert all([isinstance(sl, slice) for sl in crop_slice]), "Crop slice not recognized"
    else:
        crop_slice = slice(None)
    with h5py.File(path, 'r') as f:
        output = f[inner_path][crop_slice]
    return output

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
