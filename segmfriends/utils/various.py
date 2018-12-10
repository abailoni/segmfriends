import numpy as np
import yaml
from numba import njit


def cantor_pairing_fct(int1, int2):
    """
    Remarks:
        - int1 and int2 should be positive (or zero), otherwise use f(n) = n * 2 if n >= 0; f(n) = -n * 2 - 1 if n < 0
        - int1<=int2 to assure that cantor_pairing_fct(int1, int2)==cantor_pairing_fct(int2, int1)

    It returns an unique integer associated to (int1, int2).
    """
    return np.floor_divide((int1 + int2) * (int1 + int2 + 1), np.array(2, dtype='uint64')) + int2
    # return (int1 + int2) * (int1 + int2 + 1) / 2 + int2

@njit
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
    return slices

# Yaml to dict reader
def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f)
    return readict


