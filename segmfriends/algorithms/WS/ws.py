from vigra.analysis import watershedsNew
from scipy.ndimage.filters import gaussian_filter, median_filter

import numpy as np
from .base import WatershedBase
from ...features import superpixel_stacked, from_affinities_to_hmap

class WatershedFromAffinities(object):
    def __init__(self, offsets,
                       used_offsets=None,
                       offset_weights=None,
                       stacked_2d=True,
                       invert_affinities=False,
                       return_hmap=False,
                       n_threads=1):
        """
        :param invert_affinities: by default it uses affinities (1: merge, 0: split).
                Set to True if necessary.
        :param offsets: np.array or list
        :param used_offsets: list of offset indices
        :param offset_weights: list of weights
        """
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.offsets = offsets
        # Consistency of these inputs is checked in from_affinities_to_hmap
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights

        self.invert_affinities = invert_affinities
        self.stacked_2d = stacked_2d
        self.n_threads = n_threads
        self.return_hmap = return_hmap



    def ws_superpixels(self, hmap_z_slice):
        assert hmap_z_slice.ndim ==  2 or hmap_z_slice.ndim == 3

        # TODO: add option for Gaussian smoothing..?

        # TODO: make it optional
        # Let's see, perhaps it's too expensive...
        hmap_z_slice = median_filter(hmap_z_slice, 4)

        segmentation, max_label = watershedsNew(hmap_z_slice)
        return segmentation, max_label

    def __call__(self, affinities):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        assert affinities.shape[0] == len(self.offsets)
        assert affinities.ndim == 4

        if self.invert_affinities:
            affinities = 1. - affinities

        hmap = from_affinities_to_hmap(affinities, self.offsets, self.used_offsets,
                                self.offset_weights)

        if self.stacked_2d:
            segmentation, _ = superpixel_stacked(hmap, self.ws_superpixels, self.n_threads)
        else:
            segmentation, _ = self.ws_superpixels(hmap)

        if self.return_hmap:
            return segmentation, hmap
        else:
            return segmentation

