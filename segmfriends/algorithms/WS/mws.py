import numpy as np
try:
    import constrained_mst as cmst
except ImportError:
    print("Constrained MST not found, can't use mst watersheds")

from .base import WatershedBase
from ...features import size_filter


class MutexWatershed(WatershedBase):
    # TODO: replace with new version!
    def __init__(self, offsets, stride,
                 seperating_channel=3, invert_dam_channels=True,
                 randomize_bounds=False,
                 min_segment_size=0, stacked_2d=False, n_threads=1,
                 bias=0.5, # 0.0: merge everything; 1.0: split everything, or what can be split
                 invert_affinities=False):
        self.stacked_2d = stacked_2d
        self.bias = 1. - bias
        assert isinstance(offsets, list)
        # if we calculate stacked 2d superpixels from 3d affinity
        # maps, we must adjust the offsets by excludig all offsets
        # with z coordinates and make the rest 2d
        if self.stacked_2d:
            # only keep in-plane channels
            self.keep_channels, self.offsets = self.get_2d_from_3d_offsets(offsets)
        else:
            self.offsets = offsets
        self.dim = len(offsets[0])

        assert isinstance(stride, (int, list))
        if isinstance(stride, int):
            self.stride = self.dim * [stride]
        else:
            self.stride = stride
        assert len(stride) == self.dim

        assert seperating_channel < len(self.offsets)
        self.seperating_channel = seperating_channel
        self.invert_dam_channels = invert_dam_channels
        self.randomize_bounds = randomize_bounds
        self.min_segment_size = min_segment_size
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities

    def damws_superpixel(self, affinities):

        assert affinities.shape[0] >= len(self.offsets)
        # dam channels if ncessary
        if self.invert_dam_channels:
            affinities_ = affinities.copy()
            affinities_[self.seperating_channel:] *= -1
            affinities_[self.seperating_channel:] += 1
        else:
            affinities_ = affinities
        # sort all edges
        sorted_edges = np.argsort(affinities_.ravel())
        # run the mst watershed
        vol_shape = affinities_.shape[1:]
        mst = cmst.ConstrainedWatershed(np.array(vol_shape),
                                        self.offsets,
                                        self.seperating_channel,
                                        np.array(self.stride))
        mst.repulsive_ucc_mst_cut(sorted_edges, 0)
        if self.randomize_bounds:
            mst.compute_randomized_bounds()
        segmentation = mst.get_flat_label_image().reshape(vol_shape)
        # apply the size filter if specified
        if self.min_segment_size > 0:
            hmap = np.sum(affinities[:3], axis=0)
            segmentation, max_label = size_filter(hmap, segmentation, self.min_segment_size)
        else:
            max_label = segmentation.max()
        return segmentation, max_label

    def __call__(self, affinities):
        # FIXME: Actually these are supposed to be a prob. map, not affinities... (not with the new version!)
        if self.invert_affinities:
            affinities = 1. - affinities

        # FIXME: to be completed
        from nifty.segmentation import compute_mws_segmentation
        segmentation = compute_mws_segmentation(affinities, self.offsets, self.seperating_channel,
                                     strides=self.stride, randomize_strides=False, invert_repulsive_weights=True,
                                     bias_cut=0., mask=None,
                                     algorithm='kruskal')

        # # Apply bias (0.0: merge everything; 1.0: split everything, or what can be split)
        # affinities[:self.seperating_channel] -= 2 * (self.bias - 0.5)
        # if self.stacked_2d:
        #     affinities_ = np.require(affinities[self.keep_channels], requirements='C')
        #     segmentation, _ = superpixel_stacked_from_affinities(affinities_,
        #                                                          self.damws_superpixel,
        #                                                          self.n_threads)
        # else:
        #     segmentation, _ = self.damws_superpixel(affinities)
        return segmentation
