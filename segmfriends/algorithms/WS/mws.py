from nifty.segmentation import compute_mws_segmentation


class MutexWatershed(object):
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

    def __call__(self, affinities):
        if self.invert_affinities:
            affinities = 1. - affinities

        segmentation = compute_mws_segmentation(affinities, self.offsets, self.seperating_channel,
                                     strides=self.stride, randomize_strides=self.randomize_bounds, invert_repulsive_weights=self.invert_dam_channels,
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
