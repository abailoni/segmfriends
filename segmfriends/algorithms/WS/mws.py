try:
    from affogato.segmentation import compute_mws_segmentation, compute_mws_segmentation_from_affinities
except ImportError:
    compute_mws_segmentation_from_affinities = None
import vigra


class MutexWatershed(object):
    def __init__(self, offsets,
                 beta_bias=0.5, # 0.0: merge everything; 1.0: split everything, or what can be split
                 invert_affinities=False,
                 run_connected_components_on_final_segmentation=False):
        self.beta_bias = beta_bias
        assert isinstance(offsets, list)
        self.offsets = offsets
        self.dim = len(offsets[0])
        self.invert_affinities = invert_affinities
        self.run_connected_components_on_final_segmentation = run_connected_components_on_final_segmentation

    def __call__(self, affinities, foreground_mask=None, mask_used_edges=None):
        assert compute_mws_segmentation_from_affinities is not None, "Please update your version of affogato"
        if self.invert_affinities:
            affinities = 1. - affinities

        segmentation = compute_mws_segmentation_from_affinities(affinities,
                                                                self.offsets,
                                                                beta_parameter=self.beta_bias,
                                                                foreground_mask=foreground_mask,
                                                                edge_mask=mask_used_edges)

        if self.run_connected_components_on_final_segmentation:
            if segmentation.ndim == 3:
                if foreground_mask is None:
                    segmentation = vigra.analysis.labelVolume(segmentation.astype('uint32'))
                else:
                    segmentation = vigra.analysis.labelVolumeWithBackground(segmentation.astype('uint32'))
            elif segmentation.nidm == 2:
                if foreground_mask is None:
                    segmentation = vigra.analysis.labelImage(segmentation.astype('uint32'))
                else:
                    segmentation = vigra.analysis.labelImageWithBackground(segmentation.astype('uint32'))
            else:
                raise NotImplementedError("Connected components implemented only for 2D or 3D segmentations")

        return segmentation



class MutexWatershedOld(object):
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
