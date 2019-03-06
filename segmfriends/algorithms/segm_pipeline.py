import numpy as np


class SegmentationPipeline(object):

    def __init__(self,
                 fragmenter,
                 agglomerater=None,
                 invert_affinities=False,
                 return_fragments=False):
        # assert isinstance(fragmenter, (WatershedBase, SegmentationPipeline))
        self.fragmenter = fragmenter
        self.agglomerater = agglomerater
        self.invert_affinities = invert_affinities
        self.return_fragments = return_fragments

    def __call__(self, affinities, *args_fragmenter):
        assert isinstance(affinities, np.ndarray)
        assert affinities.ndim == 4, "Need affinities with 4 channels, got %i" % affinities.ndim
        if self.invert_affinities:
            affinities_ = 1. - affinities
        else:
            affinities_ = affinities
        segmentation = self.fragmenter(affinities_, *args_fragmenter)

        # sometimes we want to return fragments for visualisation purposes
        if self.return_fragments:
            fragments = segmentation.copy()
        if self.agglomerater is not None:
            segmentation = self.agglomerater(affinities_, segmentation)

        if self.return_fragments:
            return fragments, segmentation
        else:
            return segmentation
