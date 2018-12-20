import vigra
import numpy as np
from ...features.mappings import map_features_to_label_array
from ...features import from_affinities_to_hmap
from ...features.vigra_feat import accumulate_segment_features_vigra

import nifty.graph.rag as nrag

class SizeThreshAndGrowWithWS(object):
    """
    Ignore all segments smaller than a certain size threshold and
    then grow remaining segments with seeded WS.

    Segments are grown on every slice in 2D.
    """
    def __init__(self, size_threshold,
                 offsets,
                 hmap_kwargs=None,
                 apply_WS_growing=True,
                 debug=True):
        """
        :param apply_WS_growing: if False, then the 'seed_mask' is returned
        """
        self.size_threshold = size_threshold
        self.offsets = offsets
        assert len(offsets[0]) ==  3, "Only 3D supported atm"
        self.hmap_kwargs = {} if hmap_kwargs is None else hmap_kwargs
        self.apply_WS_growing = apply_WS_growing
        self.debug = debug

    def __call__(self, affinities, label_image):
        assert len(self.offsets) == affinities.shape[0], "Affinities does not match offsets"
        if self.debug:
            print("Computing segment sizes...")
        label_image = label_image.astype(np.uint32)
        # rag = nrag.gridRag(label_image)
        # _, node_features = nrag.accumulateMeanAndLength(rag, label_image.astype('float32'),blockShape=[1,100,100],
        #                                      numberOfThreads=8,
        #                                      saveMemory=True)
        # nodeSizes = node_features[:, [1]]
        # sizeMap = map_features_to_label_array(label_image,nodeSizes,number_of_threads=6).squeeze()

        sizeMap = accumulate_segment_features_vigra([label_image],
                                                          [label_image],
                                                          ['Count'],
                                                          map_to_image=True
        ).squeeze()

        sizeMask = sizeMap > self.size_threshold
        seeds = ((label_image+1)*sizeMask).astype(np.uint32)

        if not self.apply_WS_growing:
            return seeds
        else:
            if self.debug:
                print("Computing hmap and WS...")
            hmap = from_affinities_to_hmap(affinities, self.offsets, **self.hmap_kwargs)
            watershedResult = np.empty_like(seeds)
            for z in range(hmap.shape[0]):
                watershedResult[z], _ = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z],
                                                                     method='RegionGrowing')
            # Re-normalize indices numbers:
            return vigra.analysis.labelVolume(watershedResult.astype(np.uint32))