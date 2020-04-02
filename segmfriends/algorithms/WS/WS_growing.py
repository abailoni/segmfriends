import vigra
import numpy as np
from ...features import from_affinities_to_hmap
from nifty import tools as ntools

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
                 size_of_2d_slices=False,
                 debug=False,
                 with_background=False,
                 invert_affinities=False):
        """
        :param apply_WS_growing: if False, then the 'seed_mask' is returned
        :param size_of_2d_slices: compute size for all z-slices (memory efficient)
        """
        self.size_threshold = size_threshold
        self.invert_affinities = invert_affinities
        self.offsets = offsets
        assert len(offsets[0]) ==  3, "Only 3D supported atm"
        self.hmap_kwargs = {} if hmap_kwargs is None else hmap_kwargs
        self.apply_WS_growing = apply_WS_growing
        self.debug = debug
        self.size_of_2d_slices = size_of_2d_slices
        self.with_background = with_background

    def __call__(self, affinities, label_image):
        assert len(self.offsets) == affinities.shape[0], "Affinities does not match offsets"
        if self.invert_affinities:
            affinities = 1. - affinities

        if self.debug:
            print("Computing segment sizes...")
        label_image = label_image.astype(np.uint32)
        # rag = nrag.gridRag(label_image)
        # _, node_features = nrag.accumulateMeanAndLength(rag, label_image.astype('float32'),blockShape=[1,100,100],
        #                                      numberOfThreads=8,
        #                                      saveMemory=True)
        # nodeSizes = node_features[:, [1]]
        # sizeMap = map_features_to_label_array(label_image,nodeSizes,number_of_threads=6).squeeze()

        def get_size_map(label_image):
            node_sizes = np.bincount(label_image.flatten())
            # rag = nrag.gridRag(label_image)
            # _, node_features = nrag.accumulateMeanAndLength(rag, label_image.astype('float32'),
            #                                                 blockShape=[1, 100, 100],
            #                                                 numberOfThreads=8,
            #                                                 saveMemory=True)
            # nodeSizes = node_features[:, [1]]
            return ntools.mapFeaturesToLabelArray(label_image, node_sizes[:,None], nb_threads=6).squeeze()

        if not self.size_of_2d_slices:
            sizeMap = get_size_map(label_image)
        else:
            sizeMap = np.empty_like(label_image)
            for z in range(label_image.shape[0]):
                sizeMap[[z]] = get_size_map(label_image[[z]])
                print(z, flush=True, end=" ")

        sizeMask = sizeMap > self.size_threshold
        seeds = ((label_image+1)*sizeMask).astype(np.uint32)

        background_mask = None
        if self.with_background:
            background_mask = label_image == 0
            seeds[background_mask] = 0

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
                if self.with_background:
                    watershedResult[z][background_mask[z]] = 0
            # Re-normalize indices numbers:

            if self.with_background:
                return vigra.analysis.labelVolumeWithBackground(watershedResult.astype(np.uint32))
            else:
                return vigra.analysis.labelVolume(watershedResult.astype(np.uint32))
