import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
import vigra

from ...features import superpixel_stacked, from_affinities_to_hmap, size_filter


# watershed on distance transform:
# seeds are generated on the inverted distance transform
# the probability map is used for growing
class WatershedOnDistanceTransform(object):
    def __init__(self, threshold, sigma_seeds,
                 preserve_membrane=True, min_segment_size=0,
                 stacked_2d=False, n_threads=1,
                 from_boundary_maps=False):
        self.threshold = threshold
        self.sigma_seeds = sigma_seeds
        self.preserve_membrane = preserve_membrane
        self.min_segment_size = min_segment_size
        self.stacked_2d = stacked_2d
        self.n_threads = n_threads
        self.from_boundary_maps = from_boundary_maps

    # wrap vigra local maxima properly
    @staticmethod
    def local_maxima(image, *args, **kwargs):
        assert image.ndim in (2, 3), "Unsupported dimensionality: {}".format(image.ndim)
        if image.ndim == 2:
            return vigra.analysis.localMaxima(image, *args, **kwargs)
        if image.ndim == 3:
            return vigra.analysis.localMaxima3D(image, *args, **kwargs)

    def signed_distance_transform(self, hmap):
        # get the distance transform of the pmap
        binary_membranes = (hmap >= self.threshold)
        distance_to_membrane = distance_transform_edt(np.logical_not(binary_membranes))
        # Instead of computing a negative distance transform within the thresholded membrane areas,
        # Use the original probabilities (but inverted)
        if self.preserve_membrane:
            distance_to_membrane[binary_membranes] = -hmap[binary_membranes]
        # Compute the negative distance transform and substract it from the distance transform
        else:
            distance_to_nonmembrane = distance_transform_edt(binary_membranes)
            # Combine the inner/outer distance transforms
            distance_to_nonmembrane[distance_to_nonmembrane > 0] -= 1
            distance_to_membrane[:] -= distance_to_nonmembrane
        return distance_to_membrane.astype('float32')

    def seeds_from_distance_transform(self, distance_transform):
        # we are not using the dt after this point, so it's ok to smooth it
        # and later use it for calculating the seeds
        if self.sigma_seeds > 0.:
            distance_transform = vigra.filters.gaussianSmoothing(distance_transform, self.sigma_seeds)
        # If any seeds end up on the membranes, we'll remove them.
        # This is more likely to happen when the distance transform was generated with preserve_membrane_pmaps=True
        membrane_mask = (distance_transform < 0)
        seeds = self.local_maxima(
            distance_transform, allowPlateaus=True, allowAtBorder=True, marker=np.nan
        )
        seeds = np.isnan(seeds).astype('uint32')
        seeds[membrane_mask] = 0

        return vigra.analysis.labelMultiArrayWithBackground(seeds)

    def wsdt_superpixel(self, hmap):
        # first, we compute the signed distance transform
        dt = self.signed_distance_transform(hmap)
        # next, get the seeds via maxima on the (smoothed) distance transform
        seeds = self.seeds_from_distance_transform(dt)
        # run watershed on the pmap wit dt seeds
        segmentation, seg_max = vigra.analysis.watershedsNew(hmap, seeds=seeds)
        # apply size filter
        if self.min_segment_size > 0:
            segmentation, seg_max = size_filter(
                hmap, segmentation, self.min_segment_size
            )
        return segmentation, seg_max

    def __call__(self, affinities):
        if self.stacked_2d:
            # take the max over inplane nearest affinity channels
            if self.from_boundary_maps:
                assert affinities.ndim == 3
                hmap = affinities
            else:
                assert affinities.ndim == 4
                hmap = np.maximum(affinities[1], affinities[2])
            segmentation, _ = superpixel_stacked(hmap,
                                                 self.wsdt_superpixel,
                                                 self.n_threads)
        else:
            # take the max over all 3 nearest affinity channels
            if self.from_boundary_maps:
                hmap = affinities
            else:
                hmap = np.maximum(affinities[0], affinities[1])
                hmap = np.maximum(hmap, affinities[2])
            segmentation, _ = self.wsdt_superpixel(hmap)
        return segmentation


class IntersectWithBoundaryPixels(object):
    def __init__(self, offsets,
                 boundary_threshold=0.5, # 1.0 all boundary, 0.0 no boundary
                 used_offsets=None,
                 offset_weights=None):
        self.offsets = offsets
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights
        self.boundary_threshold = boundary_threshold

    def __call__(self, affinities, dtws_segm):
        print("FInd hmap")
        hmap = from_affinities_to_hmap(affinities, self.offsets, self.used_offsets,
                                       self.offset_weights)
        pixel_segm = np.arange(np.prod(dtws_segm.shape), dtype='uint64').reshape(dtws_segm.shape) + dtws_segm.max()
        boundary_mask = (1.-hmap) < self.boundary_threshold

        print("Relabel volume")
        dtws_segm = vigra.analysis.labelVolume((dtws_segm * np.logical_not(boundary_mask)).astype('uint32'))

        # fig, ax = segm_vis.get_figure(2, 2, figsize=(14,14))

        # segm_vis.plot_output_affin(ax[0,0], affinities, nb_offset=1, z_slice=1)
        # segm_vis.plot_output_affin(ax[0,1], affinities, nb_offset=2, z_slice=1)
        # segm_vis.plot_gray_image(ax[0,1], hmap,z_slice=1)
        # segm_vis.plot_gray_image(ax[0,1], affinities[2],z_slice=1)

        # segm_vis.plot_segm(ax[1,0], dtws_segm, z_slice=1)

        new_segmentation = np.where(boundary_mask, pixel_segm, dtws_segm)

        # segm_vis.plot_segm(ax[1,1], new_segmentation, z_slice=1)
        # segm_vis.save_plot(fig, "./", "debug_plot.pdf")
        print("Relabel consecutive")
        new_segmentation = vigra.analysis.relabelConsecutive(new_segmentation)[0]

        print("Check new number of nodes!", new_segmentation.max())

        # from ... import vis as vis
        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(7, 7))
        # for a in fig.get_axes():
        #     a.axis('off')
        #
        # # affs_repr = np.linalg.norm(affs_repr, axis=-1)
        # # ax.imshow(affs_repr, interpolation="none")
        #
        # vis.plot_gray_image(ax[0,0], hmap, z_slice=1)
        # vis.plot_gray_image(ax[0, 1], boundary_mask.astype('float32'), z_slice=1)
        # vis.plot_segm(ax[1,0], new_segmentation, z_slice=1, highlight_boundaries=False)
        # vis.plot_segm(ax[1, 1], pixel_segm, z_slice=1, highlight_boundaries=False)
        #
        # pdf_path = "./hmap.pdf"
        # fig.savefig(pdf_path)


        return new_segmentation


class WatershedOnDistanceTransformFromAffinities(WatershedOnDistanceTransform):

    def __init__(self, offsets, threshold, sigma_seeds,
                 used_offsets=None,
                 offset_weights=None,
                 return_hmap=False,
                 invert_affinities=False,
                 intersect_with_boundary_pixels=False,
                 boundary_pixels_kwargs=None,
                 **super_kwargs):
        if 'from_boundary_maps' in super_kwargs:
            assert super_kwargs['from_boundary_maps']
            super_kwargs.pop('from_boundary_maps')
        super(WatershedOnDistanceTransformFromAffinities, self).__init__(threshold,
                                                                         sigma_seeds,
                                                                         from_boundary_maps=True,
                                                                         **super_kwargs)
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.offsets = offsets
        # Consistency of these inputs is checked in from_affinities_to_hmap
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights
        self.return_hmap = return_hmap
        self.invert_affinities = invert_affinities
        self.intersect_with_boundary_pixels = intersect_with_boundary_pixels
        if self.intersect_with_boundary_pixels:
            boundary_pixels_kwargs = boundary_pixels_kwargs if boundary_pixels_kwargs is not None else {}
            self.intersect = IntersectWithBoundaryPixels(offsets, **boundary_pixels_kwargs)



    def __call__(self, *inputs):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        assert len(inputs) == 1 or len(inputs) == 2, len(inputs)
        affinities = inputs[0]
        foreground_mask = inputs[1] if len(inputs) == 2 else None

        assert affinities.shape[0] == len(self.offsets), "{}, {}".format(affinities.shape[0], len(self.offsets))
        assert affinities.ndim == 4, "{}".format(affinities.ndim)

        if self.invert_affinities:
            affinities = 1. - affinities

        print(affinities.mean())

        print("Predict hmap")
        hmap = from_affinities_to_hmap(affinities, self.offsets, self.used_offsets,
                                self.offset_weights)
        print("Run WSDT")
        segmentation = super(WatershedOnDistanceTransformFromAffinities, self).__call__(hmap)

        # Intersect with boundary pixels:
        if self.intersect_with_boundary_pixels:
            print("Intersecting with pixels")
            segmentation = self.intersect(affinities, segmentation)

        # Mask with background (e.g. ignore GT-label):
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape, "{}, {}".format(segmentation.shape, foreground_mask.shape)
            segmentation = segmentation.astype('int64')
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation)*(-1))

        if segmentation.max() > np.uint32(-1):
            print("!!!!!!!!!WARNING!!!!!!!!!! uint32 limit reached!")


        # from ... import vis as vis
        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(7, 7))
        # for a in fig.get_axes():
        #     a.axis('off')
        #
        # # affs_repr = np.linalg.norm(affs_repr, axis=-1)
        # # ax.imshow(affs_repr, interpolation="none")
        #
        # vis.plot_segm(ax[0], dtws, z_slice=1)
        # vis.plot_segm(ax[1], intersect, z_slice=1, highlight_boundaries=False)
        # vis.plot_segm(ax[2], segmentation, z_slice=1, highlight_boundaries=False)
        #
        # pdf_path = "./wsdt.pdf"
        # fig.savefig(pdf_path)


        if self.return_hmap:
            return segmentation, hmap
        else:
            return segmentation


