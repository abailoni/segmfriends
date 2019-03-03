import numpy as np
import vigra
from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
from nifty.graph import rag as nrag, agglo as nagglo

from ...features import map_features_to_label_array, map_edge_features_to_image
from ...features.vigra_feat import accumulate_segment_features_vigra
from .segm_to_bound import get_boundary_offsets
from ..segm_to_bound import compute_boundary_mask_from_label_image
from ...utils import cantor_pairing_fct
from ..combine_segms_CY import find_best_agglomeration, find_split_GT


class FindBestAgglFromOversegmAndGT(Transform):
    """
    Given an initial segm. and some GT labels, it finds the best agglomeration that can be done to
    get as close as possible to the GT labels.
    """
    def __init__(self, ignore_label=0,
                 border_thickness=0,
                 number_of_threads=8,
                 break_oversegm_on_GT_borders=False,
                 return_node_labels_array=False,
                 undersegm_rel_threshold=None,
                 **super_kwargs):
        """
        :param ignore_label:
        :param border_thickness: Erode the GT labels and insert some ignore label on the boundary between segments.
        :param break_oversegm_on_GT_borders:
                Break oversegm segments on transitions to GT ignore labels (avoid to have huge segments that
                are labelled with the ignore label in the best_agglomeration)
        :param undersegm_rel_threshold:
                The best matching GT label should cover at least this relative percentage of the segment, otherwise
                we consider it undersegmentated and we label it with the ignore label.
                E.g. 0.7 means: the best matching GT label should be at least 70% of the segment.
        """
        self.ignore_label = ignore_label
        self.border_thickness = border_thickness
        self.number_of_threads = number_of_threads
        self.break_oversegm_on_GT_borders = break_oversegm_on_GT_borders
        self.return_node_labels_array = return_node_labels_array
        self.undersegm_rel_threshold = undersegm_rel_threshold
        super(FindBestAgglFromOversegmAndGT, self).__init__(**super_kwargs)

        self.offsets = None
        if border_thickness != 0:
            self.offsets = np.array(get_boundary_offsets([0,border_thickness,border_thickness]))
            # self.get_border_mask = Segmentation2AffinitiesFromOffsets(3,
            #                                                       offsets=[[0,border_thickness,0],
            #                                                                [0,0,border_thickness]],
            #                                                       add_singleton_channel_dimension=True)

    def batch_function(self, tensors):
        init_segm, GT_labels = tensors

        if self.break_oversegm_on_GT_borders:
            # TODO: find a better bug-free solution:
            GT_border = ((GT_labels == self.ignore_label).astype(np.int32) + 3) * 3
            init_segm = np.array(
                vigra.analysis.labelMultiArray((init_segm * GT_border).astype(np.uint32)))
        else:
            init_segm, _, _ = vigra.analysis.relabelConsecutive(init_segm.astype('uint32'))

        if self.ignore_label == 0:
            # This keeps the zero label:
            GT_labels, _, _ = vigra.analysis.relabelConsecutive(GT_labels.astype('uint32'))
        else:
            raise NotImplementedError()


        if self.border_thickness != 0:

            border_affs = 1 - compute_boundary_mask_from_label_image(GT_labels,
                                                                     self.offsets,
                                                                     compress_channels=False,
                                                                     channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask==0] = self.ignore_label

        GT_labels_nodes = find_best_agglomeration(init_segm, GT_labels,
                                                  undersegm_rel_threshold=self.undersegm_rel_threshold,
                                                  ignore_label=self.ignore_label)
        if self.return_node_labels_array:
            return GT_labels_nodes

        best_agglomeration = (
            map_features_to_label_array(
                init_segm,
                np.expand_dims(GT_labels_nodes, axis=-1),
                number_of_threads=self.number_of_threads)
        ).astype(np.int64)[...,0]

        return best_agglomeration


class FindSplitGT(Transform):
    def __init__(self,
                 size_small_segments_rel,
                 ignore_label=0,
                 border_thickness_GT=0,
                 border_thickness_segm=0,
                 number_of_threads=8,
                 break_oversegm_on_GT_borders=False,
                 **super_kwargs):
        self.ignore_label = ignore_label
        self.border_thickness_GT = border_thickness_GT
        self.border_thickness_segm = border_thickness_segm
        self.number_of_threads = number_of_threads
        self.break_oversegm_on_GT_borders = break_oversegm_on_GT_borders
        self.size_small_segments_rel = size_small_segments_rel
        super(FindSplitGT, self).__init__(**super_kwargs)

        self.offsets = None
        if border_thickness_GT != 0:
            self.offsets_GT = np.array(get_boundary_offsets([0,border_thickness_GT,border_thickness_GT]))
        if border_thickness_segm != 0:
            self.offsets_segm = np.array(get_boundary_offsets([0,border_thickness_segm,border_thickness_segm]))

    def batch_function(self, tensors):
        finalSegm, GT_labels = tensors

        ignore_mask = GT_labels == self.ignore_label
        if self.break_oversegm_on_GT_borders:
            # TODO: find a better bug-free solution:
            finalSegm = np.array(
                vigra.analysis.labelMultiArray((finalSegm * ((ignore_mask.astype(np.int32) + 3) * 3)).astype(np.uint32)))
        else:
            finalSegm, _, _ = vigra.analysis.relabelConsecutive(finalSegm.astype('uint32'))

        if self.ignore_label == 0:
            # This keeps the zero label:
            GT_labels, _, _ = vigra.analysis.relabelConsecutive(GT_labels.astype('uint32'))
        else:
            raise NotImplementedError()


        if self.border_thickness_GT != 0:

            border_affs = 1 - compute_boundary_mask_from_label_image(GT_labels,
                                                                     self.offsets_GT,
                                                                     compress_channels=False,
                                                                     channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask==0] = self.ignore_label

        # Erode also oversegmentation:
        if self.border_thickness_segm != 0:
            border_affs = 1 - compute_boundary_mask_from_label_image(finalSegm,
                                                                     self.offsets_segm,
                                                                     compress_channels=False,
                                                                     channel_affs=0)
            border_mask = np.logical_and(border_affs[0], border_affs[1])
            if self.ignore_label == 0:
                GT_labels *= border_mask
            else:
                GT_labels[border_mask == 0] = self.ignore_label

        split_GT = find_split_GT(finalSegm, GT_labels,
                                                  size_small_segments_rel=self.size_small_segments_rel,
                                                  ignore_label=self.ignore_label)
        # split_GT = GT_labels


        if True:
            new_split_GT = np.zeros_like(split_GT)
            for z in range(split_GT.shape[0]):
                z_slice = split_GT[[z]].astype(np.uint32)
                z_slice_compontents = np.array(
                    vigra.analysis.labelMultiArrayWithBackground(z_slice, background_value=self.ignore_label))
                sizeMap = accumulate_segment_features_vigra([z_slice_compontents],
                                                            [z_slice_compontents],
                                                            ['Count'],
                                                            ignore_label=0,
                                                            map_to_image=True
                                                            ).squeeze(axis=-1)

                z_slice[sizeMap <= 50] = self.ignore_label

                # WS nonsense:
                mask_for_WS = compute_boundary_mask_from_label_image(finalSegm[[z]],
                                                                     np.array(
                                            get_boundary_offsets([0, 1, 1])),
                                                                     compress_channels=True)
                mask_for_WS = - vigra.filters.boundaryDistanceTransform(mask_for_WS.astype('float32'))
                mask_for_WS = np.random.normal(scale=0.001, size=mask_for_WS.shape) + mask_for_WS
                mask_for_WS += abs(mask_for_WS.min())


                mask_for_eroding_GT = 1 - compute_boundary_mask_from_label_image(finalSegm[[z]],
                                                                                 np.array(
                                            get_boundary_offsets([0, 8, 8])),
                                                                                 compress_channels=True)
                seeds = (z_slice + 1) * mask_for_eroding_GT

                z_slice, _ = vigra.analysis.watershedsNew(mask_for_WS[0].astype('float32'), seeds=seeds[0].astype('uint32'),
                                                                     method='RegionGrowing')

                new_split_GT[z] = z_slice - 1
            split_GT = new_split_GT

        split_GT = (split_GT + 1) * (1 - ignore_mask)

        return split_GT


class ComputeStructuredWeightsWrongMerges(Transform):
    def __init__(self,
                 offsets,
                 dim=3,
                 ignore_label=0,
                 number_of_threads=8,
                 weighting_merge_mistakes=1.0,
                 weighting_split_mistakes=1.0,
                 trained_mistakes='all_mistakes',
                 train_correct_predictions=False,
                 **super_kwargs):
        """
        :param trained_mistakes: 'only_merge_mistakes', 'only_split_mistakes', 'all_mistakes'
        :param weighting: max is 1.0, min is 0.0 (this function has no effect and all weights are 1.0)
        :param train_correct_predictions: if True, correct boundaries receives a weight 1.0 (inner part of the segments are instead always set to zero)
        """
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        assert len(offsets) > 0, "`offsets` must not be empty."
        assert ignore_label >= 0

        assert dim in (2, 3), "Affinities are only supported for 2d and 3d input"

        self.offsets = np.array(offsets)
        self.ignore_label = ignore_label
        self.weighting_merge_mistakes = weighting_merge_mistakes
        self.weighting_split_mistakes = weighting_split_mistakes
        self.dim = dim
        self.number_of_threads = number_of_threads
        self.train_correct_predictions = train_correct_predictions

        assert trained_mistakes in ['only_merge_mistakes', 'only_split_mistakes', 'all_mistakes'], '{} option not supported'.format(trained_mistakes)
        self.train_merge_mistakes = trained_mistakes in ['only_merge_mistakes', 'all_mistakes']
        self.train_split_mistakes = trained_mistakes in ['only_split_mistakes', 'all_mistakes']

        super(ComputeStructuredWeightsWrongMerges, self).__init__(**super_kwargs)


    def batch_function(self, tensors):
        # TODO: add check for the ignore label!!
        finalSegm, GT_labels = tensors

        intersection_segm = cantor_pairing_fct(finalSegm, GT_labels)
        # FIXME: by intersecting I could get troubles with same label in not-connected segments (run vigra connected componets)
        raise DeprecationWarning()
        intersection_segm, max_label, _ = vigra.analysis.relabelConsecutive(intersection_segm.astype('uint32'))

        rag = nrag.gridRag(intersection_segm, numberOfThreads=self.number_of_threads)

        _, node_features = nrag.accumulateMeanAndLength(rag=rag, data=GT_labels.astype('float32'),
                                        numberOfThreads=self.number_of_threads)

        GT_size_nodes = node_features[:,1].astype('int')
        GT_labels_nodes = node_features[:,0].astype('int')

        _, node_features = nrag.accumulateMeanAndLength(rag=rag, data=finalSegm.astype('float32'),
                                                           numberOfThreads=self.number_of_threads)
        segm_size_nodes = node_features[:, 1].astype('int')
        segm_labels_nodes = node_features[:,0].astype('int')


        uv_ids = rag.uvIds()
        edge_weights = np.ones(uv_ids.shape[0]) if self.train_correct_predictions else np.zeros(uv_ids.shape[0])
        # Ignore-label:
        ignore_mask = np.logical_and(GT_labels_nodes[uv_ids[:, 0]] != 0, GT_labels_nodes[uv_ids[:, 1]] != 0)


        if self.train_merge_mistakes:
            wrong_merge_condition = np.logical_and(GT_labels_nodes[uv_ids[:,0]] != GT_labels_nodes[uv_ids[:,1]],
                                       segm_labels_nodes[uv_ids[:, 0]] == segm_labels_nodes[uv_ids[:, 1]])

            edge_weights = np.where(wrong_merge_condition,
                                 1 + np.minimum(GT_size_nodes[uv_ids[:,0]], GT_size_nodes[uv_ids[:,1]]) * self.weighting_merge_mistakes,
                                 edge_weights)
            edge_weights *= ignore_mask
            # print("merges: {} ({})".format(wrong_merge_condition.sum(), edge_weights.max()), end='; ')


        if self.train_split_mistakes:
            wrong_splits_condition = np.logical_and(GT_labels_nodes[uv_ids[:, 0]] == GT_labels_nodes[uv_ids[:, 1]],
                                                   segm_labels_nodes[uv_ids[:, 0]] != segm_labels_nodes[uv_ids[:, 1]])
            edge_weights = np.where(wrong_splits_condition,
                                    1 + np.minimum(segm_size_nodes[uv_ids[:, 0]],
                                                   segm_size_nodes[uv_ids[:, 1]]) * self.weighting_split_mistakes,
                                    edge_weights)
            edge_weights *= ignore_mask
            # print("splits: {} ({})".format(wrong_splits_condition.sum(), edge_weights.max()))

        loss_weights = map_edge_features_to_image(self.offsets, np.expand_dims(edge_weights, -1),
                                                  rag=rag,
                                   channel_affs=0, fillValue=0.,
                                   number_of_threads=self.number_of_threads)[...,0]
        return loss_weights


class OverSegmentationAgglomeration(Transform):
    def __init__(self, path=None,
                 prob_agglomeration=0.3,
                 max_threshold=1.0,
                 min_threshold=0.5,
                 allow_wrong_merges=False,
                 allow_wrong_splits=False,
                 flip_probability=0.03,
                 number_of_threads=8,
                 minimum_sigma=0.05,
                 scale_factor_sigma=0.7,
                 **super_kwargs):
        assert isinstance(path, str)
        self.number_of_threads = number_of_threads

        # Expected data in the array: u, v, edge_indicator, edge_weight (shape --> (N, 4))
        if path is not None:
            if '$' in path:
                edge_data, cantor_ids = [], []
                for sample in ['A', 'B', 'C']:
                    sample_path = path.replace('$', sample)
                    edge_data.append(vigra.readHDF5(sample_path, pathInFile='edge_data').astype(np.float32))
                    cantor_ids.append(vigra.readHDF5(sample_path, pathInFile='cantor_ids').astype(np.uint64))
                edge_data = np.concatenate(edge_data, axis=0)
                cantor_ids = np.concatenate(cantor_ids, axis=0)
                sort_indices = np.argsort(cantor_ids)
                # TODO: check not to have double IDs...!
                self.edge_data = edge_data[sort_indices]
                self.cantor_ids = cantor_ids[sort_indices]
            else:
                self.edge_data = vigra.readHDF5(path, pathInFile='edge_data').astype(np.float32)
                self.cantor_ids = vigra.readHDF5(path, pathInFile='cantor_ids').astype(np.uint64)
        else:
            self.edge_data = None
            print("Path not given. Initial agglomeration skipped.")

        self.prob_agglomeration = prob_agglomeration
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.flip_probability = flip_probability
        self.allow_wrong_merges= allow_wrong_merges
        self.allow_wrong_splits= allow_wrong_splits
        self.minimum_sigma= minimum_sigma
        self.scale_factor_sigma= scale_factor_sigma

        super(OverSegmentationAgglomeration, self).__init__(**super_kwargs)

    def build_random_variables(self
                               ):
        np.random.seed()
        self.set_random_variable('agglomerate_oversegmentation',
                                 np.random.rand() <= self.prob_agglomeration)
        self.set_random_variable('threshold',
                                 np.random.uniform(self.min_threshold, self.max_threshold))


    def tensor_function(self, tensor_):
        """
        """
        if self.edge_data is None:
            return tensor_

        tensor = tensor_.astype(np.uint32)
        assert tensor_.ndim == 3

        self.build_random_variables()

        agglomerate_oversegmentation = self.get_random_variable('agglomerate_oversegmentation')

        if agglomerate_oversegmentation:
            # print("Running initial clustering...")
            # tick = time.time()

            rag = nrag.gridRag(tensor, self.number_of_threads)

            uvIds = np.sort(rag.uvIds(), axis=1)
            cantor_coeff = cantor_pairing_fct(uvIds[:, 0], uvIds[:, 1])

            edge_indices = np.searchsorted(self.cantor_ids, cantor_coeff)
            # Due to elastic transform, there could be some edges that are not found (tiny
            # pixels, probably). In that case set affinity to 0.:
            edges_not_found = cantor_coeff != self.cantor_ids[edge_indices]
            selected_edges = self.edge_data[edge_indices]
            selected_edges[edges_not_found] = np.tile(np.array([0., 1.]), (edges_not_found.sum(), 1))
            if edges_not_found.sum() > 30:
                print("WARNING: {} edges where not found out of {}".format(edges_not_found.sum(), edges_not_found.shape))
            # Edge indicators should be affinities (merge: 1.0; split: 0.0)
            edge_indicators = selected_edges[:, 0]
            edge_sizes = selected_edges[:, 1]


            # # Perform random flips:
            # if self.wrong_split_flips or self.wrong_merge_flips:
            #     random_probs = np.random.uniform(low=0., high=1., size=edge_indicators.shape)
            #     edges_to_flip = random_probs < self.flip_probability
            #     if self.wrong_merge_flips and not self.wrong_split_flips:
            #         # Only allow 'false-merge' flips:
            #         edges_to_flip = np.logical_and(edges_to_flip, edge_indicators < 0.5)
            #     elif self.wrong_split_flips and not self.wrong_merge_flips:
            #         # Only allow 'false-split' flips:
            #         edges_to_flip = np.logical_and(edges_to_flip, edge_indicators > 0.5)
            #     print("Nb. flips: {} out of {}".format(edges_to_flip.sum(), edges_to_flip.shape))
            #     edge_indicators[edges_to_flip] = 1 - edge_indicators[edges_to_flip]

            # Add "smart" noise to the edge indicators:
            if self.allow_wrong_splits or self.allow_wrong_merges:
                # Get uncertainty prediction as absolute difference from 0.5:
                # TODO: collect statistics during the avg-accumulation (better idea of the uncertainty)
                sigma =  self.minimum_sigma + (0.5 - np.abs(edge_indicators - 0.5))*self.scale_factor_sigma
                noise = np.random.normal(scale=sigma, size=edge_indicators.shape)
                if not self.allow_wrong_splits:
                    # Get rid of noise that decrease the affinities:
                    noise[noise < 0.] = 0.
                elif not self.allow_wrong_merges:
                    # Get rid of noise that increase the affinities:
                    noise[noise > 0.] = 0.
                edge_indicators = np.clip(edge_indicators + noise, a_min=0., a_max=1.)

            node_sizes = np.ones((rag.numberOfNodes ,))
            is_local_edge = np.ones_like(edge_indicators)
            threshold = self.get_random_variable('threshold')
            cluster_policy = nagglo.fixationClusterPolicy(graph=rag,
                                                          mergePrios=edge_indicators,
                                                          notMergePrios=1 - edge_indicators,
                                                          edgeSizes=edge_sizes,
                                                          nodeSizes=node_sizes,
                                                          isMergeEdge=is_local_edge,
                                                          updateRule0=nagglo.updatRule('mean'),
                                                          updateRule1=nagglo.updatRule('mean'),
                                                          zeroInit=False,
                                                          sizeRegularizer=0.,
                                                          sizeThreshMin=0.,
                                                          sizeThreshMax=1000.,
                                                          postponeThresholding=False,
                                                          threshold=threshold,
                                                          )
            agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)


            agglomerativeClustering.run(False, 1000)  # (True, 10000)
            node_labels = agglomerativeClustering.result()

            final_segm = map_features_to_label_array(
                tensor,
                np.expand_dims(node_labels, axis=-1),
                number_of_threads=3
            )[..., 0]

            final_segm, _, _ = vigra.analysis.relabelConsecutive(final_segm.astype('uint32'))

            # print("Took {} s!".format(time.time() - tick))

            return final_segm.astype('float32')

        else:
            return tensor_