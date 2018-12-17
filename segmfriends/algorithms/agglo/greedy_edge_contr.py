from nifty.graph import rag as nrag
import nifty.graph.agglo as nagglo
import numpy as np

from ...features import mappings
from ...features import vigra_feat
from ...features.featurer import FeaturerLongRangeAffs
from ...utils.graph import build_pixel_lifted_graph_from_offsets
from ..segm_pipeline import SegmentationPipeline
from ...features.utils import probs_to_costs

import time

class GreedyEdgeContractionClustering(SegmentationPipeline):
    def __init__(self, offsets, fragmenter=None,
                 max_distance_lifted_edges=3,
                 offsets_probabilities=None,
                 used_offsets=None,
                 offsets_weights=None,
                 n_threads=1,
                 invert_affinities=False,
                 extra_aggl_kwargs=None,
                 extra_runAggl_kwargs=None,
                 **super_kwargs):
        """
        If a fragmenter is passed (DTWS, SLIC, etc...) then the agglomeration is done
        starting from superpixels.

        Alternatively, agglomeration starts from pixels.

        Remarks:
          - the initial SP accumulation at the moment is always given
            by an average!
          - it expects REAL affinities (1.0 = merge, 0. = not merge).
            If the opposite is passed, use `invert_affinities`
        """
        # TODO: add option to pass directly a segmentation (not only affinities)
        if fragmenter is not None:
            agglomerater = GreedyEdgeContractionAgglomeraterFromSuperpixels(
                offsets,
                max_distance_lifted_edges=max_distance_lifted_edges,
                used_offsets=used_offsets,
                offsets_weights=offsets_weights,
                n_threads=n_threads,
                invert_affinities=invert_affinities,
                extra_aggl_kwargs=extra_aggl_kwargs,
                extra_runAggl_kwargs=extra_runAggl_kwargs
            )
            super(GreedyEdgeContractionClustering, self).__init__(fragmenter, agglomerater, **super_kwargs)
        else:
            agglomerater = GreedyEdgeContractionAgglomerater(
                offsets,
                used_offsets=used_offsets,
                n_threads=n_threads,
                offsets_probabilities=offsets_probabilities,
                offsets_weights=offsets_weights,
                invert_affinities=invert_affinities,
                extra_aggl_kwargs=extra_aggl_kwargs,
                extra_runAggl_kwargs=extra_runAggl_kwargs
            )
            super(GreedyEdgeContractionClustering, self).__init__(agglomerater, **super_kwargs)



class GreedyEdgeContractionAgglomeraterBase(object):
    def __init__(self, offsets, used_offsets=None,
                 n_threads=1,
                 invert_affinities=False,
                 offsets_weights=None,
                 extra_aggl_kwargs=None,
                 extra_runAggl_kwargs=None,
                 debug=True,
                 ):
        """
                Starts from pixels.

                Examples of accepted update rules:

                 - 'mean'
                 - 'max'
                 - 'min'
                 - 'sum'
                 - {name: 'rank', q=0.5, numberOfBins=40}
                 - {name: 'generalized_mean', p=2.0}   # 1.0 is mean
                 - {name: 'smooth_max', p=2.0}   # 0.0 is mean

                """
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights

        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.debug = debug
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.extra_aggl_kwargs = extra_aggl_kwargs if extra_aggl_kwargs is not None else {}
        self.use_log_costs = self.extra_aggl_kwargs.pop('use_log_costs', False)
        self.extra_runAggl_kwargs = extra_runAggl_kwargs if extra_runAggl_kwargs is not None else {}


class GreedyEdgeContractionAgglomeraterFromSuperpixels(GreedyEdgeContractionAgglomeraterBase):
    def __init__(self, *super_args, max_distance_lifted_edges=3,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(GreedyEdgeContractionAgglomeraterFromSuperpixels, self).__init__(*super_args, **super_kwargs)
        assert isinstance(max_distance_lifted_edges, int)
        self.max_distance_lifted_edges = max_distance_lifted_edges

        self.featurer = FeaturerLongRangeAffs(self.offsets,
                                              self.offsets_weights,
                                              self.used_offsets,
                                              self.debug,
                                              self.n_threads,
                                              self.invert_affinities,
                                              statistic='mean',
                                              max_distance_lifted_edges=self.max_distance_lifted_edges,
                                              return_dict=True)



    def __call__(self, affinities, segmentation):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        tick = time.time()
        featurer_outputs = self.featurer(affinities, segmentation)

        graph = featurer_outputs['graph']
        if 'merge_prio' in featurer_outputs:
            raise DeprecationWarning('Max accumulation no longer supported')
            # merge_prio = featurer_outputs['merge_prio']
            # not_merge_prio = featurer_outputs['not_merge_prio']


        # FIXME: set edge_sizes to rag!!!
        edge_indicators = featurer_outputs['edge_indicators']
        edge_sizes = featurer_outputs['edge_sizes']
        is_local_edge = featurer_outputs['is_local_edge']



        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Computing node_features...")
            tick = time.time()

        # TODO: atm node sizes are used only with a size regularizer
        # node_sizes = np.squeeze(vigra_feat.accumulate_segment_features_vigra(segmentation,
        #                                                                                           segmentation, statistics=["Count"],
        #                                                                                           normalization_mode=None, map_to_image=False))

        if self.use_log_costs:
            log_costs = probs_to_costs(1 - edge_indicators, beta=0.5)
            log_costs = log_costs * edge_sizes / edge_sizes.max()
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - 0.5

        cluster_policy = nagglo.greedyGraphEdgeContraction(graph, signed_weights,
                                                           edge_sizes=edge_sizes,
                                                           # node_sizes=node_sizes,
                                                           is_merge_edge=is_local_edge,
                                                           **self.extra_aggl_kwargs
                                                           )

        # Run agglomerative clustering:
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Running clustering...")
            tick = time.time()

        agglomerativeClustering.run(**self.extra_runAggl_kwargs) # (True, 10000)
        node_labels = agglomerativeClustering.result()

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Getting final segm...")

        final_segm = mappings.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=self.n_threads
        )[..., 0]

        return final_segm






class GreedyEdgeContractionAgglomerater(GreedyEdgeContractionAgglomeraterBase):
    def __init__(self, *super_args, offsets_probabilities=None,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(GreedyEdgeContractionAgglomerater, self).__init__(*super_args, **super_kwargs)

        self.offsets_probabilities = offsets_probabilities


    def __call__(self, affinities):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        offsets = self.offsets
        offsets_probabilities = self.offsets_probabilities
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            offsets_probabilities = self.offsets_probabilities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        assert affinities.ndim == 4
        assert affinities.shape[0] == offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        image_shape = affinities.shape[1:]

        # Build graph:
        graph, is_local_edge, _, edge_sizes = \
            build_pixel_lifted_graph_from_offsets(
                image_shape,
                offsets,
                offsets_probabilities=offsets_probabilities,
                offsets_weights=offsets_weights,
                nb_local_offsets=3
            )

        # Build policy:
        edge_weights = graph.edgeValues(np.rollaxis(affinities, 0, 4))

        if self.use_log_costs:
            new_aff = edge_weights

            # # Cost setup B:
            # new_aff = edge_weights / 2.00
            # new_aff[np.logical_not(is_local_edge)] -= 0.5
            # new_aff += 0.5

            # # Cost setup C:
            # new_aff = edge_weights.copy()
            # is_long_range_edge = np.logical_not(is_local_edge)
            # new_aff[is_local_edge][new_aff[is_local_edge] < 0.5] = 0.5
            # new_aff[is_long_range_edge][new_aff[is_long_range_edge] > 0.5] = 0.5

            log_costs = probs_to_costs(1 - new_aff, beta=0.5)
            log_costs = log_costs * edge_sizes / edge_sizes.max()

            signed_weights = log_costs
        else:
            signed_weights = edge_weights - 0.5

            # # MWS setup with repulsive lifted edges:
            # positive_weights = edge_weights * is_local_edge
            # negative_weights = (edge_weights - 1.) * np.logical_not(is_local_edge)
            # signed_weights = positive_weights + negative_weights

        cluster_policy = nagglo.greedyGraphEdgeContraction(graph, signed_weights,
                                          edge_sizes=edge_sizes,
                                          is_merge_edge=is_local_edge,
                                          **self.extra_aggl_kwargs
                                          )

        # Run agglomerative clustering:
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
        # agglomerativeClustering.run(**self.extra_runAggl_kwargs)
        outputs = agglomerativeClustering.runAndGetMergeTimesAndDendrogramHeight(verbose=False)
        mergeTimes, UCMap = outputs

        edge_IDs = graph.mapEdgesIDToImage()
        # Take only local:
        edge_IDs = edge_IDs

        final_UCM = np.squeeze(
            mappings.map_features_to_label_array(edge_IDs, np.expand_dims(mergeTimes, axis=-1)))

        nodeSeg = agglomerativeClustering.result()


        edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
        MC_energy = (log_costs * edge_labels).sum()
        print("MC energy: {}".format(MC_energy))

        segmentation = nodeSeg.reshape(image_shape)

        return segmentation, final_UCM



