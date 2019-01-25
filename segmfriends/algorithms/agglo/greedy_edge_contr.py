from nifty.graph import rag as nrag
import nifty.graph.agglo as nagglo
import numpy as np

from ...features import mappings
from ...features import vigra_feat
from ...features.featurer import FeaturerLongRangeAffs
from ...utils.graph import build_pixel_lifted_graph_from_offsets
from ..segm_pipeline import SegmentationPipeline
from ...features.utils import probs_to_costs

from affogato.segmentation import compute_mws_clustering

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
                 strides=None,
                 return_UCM=False,
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
                extra_runAggl_kwargs=extra_runAggl_kwargs,
                strides=strides,
                return_UCM=return_UCM
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
        # if 'merge_prio' in featurer_outputs:
        #     raise DeprecationWarning('Max accumulation no longer supported')
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

        log_costs = probs_to_costs(1 - edge_indicators, beta=0.5)
        log_costs = log_costs * edge_sizes / edge_sizes.max()
        if self.use_log_costs:
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - 0.5

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Running clustering...")

        node_labels, runtime = \
            runGreedyGraphEdgeContraction(graph, signed_weights,
                                          edge_sizes=edge_sizes,
                                          # node_sizes=node_sizes,
                                          is_merge_edge=is_local_edge,
                                          return_UCM=False,
                                          **self.extra_aggl_kwargs,
                                          **self.extra_runAggl_kwargs)

        if self.debug:
            print("Took {} s!".format(runtime))
            print("Getting final segm...")

        final_segm = mappings.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=self.n_threads
        )[..., 0]

        # Compute MC energy:
        edge_labels = graph.nodesLabelsToEdgeLabels(node_labels)
        MC_energy = (log_costs * edge_labels).sum()
        if self.debug:
            print("MC energy: {}".format(MC_energy))

        return final_segm, MC_energy






class GreedyEdgeContractionAgglomerater(GreedyEdgeContractionAgglomeraterBase):
    def __init__(self, *super_args, offsets_probabilities=None, strides=None, return_UCM=False,
                downscaling_factor=None,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(GreedyEdgeContractionAgglomerater, self).__init__(*super_args, **super_kwargs)

        self.impose_local_attraction = self.extra_aggl_kwargs.pop('impose_local_attraction', False)
        self.offsets_probabilities = offsets_probabilities
        self.strides = strides
        self.return_UCM = return_UCM
        self.downscaling_factor = downscaling_factor


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
                nb_local_offsets=3,
                strides=self.strides
            )

        # Build policy:
        edge_weights = graph.edgeValues(np.rollaxis(affinities, 0, 4))

        # Compute log costs:

        # # Cost setup B:
        # new_aff = edge_weights / 2.00
        # new_aff[np.logical_not(is_local_edge)] -= 0.5
        # new_aff += 0.5

        # # Cost setup C:
        # new_aff = edge_weights.copy()
        # is_long_range_edge = np.logical_not(is_local_edge)
        # new_aff[is_local_edge][new_aff[is_local_edge] < 0.5] = 0.5
        # new_aff[is_long_range_edge][new_aff[is_long_range_edge] > 0.5] = 0.5

        log_costs = probs_to_costs(1 - edge_weights, beta=0.5)
        log_costs = log_costs * edge_sizes / edge_sizes.max()

        if self.use_log_costs:
            signed_weights = log_costs
        else:
            signed_weights = edge_weights - 0.5

        ignored_edge_weights = None
        if self.impose_local_attraction:
            # FIXME: only working for max and average at the moment! Not working with the costs
            # # APPROACH 1: NOT WORKING
            # # Ignore local repulsive edges:
            # ignored_edge_weights = np.logical_and(is_local_edge, edge_weights < 0)
            # # Ignore lifted attractive edges:
            # ignored_edge_weights = np.logical_or(np.logical_and(np.logical_not(is_local_edge), edge_weights > 0), ignored_edge_weights)


            # MWS setup with repulsive lifted edges: (this works)
            positive_weights = edge_weights * is_local_edge
            negative_weights = (edge_weights - 1.) * np.logical_not(is_local_edge)
            signed_weights = positive_weights + negative_weights

            # # Setting to zero the weights:
            # positive_weights = signed_weights * np.logical_and(is_local_edge, signed_weights > 0 )
            # negative_weights = signed_weights * np.logical_and(np.logical_not(is_local_edge), signed_weights < 0)
            # signed_weights = positive_weights + negative_weights

        outs = \
            runGreedyGraphEdgeContraction(graph, signed_weights,
                                          edge_sizes=edge_sizes,
                                          is_merge_edge=is_local_edge,
                                         return_UCM=self.return_UCM,
                                          ignored_edge_weights=ignored_edge_weights,
                                          **self.extra_aggl_kwargs)

        if self.return_UCM:
            nodeSeg, runtime, UCM, mergeTimes = outs

            edge_IDs = graph.mapEdgesIDToImage()
            UCM = np.squeeze(
                mappings.map_features_to_label_array(edge_IDs, np.expand_dims(UCM, axis=-1), fill_value=-15.))
            mergeTimes = np.squeeze(
                mappings.map_features_to_label_array(edge_IDs, np.expand_dims(mergeTimes, axis=-1), fill_value=-15.)).astype('int64')
            UCM = np.rollaxis(UCM, -1, 0)
            mergeTimes = np.rollaxis(mergeTimes, -1, 0)


        else:
            nodeSeg, runtime = outs

        edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
        if self.impose_local_attraction:
            # Set ignored edge costs to zero:
            log_costs *= np.logical_not(ignored_edge_weights)
        MC_energy = (log_costs * edge_labels).sum()
        print("MC energy: {}".format(MC_energy))
        print("Agglomerated in {} s".format(runtime))

        segmentation = nodeSeg.reshape(image_shape)

        if self.return_UCM:
            return segmentation, MC_energy, UCM, mergeTimes
        else:
            return segmentation, MC_energy


def runGreedyGraphEdgeContraction(
                          graph,
                          signed_edge_weights,
                          update_rule = 'mean',
                          threshold = 0.5,
                          add_cannot_link_constraints= False,
                          edge_sizes = None,
                          node_sizes = None,
                          is_merge_edge = None,
                          size_regularizer = 0.0,
                          return_UCM = False,
                          ignored_edge_weights = None,
                          remove_small_segments = False,
                          small_segments_thresh = 10,
                          **run_kwargs):
    """
    :param ignored_edge_weights: boolean array, if an edge label is True, than the passed signed weight is ignored
            (neither attractive nor repulsive)

    Returns node_labels and runtime. If return_UCM == True, then also returns the UCM and the merging iteration for
    every edge.
    """
    if update_rule == 'max':
        # In this case we use the efficient MWS clustering implementation in affogato:
        nb_nodes = graph.numberOfNodes
        uv_ids = graph.uvIds()
        mutex_edges = signed_edge_weights < 0.
        tick = time.time()
        # This function will sort the edges in ascending order, so we transform all the edges to negative values
        nodeSeg = compute_mws_clustering(nb_nodes,
                               uv_ids[np.logical_not(mutex_edges)],
                               uv_ids[mutex_edges],
                               signed_edge_weights[np.logical_not(mutex_edges)],
                               -signed_edge_weights[mutex_edges])
        runtime = time.time() - tick
        return nodeSeg, runtime
    else:

        cluster_policy = nagglo.greedyGraphEdgeContraction(graph, signed_edge_weights,
                                                               edge_sizes=edge_sizes,
                                                               update_rule=update_rule,
                                                               threshold=threshold,
                                                               add_cannot_link_constraints=add_cannot_link_constraints,
                                                               node_sizes=node_sizes,
                                                               is_merge_edge=is_merge_edge,
                                                               size_regularizer=size_regularizer,
                                                               ignored_edge_weights=ignored_edge_weights,
                                                                remove_small_segments=remove_small_segments,
                                                                small_segments_thresh=small_segments_thresh
                                                               )
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)

        tick = time.time()
        if not return_UCM:
            agglomerativeClustering.run(**run_kwargs)
        else:
            # TODO: add run_kwargs with UCM
            outputs = agglomerativeClustering.runAndGetMergeTimesAndDendrogramHeight(verbose=False)
            mergeTimes, UCM = outputs

        runtime = time.time() - tick

        nodeSeg = agglomerativeClustering.result()

        if return_UCM:
            return nodeSeg, runtime, UCM, mergeTimes
        else:
            return nodeSeg, runtime





