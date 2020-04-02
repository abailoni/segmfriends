from nifty.graph import rag as nrag
import nifty.graph.agglo as nagglo
import numpy as np
from copy import deepcopy

from ...features import mappings
from ...features import vigra_feat
from ...features.featurer import FeaturerLongRangeAffs
from ...utils.graph import build_pixel_lifted_graph_from_offsets
from ..segm_pipeline import SegmentationPipeline
from ...features.utils import probs_to_costs
from ...transform.segm_to_bound import compute_mask_boundaries_graph

from nifty.segmentation import compute_mws_clustering, compute_single_linkage_clustering

import time

# TODO: rename to GASP

class GreedyEdgeContractionClustering(SegmentationPipeline):
    def __init__(self, offsets, fragmenter=None,
                 offsets_probabilities=None,
                 used_offsets=None,
                 offsets_weights=None,
                 n_threads=1,
                 invert_affinities=False,
                 extra_aggl_kwargs=None,
                 extra_runAggl_kwargs=None,
                 strides=None,
                 return_UCM=False,
                 nb_merge_offsets=3,
                 debug=False,
                 mask_used_edges=None,
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
                used_offsets=used_offsets,
                offsets_weights=offsets_weights,
                n_threads=n_threads,
                offsets_probabilities=offsets_probabilities,
                invert_affinities=invert_affinities,
                extra_aggl_kwargs=extra_aggl_kwargs,
                extra_runAggl_kwargs=extra_runAggl_kwargs,
                nb_merge_offsets=nb_merge_offsets,
                return_UCM=return_UCM,
                debug=debug,
                mask_used_edges=mask_used_edges
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
                nb_merge_offsets=nb_merge_offsets,
                strides=strides,
                return_UCM=return_UCM,
                debug=debug,
                mask_used_edges=mask_used_edges
            )
            super(GreedyEdgeContractionClustering, self).__init__(agglomerater, **super_kwargs)



class GreedyEdgeContractionAgglomeraterBase(object):
    def __init__(self, offsets, used_offsets=None,
                 n_threads=1,
                 invert_affinities=False,
                 offsets_weights=None,
                 extra_aggl_kwargs=None,
                 extra_runAggl_kwargs=None,
                 nb_merge_offsets=3,
                 debug=True,
                 return_UCM=False,
                 offsets_probabilities=None,
                 mask_used_edges=None,
                 ):
        """
                Starts from pixels.

                Examples of accepted update rules:

                 - 'mean'
                 - 'mutex_watershed'
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
        self.return_UCM = return_UCM
        self.invert_affinities = invert_affinities
        self.offset_probabilities = offsets_probabilities
        self.nb_merge_offsets = nb_merge_offsets
        self.extra_aggl_kwargs = deepcopy(extra_aggl_kwargs) if extra_aggl_kwargs is not None else {}
        self.use_log_costs = self.extra_aggl_kwargs.pop('use_log_costs', False)
        self.extra_runAggl_kwargs = extra_runAggl_kwargs if extra_runAggl_kwargs is not None else {}
        self.mask_used_edges = mask_used_edges


class GreedyEdgeContractionAgglomeraterFromSuperpixels(GreedyEdgeContractionAgglomeraterBase):
    def __init__(self, *super_args,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(GreedyEdgeContractionAgglomeraterFromSuperpixels, self).__init__(*super_args, **super_kwargs)

        assert self.nb_merge_offsets == 3, "Other options are not implemented yet"

        self.featurer = FeaturerLongRangeAffs(self.offsets,
                                              self.offsets_weights,
                                              self.used_offsets,
                                              self.debug,
                                              self.n_threads,
                                              self.invert_affinities,
                                              statistic='mean',
                                              offset_probabilities=self.offset_probabilities,
                                              return_dict=True,
                                              mask_used_edges=self.mask_used_edges)



    def __call__(self, affinities, segmentation, previous_uv_ids=None,
                                     previous_edge_weights=None,
                                     previous_edge_sizes=None):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """

        # FIXME: check that invert_affs is not applied twice (in featurer, WSDT and in segm_pipeline)

        tick = time.time()
        # TODO: The accumulate function should really consider the current update rule (!!!)
        featurer_outputs = self.featurer(affinities, segmentation)

        graph = featurer_outputs['graph']
        # if 'merge_prio' in featurer_outputs:
        #     raise DeprecationWarning('Max accumulation no longer supported')
            # merge_prio = featurer_outputs['merge_prio']
            # not_merge_prio = featurer_outputs['not_merge_prio']

        print("Number of edges: ", featurer_outputs['edge_sizes'].shape)

        # FIXME: set edge_sizes to rag!!!
        edge_indicators = featurer_outputs['edge_indicators']
        edge_sizes = featurer_outputs['edge_sizes']
        node_sizes = featurer_outputs['node_sizes']
        is_local_edge = featurer_outputs['is_local_edge']


        if previous_uv_ids is not None:
            assert previous_edge_weights is not None
            previous_edge_sizes = np.ones_like(previous_edge_weights) if previous_edge_sizes is None else previous_edge_sizes
            assert previous_edge_sizes.shape == previous_edge_weights.shape
            assert previous_edge_sizes.shape[0] == previous_uv_ids.shape[0]

            uvIds = graph.uvIds()
            uvIds = np.sort(uvIds, axis=1)
            previous_uv_ids = np.sort(previous_uv_ids, axis=1)

            from ...utils.various import cantor_pairing_fct, search_sorted
            edge_indices = search_sorted(cantor_pairing_fct(uvIds[:, 0], uvIds[:, 1]),
                          cantor_pairing_fct(previous_uv_ids[:,0], previous_uv_ids[:,1]))
            assert np.ma.count_masked(edge_indices) == 0
            assert edge_indices.shape[0] == previous_edge_weights.shape[0]

            # Replace the previously computed edge weights and sizes:
            edge_indicators[edge_indices] = previous_edge_weights
            edge_sizes[edge_indices] = previous_edge_sizes


        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Computing node_features...")
            tick = time.time()


        threshold = self.extra_aggl_kwargs.pop('threshold', 0.5)

        log_costs = probs_to_costs(1 - edge_indicators, beta=threshold)
        # FIXME: CAREFUL! Here we are changing the costs depending on the edge size! Sum or average will give really different results...
        # log_costs = log_costs * edge_sizes / edge_sizes.max()
        if self.use_log_costs:
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - threshold

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Running clustering...")

        node_labels, out_dict = \
            runGreedyGraphEdgeContraction(graph, signed_weights,
                                          edge_sizes=edge_sizes,
                                          node_sizes=node_sizes,
                                          is_merge_edge=is_local_edge,
                                          return_UCM=self.return_UCM,
                                          return_agglomeration_data=True,
                                          **self.extra_aggl_kwargs,
                                          **self.extra_runAggl_kwargs)

        if self.debug:
            print("Took {} s!".format(out_dict["runtime"]))
            print("Getting final segm...")

        # ignore_mask = segmentation == -1
        final_segm = mappings.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=self.n_threads,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)
        # Increase by one, so ignore label becomes 0:
        final_segm += 1


        # Compute MC energy:
        edge_labels = graph.nodesLabelsToEdgeLabels(node_labels)
        MC_energy = (log_costs * edge_labels).sum()
        if self.debug:
            print("MC energy: {}".format(MC_energy))

        if self.return_UCM:
            boundary_IDs = compute_mask_boundaries_graph(self.offsets[:3], graph,
                                          segmentation,
                                          return_boundary_IDs=True,
                                          channel_axis=0
                                          )
            out_dict['UCM'] = np.squeeze(
                mappings.map_features_to_label_array(boundary_IDs, np.expand_dims(out_dict['UCM'], axis=-1),
                                                     fill_value=-15.), axis=-1)
            out_dict['mergeTimes'] = np.squeeze(
                mappings.map_features_to_label_array(boundary_IDs, np.expand_dims(out_dict['mergeTimes'], axis=-1),
                                                     fill_value=-15.), axis=-1)



        out_dict['MC_energy'] = MC_energy
        return final_segm, out_dict






class GreedyEdgeContractionAgglomerater(GreedyEdgeContractionAgglomeraterBase):
    def __init__(self, *super_args, strides=None,
                downscaling_factor=None,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(GreedyEdgeContractionAgglomerater, self).__init__(*super_args, **super_kwargs)

        self.impose_local_attraction = self.extra_aggl_kwargs.pop('impose_local_attraction', False)
        self.strides = strides
        self.downscaling_factor = downscaling_factor


    def __call__(self, affinities):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        offsets = self.offsets
        offset_probabilities = self.offset_probabilities
        offsets_weights = self.offsets_weights
        mask_used_edges = self.mask_used_edges
        if self.used_offsets is not None:
            # TODO: move most of this stuff to init!
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            if isinstance(offset_probabilities, np.ndarray):
                offset_probabilities = self.offset_probabilities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            if offsets_weights is not None:
                offsets_weights = offsets_weights[self.used_offsets]

            if self.mask_used_edges is not None:
                mask_used_edges = self.mask_used_edges[self.used_offsets]

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
                offsets_probabilities=offset_probabilities,
                offsets_weights=offsets_weights,
                nb_local_offsets=self.nb_merge_offsets,
                strides=self.strides,
                mask_used_edges=mask_used_edges
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

        threshold = self.extra_aggl_kwargs.pop('threshold', 0.5)

        log_costs = probs_to_costs(1 - edge_weights, beta=threshold)
        log_costs = log_costs * edge_sizes / edge_sizes.max()

        if self.use_log_costs:
            signed_weights = log_costs
        else:
            signed_weights = edge_weights - threshold

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
            # Make sure to keep them in [-0.5, 0.5]:
            signed_weights = (positive_weights + negative_weights) / 2

            # # Setting to zero the weights:
            # positive_weights = signed_weights * np.logical_and(is_local_edge, signed_weights > 0 )
            # negative_weights = signed_weights * np.logical_and(np.logical_not(is_local_edge), signed_weights < 0)
            # signed_weights = positive_weights + negative_weights


        nodeSeg, out_dict = \
            runGreedyGraphEdgeContraction(graph, signed_weights,
                                          edge_sizes=edge_sizes,
                                          is_merge_edge=is_local_edge,
                                         return_UCM=self.return_UCM,
                                          return_agglomeration_data=True,
                                          ignored_edge_weights=ignored_edge_weights,
                                          **self.extra_aggl_kwargs)

        if self.return_UCM:
            edge_IDs = graph.mapEdgesIDToImage()
            UCM = np.squeeze(
                mappings.map_features_to_label_array(edge_IDs, np.expand_dims(out_dict['UCM'], axis=-1), fill_value=-15.), axis=-1)
            mergeTimes = np.squeeze(
                mappings.map_features_to_label_array(edge_IDs, np.expand_dims(out_dict['mergeTimes'], axis=-1), fill_value=-15.), axis=-1).astype('int64')
            out_dict['UCM'] = np.rollaxis(UCM, -1, 0)
            out_dict['mergeTimes'] = np.rollaxis(mergeTimes, -1, 0)


        edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
        if self.impose_local_attraction:
            # Set ignored edge costs to zero:
            log_costs *= np.logical_not(ignored_edge_weights)
        MC_energy = (log_costs * edge_labels).sum()
        if self.debug:
            print("MC energy: {}".format(MC_energy))
            print("Agglomerated in {} s".format(out_dict['runtime']))

        # TODO: map ignore label -1 to 0!
        segmentation = nodeSeg.reshape(image_shape)

        out_dict['MC_energy'] = MC_energy
        return segmentation, out_dict


def runGreedyGraphEdgeContraction(
                          graph,
                          signed_edge_weights,
                          linkage_criteria = 'mean',
                          add_cannot_link_constraints= False,
                          edge_sizes = None,
                          node_sizes = None,
                          is_merge_edge = None,
                          size_regularizer = 0.0,
                          return_UCM = False,
                          return_agglomeration_data=False,
                          ignored_edge_weights = None,
                          **run_kwargs):
    """
    :param ignored_edge_weights: boolean array, if an edge label is True, than the passed signed weight is ignored
            (neither attractive nor repulsive)

    Returns node_labels and runtime. If return_UCM == True, then also returns the UCM and the merging iteration for
    every edge.
    """
    # Legacy:
    if "update_rule" in run_kwargs:
        update_rule = run_kwargs.pop("update_rule")
    else:
        update_rule = linkage_criteria

    if update_rule == 'mutex_watershed' or (update_rule == 'max' and not add_cannot_link_constraints):
    # if False:
        assert not return_UCM
        if is_merge_edge is not None:
            if not is_merge_edge.all():
                print("WARNING: Efficient implementations only works when all edges are mergeable")
        # In this case we use the efficient MWS clustering implementation in affogato:
        nb_nodes = graph.numberOfNodes
        uv_ids = graph.uvIds()
        mutex_edges = signed_edge_weights < 0.

        # if is_merge_edge is not None:
        #     # If we have edges labelled as lifted, they should all be repulsive in this implementation!
        #     if not is_merge_edge.min():
        #         assert all(is_merge_edge == np.logical_not(mutex_edges)), "Affogato MWS cannot enforce local merges!"

        tick = time.time()
        # This function will sort the edges in ascending order, so we transform all the edges to negative values
        if update_rule == 'mutex_watershed':
            nodeSeg = compute_mws_clustering(nb_nodes,
                                   uv_ids[np.logical_not(mutex_edges)],
                                   uv_ids[mutex_edges],
                                   signed_edge_weights[np.logical_not(mutex_edges)],
                                   -signed_edge_weights[mutex_edges])
        else:
            nodeSeg = compute_single_linkage_clustering(nb_nodes,
                                             uv_ids[np.logical_not(mutex_edges)],
                                             uv_ids[mutex_edges],
                                             signed_edge_weights[np.logical_not(mutex_edges)],
                                             -signed_edge_weights[mutex_edges])
        runtime = time.time() - tick
        out_dict = {'runtime': runtime}

        return nodeSeg, out_dict
    else:
        # FIXME: temporary fix for the sum rule
        # if update_rule == 'sum':
        #     signed_edge_weights *= edge_sizes


        cluster_policy = nagglo.get_GASP_policy(graph, signed_edge_weights,
                                                               edge_sizes=edge_sizes,
                                                               linkage_criteria=update_rule,
                                                                linkage_criteria_kwargs = None,
                                                               add_cannot_link_constraints=add_cannot_link_constraints,
                                                               node_sizes=node_sizes,
                                                               is_mergeable_edge=is_merge_edge,
                                                               size_regularizer=size_regularizer,
                                                               )
        agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)

        out_dict = {}


        tick = time.time()
        if not return_UCM:
            agglomerativeClustering.run(**run_kwargs)
        else:
            # TODO: add run_kwargs with UCM
            outputs = agglomerativeClustering.runAndGetMergeTimesAndDendrogramHeight(verbose=False)
            mergeTimes, UCM = outputs
            out_dict['UCM'] = UCM
            out_dict['mergeTimes'] = mergeTimes

        runtime = time.time() - tick

        nodeSeg = agglomerativeClustering.result()
        out_dict['runtime'] = runtime
        if return_agglomeration_data:
            out_dict['agglomeration_data'], out_dict['edge_data_contracted_graph'] = cluster_policy.exportAgglomerationData()
        return nodeSeg, out_dict





