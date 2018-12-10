from nifty.graph import rag as nrag
import nifty.graph.agglo as nagglo
import numpy as np

import segmfriends.features.mappings
import segmfriends.features.vigra_feat
from ...features.featurer import FeaturerLongRangeAffs
from ...utils.graph import build_pixel_lifted_graph_from_offsets
from ..segm_pipeline import SegmentationPipeline
from ...features.utils import probs_to_costs

import time

class FixationAgglomerativeClustering(SegmentationPipeline):
    def __init__(self, offsets, fragmenter=None,
                 update_rule_merge='mean', update_rule_not_merge='mean',
                 zero_init=False,
                 initSignedWeights=False,
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
            agglomerater = FixationAgglomeraterFromSuperpixels(
                offsets,
                max_distance_lifted_edges=max_distance_lifted_edges,
                used_offsets=used_offsets,
                offsets_weights=offsets_weights,
                update_rule_merge=update_rule_merge,
                update_rule_not_merge=update_rule_not_merge,
                zero_init=zero_init,
                initSignedWeights=initSignedWeights,
                n_threads=n_threads,
                invert_affinities=invert_affinities,
                extra_aggl_kwargs=extra_aggl_kwargs,
                extra_runAggl_kwargs=extra_runAggl_kwargs
            )
            super(FixationAgglomerativeClustering, self).__init__(fragmenter, agglomerater, **super_kwargs)
        else:
            agglomerater = FixationAgglomerater(
                offsets,
                used_offsets=used_offsets,
                update_rule_merge=update_rule_merge,
                update_rule_not_merge=update_rule_not_merge,
                zero_init=zero_init,
                initSignedWeights=initSignedWeights,
                n_threads=n_threads,
                offsets_probabilities=offsets_probabilities,
                offsets_weights=offsets_weights,
                invert_affinities=invert_affinities,
                extra_aggl_kwargs=extra_aggl_kwargs,
                extra_runAggl_kwargs=extra_runAggl_kwargs
            )
            super(FixationAgglomerativeClustering, self).__init__(agglomerater, **super_kwargs)



class FixationAgglomeraterBase(object):
    def __init__(self, offsets, used_offsets=None,
                 update_rule_merge='mean', update_rule_not_merge='mean',
                 zero_init=False,
                 initSignedWeights=False,
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

        self.passed_rules = [update_rule_merge, update_rule_not_merge]
        self.update_rules = [self.parse_update_rule(rule) for rule in self.passed_rules]

        assert isinstance(zero_init, bool)
        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.debug = debug
        self.zeroInit = zero_init
        self.initSignedWeights = initSignedWeights
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.extra_aggl_kwargs = extra_aggl_kwargs if extra_aggl_kwargs is not None else {}
        self.extra_runAggl_kwargs = extra_runAggl_kwargs if extra_runAggl_kwargs is not None else {}

    def parse_update_rule(self, rule):
        accepted_rules_1 = ['max', 'min', 'mean', 'ArithmeticMean', 'sum']
        accepted_rules_2 = ['generalized_mean', 'rank', 'smooth_max']
        if not isinstance(rule, str):
            rule = rule.copy()
            assert isinstance(rule, dict)
            rule_name = rule.pop('name')
            p = rule.get('p')
            q = rule.get('q')
            assert rule_name in accepted_rules_1 + accepted_rules_2
            assert not (p is None and q is None)
            parsed_rule = nagglo.updatRule(rule_name, **rule)
        else:
            assert rule in accepted_rules_1
            parsed_rule = nagglo.updatRule(rule)

        return parsed_rule

    def __getstate__(self):
        state_dict = dict(self.__dict__)
        state_dict.pop('update_rules', None)
        return state_dict

    def __setstate__(self, state_dict):
        if 'passed_rules' in state_dict:
            state_dict['update_rules'] = [self.parse_update_rule(rule) for rule in state_dict['passed_rules']]
        else:
            print (key for key in state_dict)
            raise DeprecationWarning()

        self.__dict__.update(state_dict)



class FixationAgglomeraterFromSuperpixels(FixationAgglomeraterBase):
    def __init__(self, *super_args, max_distance_lifted_edges=3,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(FixationAgglomeraterFromSuperpixels, self).__init__(*super_args, **super_kwargs)
        assert isinstance(max_distance_lifted_edges, int)
        self.max_distance_lifted_edges = max_distance_lifted_edges

        if all([rule == 'max' for rule in self.passed_rules]):
            accumulate_statistic = 'max'
        else:
            accumulate_statistic = 'mean'

        self.featurer = FeaturerLongRangeAffs(self.offsets,
                                              self.offsets_weights,
                                              self.used_offsets,
                                              self.debug,
                                              self.n_threads,
                                              self.invert_affinities,
                                              statistic=accumulate_statistic,
                                              max_distance_lifted_edges=self.max_distance_lifted_edges,
                                              return_dict=True)



    def __call__(self, affinities, segmentation):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        tick = time.time()
        featurer_outputs = self.featurer(affinities, segmentation)

        lifted_graph = featurer_outputs['graph']
        if 'merge_prio' in featurer_outputs:
            merge_prio = featurer_outputs['merge_prio']
            not_merge_prio = featurer_outputs['not_merge_prio']

        else:
            edge_indicators = featurer_outputs['edge_indicators']
            merge_prio = edge_indicators
            not_merge_prio = 1. - edge_indicators

        # FIXME: set edge_sizes to rag!!!
        edge_sizes = featurer_outputs['edge_sizes']
        is_local_edge = featurer_outputs['is_local_edge']



        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Computing node_features...")
            tick = time.time()

        node_sizes = np.squeeze(segmfriends.features.vigra_feat.accumulate_segment_features_vigra(segmentation,
                                                                                                  segmentation, statistics=["Count"],
                                                                                                  normalization_mode=None, map_to_image=False))

        cluster_policy = nagglo.fixationClusterPolicy(graph=lifted_graph,
                                                      mergePrios=merge_prio,
                                                      notMergePrios=not_merge_prio,
                                                      edgeSizes=edge_sizes,
                                                      nodeSizes=node_sizes,
                                                      isMergeEdge=is_local_edge,
                                                      updateRule0=self.update_rules[0],
                                                      updateRule1=self.update_rules[1],
                                                      zeroInit=self.zeroInit,
                                                      initSignedWeights=self.initSignedWeights,
                                                      sizeRegularizer=self.extra_aggl_kwargs.get('sizeRegularizer', 0.),
                                                      sizeThreshMin=self.extra_aggl_kwargs.get('sizeThreshMin', 0.),
                                                      sizeThreshMax=self.extra_aggl_kwargs.get('sizeThreshMax', 300.),
                                                      postponeThresholding=self.extra_aggl_kwargs.get('postponeThresholding', True),
                                                      costsInPQ=self.extra_aggl_kwargs.get('costInPQ', False),
                                                      checkForNegCosts=self.extra_aggl_kwargs.get('checkForNegCosts', True),
                                                      threshold=self.extra_aggl_kwargs.get('threshold', 0.5),
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

        final_segm = segmfriends.features.mappings.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=self.n_threads
        )[..., 0]

        return final_segm






class FixationAgglomerater(FixationAgglomeraterBase):
    def __init__(self, *super_args, offsets_probabilities=None,
                 **super_kwargs):
        """
        Note that the initial SP accumulation at the moment is always given
        by an average!
        """
        super(FixationAgglomerater, self).__init__(*super_args, **super_kwargs)

        self.offsets_probabilities = offsets_probabilities


    def __call__(self, affinities):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        offsets = self.offsets
        offsets_rpobabilities = self.offsets_probabilities
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            offsets_rpobabilities = self.offsets_probabilities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        assert affinities.ndim == 4
        assert affinities.shape[0] == offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        image_shape = affinities.shape[1:]

        # print("Offset avg-weights: ", offsets_weights)

        # Build graph:
        graph, is_local_edge, _, edge_sizes = \
            build_pixel_lifted_graph_from_offsets(
                image_shape,
                offsets,
                offsets_probabilities=offsets_rpobabilities,
                offsets_weights=offsets_weights,
                nb_local_offsets=3
            )
        # print("Number of edges in graph", graph.numberOfEdges)
        # print("Number of nodes in graph", graph.numberOfNodes)
        # print("Local edges:")

        # Build policy:
        # edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
        node_sizes = np.ones(graph.numberOfNodes, dtype='float32')
        edge_weights = graph.edgeValues(np.rollaxis(affinities, 0, 4))

        # # Cost setup A:
        new_aff = edge_weights / 2.00
        new_aff[np.logical_not(is_local_edge)] -= 0.5
        new_aff += 0.5

        # # Cost setup B:
        # new_aff = edge_weights.copy()
        # is_long_range_edge = np.logical_not(is_local_edge)
        # new_aff[is_local_edge][new_aff[is_local_edge] < 0.5] = 0.5
        # new_aff[is_long_range_edge][new_aff[is_long_range_edge] > 0.5] = 0.5

        log_costs = probs_to_costs(1 - new_aff, beta=0.5)
        log_costs = log_costs * edge_sizes / edge_sizes.max()
        if self.extra_aggl_kwargs.get('useLogCosts', False):
            # merge_prio -= 4.0

            merge_prio = log_costs.copy()
            not_merge_prio = - log_costs
            negative_edges = log_costs < 0.
            merge_prio[negative_edges] = -1.
            not_merge_prio[np.logical_not(negative_edges)] = -1.

            # Avoid pixel splits:
            # not_merge_prio[is_local_edge==1] = -1.
        else:
            merge_prio = edge_weights
            not_merge_prio = 1. - merge_prio

            # MWS setup with repulsive lifted edges:
            merge_prio[np.logical_not(is_local_edge)] = -1.
            not_merge_prio[is_local_edge] = -1.

        cluster_policy = nagglo.fixationClusterPolicy(graph=graph,
                              mergePrios=merge_prio, notMergePrios=not_merge_prio,
                              edgeSizes=edge_sizes, nodeSizes=node_sizes,
                              isMergeEdge=is_local_edge,
                              updateRule0=self.update_rules[0],
                              updateRule1=self.update_rules[1],
                              zeroInit=self.zeroInit,
                              initSignedWeights=self.initSignedWeights,
                              sizeRegularizer=self.extra_aggl_kwargs.get('sizeRegularizer', 0.),
                              sizeThreshMin=self.extra_aggl_kwargs.get('sizeThreshMin', 0.),
                              sizeThreshMax=self.extra_aggl_kwargs.get('sizeThreshMax', 300.),
                              postponeThresholding=self.extra_aggl_kwargs.get(
                                                          'postponeThresholding', True),
                                                      costsInPQ=self.extra_aggl_kwargs.get('costInPQ', False),
                                                      checkForNegCosts=self.extra_aggl_kwargs.get('checkForNegCosts',
                                                                                                  True),
                              threshold=self.extra_aggl_kwargs.get('threshold', 0.5),
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
            segmfriends.features.mappings.map_features_to_label_array(edge_IDs, np.expand_dims(mergeTimes, axis=-1)))

        nodeSeg = agglomerativeClustering.result()


        edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
        MC_energy = (log_costs * edge_labels).sum()
        print("MC energy: {}".format(MC_energy))

        segmentation = nodeSeg.reshape(image_shape)

        return segmentation, final_UCM



