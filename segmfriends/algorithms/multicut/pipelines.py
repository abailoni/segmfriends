import nifty.graph.rag as nrag
import numpy as np

# from ..features import LocalAffinityFeatures, LiftedAffinityFeatures, BoundaryMapFeatures
from .multicut import multicut, lifted_multicut
from ..segm_pipeline import SegmentationPipeline

from ...features import probs_to_costs

# FIXME median looks much worse than mean !
# is it broken ?!
STAT_TO_INDEX = {'mean': 0,
                 'min': 2,
                 'q10': 3,
                 'q25': 4,
                 'q50': 5,
                 'q75': 6,
                 'q90': 7,
                 'max': 8,
                 'median': 5}


class Multicut(object):
    def __init__(self, featurer, edge_statistic, weighting_scheme, weight, time_limit=None,
                 beta=0.5, # 0.0 merge everything, 1.0 split everything
                 solver_type='kernighanLin',
                 verbose_visitNth=100000000):
        assert edge_statistic in STAT_TO_INDEX, str(edge_statistic)
        assert weighting_scheme in ("all", "z", "xyz", None), str(weighting_scheme)
        assert isinstance(weight, float), str(weight)
        self.featurer = featurer
        self.stat_id = STAT_TO_INDEX[edge_statistic]
        self.weighting_scheme = weighting_scheme
        self.weight = weight
        self.time_limit = time_limit
        self.solver_type = solver_type
        self.verbose_visitNth = verbose_visitNth
        self.beta = beta

    def __call__(self, affinities, segmentation):
        rag, edge_features = self.featurer(affinities, segmentation)

        # this might happen in weird cases when our watershed predicts one single region
        # -> if this happens in one of the first validation runs this is no reason to worry
        if rag.numberOfEdges == 0:
            print("Valdidation stopped because we have no graph edges")
            return np.zeros_like(segmentation, dtype='uint32')

        edge_features = edge_features[self.stat_id]
        costs = probs_to_costs(edge_features, beta=self.beta,
                               weighting_scheme=self.weighting_scheme,
                               rag=rag, segmentation=segmentation,
                               weight=self.weight)
        node_labels = multicut(rag, rag.numberOfNodes, rag.uvIds(), costs, self.time_limit, solver_type=self.solver_type,
                               verbose_visitNth=100000000)
        edge_labels = rag.nodesLabelsToEdgeLabels(node_labels)
        out_dict = {}
        out_dict['MC_energy'] = (costs * edge_labels).sum()
        return nrag.projectScalarNodeDataToPixels(rag, node_labels), out_dict


class MulticutPipelineFromAffinities(SegmentationPipeline):
    def __init__(self,
                 fragmenter,
                 featurer, edge_statistic, weighting_scheme, weight, time_limit=None,
                 beta=0.5,
                 solver_type='kernighanLin',
                 verbose_visitNth=100000000,
                 **super_kwargs):
        mc = Multicut(featurer, edge_statistic, weighting_scheme, weight, time_limit, beta, solver_type,
                      verbose_visitNth)
        super(MulticutPipelineFromAffinities, self).__init__(fragmenter, mc, **super_kwargs)


# class MulticutPipelineFromLocalAffinities(SegmentationPipeline):
#     def __init__(self,
#                  fragmenter,
#                  edge_statistic='mean',
#                  weighting_scheme='all',
#                  weight=16.,
#                  time_limit=None,
#                  n_threads=1,
#                  **super_kwargs):
#         featurer = LocalAffinityFeatures(n_threads)
#         mc = Multicut(featurer, edge_statistic, weighting_scheme, weight, time_limit)
#         super(MulticutPipelineFromLocalAffinities, self).__init__(fragmenter, mc, **super_kwargs)


# class MulticutPipelineFromBoundaryMaps(SegmentationPipeline):
#     def __init__(self,
#                  fragmenter,
#                  edge_statistic='mean',
#                  weighting_scheme='all',
#                  weight=16.,
#                  time_limit=None,
#                  n_threads=1,
#                  **super_kwargs):
#         featurer = BoundaryMapFeatures(n_threads)
#         mc = Multicut(featurer, edge_statistic, weighting_scheme, weight, time_limit)
#         super(MulticutPipelineFromLocalAffinities, self).__init__(fragmenter, mc, **super_kwargs)
#
#
# class LiftedMulticut(object):
#     def __init__(self,
#                  featurer,
#                  featurer_lifted,
#                  edge_statistic,
#                  weighting_scheme,
#                  weight,
#                  weight_lifted,
#                  time_limit=None):
#         assert edge_statistic in STAT_TO_INDEX, str(edge_statistic)
#         assert weighting_scheme in ("all", "z", "xyz", None), str(weighting_scheme)
#         assert isinstance(weight, float), str(weight)
#         assert isinstance(weight_lifted, float), str(weight_lifted)
#         self.featurer = featurer
#         self.featurer_lifted = featurer_lifted
#         self.stat_id = STAT_TO_INDEX[edge_statistic]
#         self.weighting_scheme = weighting_scheme
#         self.weight = weight
#         self.weight_lifted = weight_lifted
#         self.time_limit = time_limit
#
#     def __call__(self, affinities, segmentation):
#         rag, edge_features = self.featurer(affinities, segmentation)
#
#         # this might happen in weird cases when our watershed predicts one single region
#         # -> if this happens in one of the first validation runs this is no reason to worry
#         if rag.numberOfEdges == 0:
#             print("Valdidation stopped because we have no graph edges")
#             return np.zeros_like(segmentation, dtype='uint32')
#
#         lifted_nh, local_feats, lifted_features = self.featurer_lifted(affinities, rag)
#         # This is just a sanity check that the lifted features do the right thing
#         # assert local_feats.shape == edge_features.shape
#         # close_vals = np.isclose(local_feats, edge_features)
#         # print("Values that are close: %i / %i" % (np.sum(close_vals), local_feats.size))
#         # assert np.allclose(local_feats, edge_features)
#
#         edge_features = edge_features[:, self.stat_id]
#         lifted_features = lifted_features[:, self.stat_id]
#         costs = probs_to_costs(
#             edge_features,
#             weighting_scheme=self.weighting_scheme,
#             rag=rag, segmentation=segmentation,
#             weight=self.weight
#         )
#         lifted_costs = probs_to_costs(lifted_features)
#         if self.weight_lifted != 1:
#             costs *= self.weight_lifted
#
#         node_labels = lifted_multicut(
#             rag.numberOfNodes, rag.uvIds(), costs, lifted_nh, lifted_costs, self.time_limit
#         )
#         return nrag.projectScalarNodeDataToPixels(rag, node_labels)
#
#
# class LiftedMulticutFromAffinities(SegmentationPipeline):
#     def __init__(self,
#                  fragmenter,
#                  axes,
#                  ranges,
#                  edge_statistic='mean',
#                  weighting_scheme='all',
#                  weight=16,
#                  weight_lifted=1.,
#                  time_limit=None,
#                  n_threads=1,
#                  **super_kwargs):
#         featurer = LocalAffinityFeatures(n_threads)
#         lifted_featurer = LiftedAffinityFeatures(axes, ranges, n_threads)
#         lmc = LiftedMulticut(featurer,
#                              lifted_featurer,
#                              edge_statistic,
#                              weighting_scheme,
#                              weight,
#                              weight_lifted,
#                              time_limit)
#         super(LiftedMulticutFromAffinities, self).__init__(fragmenter, lmc, **super_kwargs)
#
#
# class MulticutPipelineLearned(SegmentationPipeline):
#     pass
#
#
# #  TODO TODO TODO
# # # TODO time limit
# # def lifted_multicut_from_lrz_affinities(
# #     affinities,
# #     n_threads,
# #     statistic='mean',
# #     weighting_scheme='xyz',
# #     time_limit=None,
# #     **wsdt_kwargs
# # ):
# #
# #     # make sure that `statistics` exists and get the correct index
# #     assert statistic in STAT_TO_INDEX
# #     stat_id = STAT_TO_INDEX[statistic]
# #
# #     # run watershed and compute the rag
# #     hmap = np.maximum(affinities[1], affinities[2])
# #     seg = watershed_on_distance_transform_2d_stacked(hmap, n_threads=n_threads, **wsdt_kwargs)
# #     rag = nrag.gridRag(seg, n_threads)
# #
# #     # get the proper feature from statistics accumulated over the edges and transform to
# #     # costs
# #     feature = compute_local_features(rag, seg, affinities, n_threads)[:, stat_id]
# #     costs = probs_to_costs(feature, weighting_scheme=weighting_scheme, rag=rag, seg=seg)
# #
# #     lifted_uvs, lifted_features = compute_lrz_costs_and_nh(
# #         seg, affinities, n_threads=n_threads
# #     )
# #     lifted_costs = probs_to_costs(lifted_features[:, stat_id])
# #     # TODO weight costs properly ?!
# #
# #     # run the multicut and project the node labels to a segmentation
# #     node_labels = lifted_multicut(
# #         rag.numberOfNodes, rag.uvIds(), costs, lifted_uvs, lifted_costs
# #     )
# #
# #
# # # TODO
# # # def pipeline_lmc(
# # #     affinities,
# # #     axes,
# # #     ranges,
# # #     n_threads,
# # #     invert_affinities=False,
# # #     statistic='mean',
# # #     weighting_scheme='xyz',
# # #     gamma=1.,
# # #     **wsdt_kwargs
# # # ):
# # #
# # #     # make sure that `statistics` exists and get the correct index
# # #     assert statistic in STAT_TO_INDEX
# # #     stat_id = STAT_TO_INDEX[statistic]
# # #
# # #     # we assume that these will not be needed otherwise,
# # #     # so we change them globally
# # #     if invert_affinities:
# # #         affinities = 1. - affinities
# # #
# # #     # run watershed and compute the rag
# # #     hmap = np.sum(affinities[1:3], axis=0)
# # #     seg = watershed_on_distance_transform_2d_stacked(hmap, n_threads, **wsdt_kwargs)
# # #     rag = nrag.gridRag(seg, n_threads)
# # #
# # #     # get the features from statistics accumulated over all affinity channels
# # #     # as well as the lifted neighborhood, then transform the specified feature to costs
# # #     lifted_nh, local_features, lifted_features = compute_lifted_nh_and_features(
# # #         rag, affinities, axes, ranges, n_threads
# # #     )
# # #     local_costs = probs_to_costs(local_features[:, stat_id], weighting_scheme=weighting_scheme, rag=rag, seg=seg)
# # #     lifted_costs = probs_to_costs(lifted_features[:, stat_id])
# # #
# # #     # TODO weight z - (lifted) edges differently ?!
# # #     # weight the costs against each other and apply gamma
# # #     n_local, n_lifted = len(local_costs), len(lifted_costs)
# # #     local_costs *= (n_local / (n_local + n_lifted))
# # #     lifted_costs *= (n_lifted / (n_local + n_lifted))
# # #     local_costs *= gamma
# # #
# # #     # run the lifted multicut and project node labels to segmentation
# # #     node_labels = lifted_multicut(rag.numberOfNodes, rag.uvIds(), local_costs, lifted_nh, lifted_costs)
# # #     return nrag.projectScalarNodeDataToPixels(rag, node_labels, n_threads)
