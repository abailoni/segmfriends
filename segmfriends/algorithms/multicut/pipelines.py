import nifty.graph.rag as nrag
import numpy as np

# from ..features import LocalAffinityFeatures, LiftedAffinityFeatures, BoundaryMapFeatures
from .multicut import multicut, lifted_multicut
from ..segm_pipeline import SegmentationPipeline

from ...features import probs_to_costs
from ...features import mappings
import time

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
        featurer_outputs = self.featurer(affinities, segmentation)

        edge_indicators = featurer_outputs['edge_indicators']
        edge_sizes = featurer_outputs['edge_sizes']
        graph = featurer_outputs['graph']

        # this might happen in weird cases when our watershed predicts one single region
        # -> if this happens in one of the first validation runs this is no reason to worry
        if graph.numberOfEdges == 0:
            print("Valdidation stopped because we have no graph edges")
            return np.zeros_like(segmentation, dtype='uint32')

        # edge_features = edge_features[self.stat_id]
        costs = probs_to_costs(edge_indicators, beta=self.beta,
                               weighting_scheme=self.weighting_scheme,
                               rag=graph, segmentation=segmentation,
                               edge_sizes=edge_sizes,
                               weight=self.weight)
        tick = time.time()
        node_labels = multicut(graph, graph.numberOfNodes, graph.uvIds(), costs, self.time_limit, solver_type=self.solver_type,
                               verbose_visitNth=100000000)
        runtime = time.time() - tick

        # Map back segmentation to pixels:
        final_segm = mappings.map_features_to_label_array(
            segmentation,
            np.expand_dims(node_labels, axis=-1),
            number_of_threads=1,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)
        # Increase by one, so ignore label becomes 0:
        final_segm += 1

        # Compute MC energy:
        edge_labels = graph.nodesLabelsToEdgeLabels(node_labels)

        out_dict = {}
        out_dict['MC_energy'] = (costs * edge_labels).sum()
        out_dict['runtime'] = runtime
        return final_segm, out_dict


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

