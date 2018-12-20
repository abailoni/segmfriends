import time

import numpy as np
from nifty.graph import rag as nrag

from ..features import accumulate_affinities_on_graph_edges
from ..utils.graph import build_lifted_graph_from_rag


class FeaturerLongRangeAffs(object):
    def __init__(self, offsets,
                       offsets_weights=None,
                       used_offsets=None,
                       debug=True,
                       n_threads=1,
                   invert_affinities=False,
                 statistic='mean',
                 max_distance_lifted_edges=1,
                 return_dict=False):

        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.offsets_weights = offsets_weights
        self.statistic = statistic


        assert isinstance(n_threads, int)
        assert isinstance(max_distance_lifted_edges, int)

        self.offsets = offsets
        self.debug = debug
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.max_distance_lifted_edges = max_distance_lifted_edges


    def __call__(self, affinities, segmentation):
        tick = time.time()
        offsets = self.offsets
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        assert affinities.ndim == 4
        # affinities = affinities[:3]
        assert affinities.shape[0] == offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        # Build rag and compute node sizes:
        if self.debug:
            print("Computing rag...")
            tick = time.time()
        rag = nrag.gridRag(segmentation.astype(np.uint32))

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        if self.max_distance_lifted_edges != 1:
            # Build lifted graph:
            print("Building graph...")
            lifted_graph, is_local_edge = build_lifted_graph_from_rag(
                rag,
                segmentation,
                offsets,
                max_lifted_distance=self.max_distance_lifted_edges,
                number_of_threads=self.n_threads)

            # lifted_graph, is_local_edge, _, edge_sizes = build_pixel_lifted_graph_from_offsets(
            #     segmentation.shape,
            #     offsets,
            #     label_image=segmentation,
            #     offsets_weights=None,
            #     nb_local_offsets=3,
            #     GT_label_image=None
            # )

            if self.debug:
                print("Took {} s!".format(time.time() - tick))
                print("Computing edge_features...")
                tick = time.time()

            # Compute edge sizes and accumulate average/max:
            edge_indicators, edge_sizes = \
                accumulate_affinities_on_graph_edges(
                    affinities, offsets,
                    graph=lifted_graph,
                    label_image=segmentation,
                    use_undirected_graph=True,
                    mode=self.statistic,
                    offsets_weights=offsets_weights,
                    number_of_threads=self.n_threads)
            out_dict['graph'] = lifted_graph
            out_dict['edge_indicators'] = edge_indicators
            out_dict['edge_sizes'] = edge_sizes
        else:
            out_dict['graph'] = rag
            print("Computing edge_features...")
            is_local_edge = np.ones(rag.numberOfEdges, dtype=np.int8)
            # TODO: her we have rag (no need to pass egm.), but fix nifty function first.
            if self.statistic == 'mean':
                edge_indicators, edge_sizes = \
                    accumulate_affinities_on_graph_edges(
                        affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                out_dict['edge_indicators'] = edge_indicators
                # out_dict['merge_prio'] = edge_indicators
                # out_dict['not_merge_prio'] = 1 - edge_indicators
                out_dict['edge_sizes'] = edge_sizes
            elif self.statistic == 'max':
                # DEPRECATED
                merge_prio, edge_sizes = \
                    accumulate_affinities_on_graph_edges(
                        affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                not_merge_prio, _ = \
                    accumulate_affinities_on_graph_edges(
                        1 - affinities, offsets,
                        graph=rag,
                        label_image=segmentation,
                        use_undirected_graph=True,
                        mode=self.statistic,
                        offsets_weights=offsets_weights,
                        number_of_threads=self.n_threads)
                edge_indicators = merge_prio
                out_dict['edge_indicators'] = merge_prio
                # out_dict['merge_prio'] = merge_prio
                # out_dict['not_merge_prio'] = not_merge_prio
                out_dict['edge_sizes'] = edge_sizes
            else:
                raise NotImplementedError


        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            # NOTE: lifted graph is not returned!
            return rag, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


