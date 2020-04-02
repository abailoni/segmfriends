import time

import numpy as np
from nifty.graph import rag as nrag

from ..features import accumulate_affinities_on_graph_edges
from ..utils.graph import build_lifted_graph_from_rag


def get_rag(segmentation, nb_threads):
    # Check if the segmentation has a background label that should be ignored in the graph:
    min_label = segmentation.min()
    if min_label >=0:
        return  nrag.gridRag(segmentation.astype(np.uint32), numberOfThreads=nb_threads), False
    else:
        assert min_label == -1, "The only accepted background label is -1"
        max_valid_label = segmentation.max()
        assert max_valid_label >= 0, "A label image with only background label was passed!"
        mod_segmentation = segmentation.copy()
        background_mask = segmentation == min_label
        mod_segmentation[background_mask] = max_valid_label + 1

        # Build rag including background:
        return nrag.gridRag(mod_segmentation.astype(np.uint32), numberOfThreads=nb_threads), True


class FeaturerLongRangeAffs(object):
    def __init__(self, offsets,
                       offsets_weights=None,
                       used_offsets=None,
                       debug=True,
                       n_threads=1,
                   invert_affinities=False,
                 statistic='mean',
                 offset_probabilities=None,
                 return_dict=False,
                 mask_used_edges=None):

        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.offsets_weights = offsets_weights
        self.statistic = statistic


        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.debug = debug
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.offset_probabilities = offset_probabilities
        self.mask_used_edges = mask_used_edges


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

        # If there was a label -1, now its value in the rag is given by the maximum label (and it will be ignored later on)
        rag, has_background_label = get_rag(segmentation, self.n_threads)

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        # Compute node_sizes:
        # TODO: better compute node sizes with vigra..?
        # FIXME: this won't work
        # fake_data = np.empty(rag.shape, dtype='float32')
        # out_dict['node_sizes'] = nrag.accumulateMeanAndLength(rag, fake_data)[1][:, 1]
        out_dict['node_sizes'] = None

        # TODO: add bck option to use only local edges
        # if self.max_distance_lifted_edges != 1:
        # Build lifted graph:
        if self.debug:
            print("Building graph...")
        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            segmentation,
            offsets,
            offset_probabilities=self.offset_probabilities,
            number_of_threads=self.n_threads,
            has_background_label=has_background_label,
            mask_used_edges=self.mask_used_edges
        )

        if self.debug:
            print("Took {} s!".format(time.time() - tick))
            print("Computing edge_features...")
            tick = time.time()

        # Compute edge sizes and accumulate average/max:
        edge_indicators, edge_sizes = \
            accumulate_affinities_on_graph_edges(
                affinities, offsets,
                label_image=segmentation,
                graph=lifted_graph,
                mode=self.statistic,
                offsets_weights=offsets_weights,
                number_of_threads=self.n_threads)
        out_dict['graph'] = lifted_graph
        out_dict['edge_indicators'] = edge_indicators
        out_dict['edge_sizes'] = edge_sizes

        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            if isinstance(self.offset_probabilities, np.ndarray):
                assert all(self.offset_probabilities == 0.), "For the moment only local edges supported"
            else:
                assert self.offset_probabilities == 0., "For the moment only local edges supported"
            # FIXME: multicut complains if it receices a graph instead of a rag
            return rag, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


