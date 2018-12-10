import numpy as np
from nifty.graph import rag as nrag


def accumulate_affinities_on_graph_edges(affinities, offsets, graph=None, label_image=None,
                                         contractedRag=None,
                                         use_undirected_graph=False,
                                         mode="mean",
                                         number_of_threads=6,
                                         offsets_weights=None):
    # TODO: Create class and generalize...
    """
    Label image or graph should be passed. Using nifty rag or undirected graph.

    :param affinities: expected to have the offset dimension as last/first one
    """
    assert mode in ['mean', 'max'], "Only max and mean are implemented"

    if affinities.shape[-1] != offsets.shape[0]:
        assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
        ndims = affinities.ndim
        # Move first axis to the last dimension:
        affinities = np.rollaxis(affinities, 0, ndims)

    assert label_image is not None or graph is not None
    if contractedRag is not None:
        assert graph is not None

    if offsets_weights is not None:
        if isinstance(offsets_weights, (list, tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == affinities.shape[-1]
        if all([w>=1.0 for w in offsets_weights]):
            # Take the inverse:
            offsets_weights = 1. / offsets_weights
        else:
            assert all([w<=1.0 for w in offsets_weights]) and all([w>=0.0 for w in offsets_weights])
    else:
        offsets_weights = np.ones(affinities.shape[-1])

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

    if not use_undirected_graph:
        if contractedRag is None:
            raise DeprecationWarning("There is some problem in the nifty function...")
            assert mode == 'mean'
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                              affinities.astype(np.float32),
                                                                              offsets.astype(np.int32),
                                                                              offsets_weights.astype(np.float32),
                                                                              number_of_threads)
        else:
            print("Warning: multipleThread option not implemented!")
            assert mode == 'mean'
            accumulated_feat, counts = nrag.accumulateAffinitiesMeanAndLength(graph, contractedRag,
                                                                  affinities.astype(np.float32), offsets.astype(np.int32))
    else:
        assert contractedRag is None, "Not implemented atm"
        assert label_image is not None
        # Here 'graph' is actually a general undirected graph (thus label image is needed):
        accumulated_feat, counts, max_affinities = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                          label_image.astype(np.int32),
                                                                          affinities.astype(np.float32),
                                                                          offsets.astype(np.int32),
                                                                          offsets_weights.astype(np.float32),
                                                                          number_of_threads)
    if mode == 'mean':
        return accumulated_feat, counts
    elif mode == 'max':
        return accumulated_feat, max_affinities