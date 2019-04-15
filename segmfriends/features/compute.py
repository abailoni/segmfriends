import numpy as np
from nifty.graph import rag as nrag


def accumulate_affinities_on_graph_edges(affinities, offsets, label_image, graph=None,
                                         contractedRag=None, # Deprecated
                                         use_undirected_graph=False, # Ignored
                                         mode="mean",
                                         number_of_threads=6,
                                         offsets_weights=None):
    # TODO: Create class and generalize...
    """
    Label image or graph should be passed. Using nifty rag or undirected graph.

    :param affinities: expected to have the offset dimension as last/first one
    """
    assert mode in ['mean', 'max'], "Only max and mean are implemented"
    assert contractedRag is None, "Contracted graph was deprecated"

    if affinities.shape[-1] != offsets.shape[0]:
        assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
        ndims = affinities.ndim
        # Move first axis to the last dimension:
        affinities = np.rollaxis(affinities, 0, ndims)

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

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



# def accumulate_affinities_on_graph_edges(affinities, offsets, rag=None, label_image=None,
#                                          compute_lifted_edges=False,
#                                          mode="mean",
#                                          number_of_threads=-1,
#                                          offsets_weights=None):
#     # FIXME: offsets_weights are not implemented anymore...
#     # FIXME: it only works for RAG but not a general graph
#     """
#     Standard features:
#
#         0   Mean,
#         1   Variance,
#         2:8 Quantiles: (0%, 10%, 25%, 50%, 75%, 90%, 100%)
#         9   Count
#
#     Label image or rag should be passed.
#
#     :param affinities: expected to have the offset dimension as last/first one
#     """
#     assert mode in ['mean', 'max'], "Only max and mean are implemented"
#
#     if affinities.shape[0] != offsets.shape[0]:
#         assert affinities.shape[-1] == offsets.shape[0], "Offsets do not match passed affs"
#         ndims = affinities.ndim
#         # Move last dimension to the first one:
#         affinities = np.rollaxis(affinities, ndims-1, 0)
#
#     assert label_image is not None or rag is not None
#     assert rag is not None
#
#     if isinstance(offsets, np.ndarray):
#         offsets = [list(of) for of in offsets]
#
#     if offsets_weights is not None:
#         raise NotImplementedError()
#         if isinstance(offsets_weights, (list, tuple)):
#             offsets_weights = np.array(offsets_weights)
#         assert offsets_weights.shape[0] == affinities.shape[0]
#         if all([w>=1.0 for w in offsets_weights]):
#             # Take the inverse:
#             offsets_weights = 1. / offsets_weights
#         else:
#             assert all([w<=1.0 for w in offsets_weights]) and all([w>=0.0 for w in offsets_weights])
#     # else:
#     #     offsets_weights = np.ones(affinities.shape[0])
#
#     if rag is None:
#         rag = nrag.gridRag(label_image.astype(np.uint32))
#
#     if not compute_lifted_edges:
#         features = nrag.accumulateAffinityStandartFeatures(rag,
#                                             affinities.astype(np.float32),
#                                             list(offsets),
#                                             min=0., max=1.,
#                                             numberOfThreads=number_of_threads
#                                             )
#     else:
#         lifted_uvIDs, local_features, lifted_features= nrag.computeFeaturesAndNhFromAffinities(rag,
#                                                                 affinities.astype(np.float32),
#                                                                 offsets.astype(np.int32),
#                                                                 numberOfThreads=number_of_threads
#                                                                 )
#     # TODO: find out outputs...
#     if mode == 'mean':
#         return accumulated_feat, counts
#     elif mode == 'max':
#         return accumulated_feat, max_affinities