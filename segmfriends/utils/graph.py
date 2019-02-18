import time
import nifty
import numpy as np
from nifty import graph as ngraph
from nifty.graph import undirectedLongRangeGridGraph

from ..features import accumulate_affinities_on_graph_edges


def build_lifted_graph_from_rag(rag,
                                label_image,
                                offsets, offset_probabilities=None,
                                nb_offsets_direct_neighbors=3,
                                number_of_threads=8):


    if isinstance(offset_probabilities, np.ndarray):
        only_local = all(offset_probabilities == 0.)
    else:
        only_local = offset_probabilities == 0.

    nb_local_edges = rag.numberOfEdges
    if only_local:
        return rag, np.ones((nb_local_edges, ), dtype='bool')
    else:
        local_edges = rag.uvIds()


        used_offsets = offsets[nb_offsets_direct_neighbors:]

        possibly_lifted_edges = ngraph.compute_lifted_edges_from_rag_and_offsets(rag,
                                                                                 label_image,
                                                  used_offsets,
                                                  offsets_probabilities=offset_probabilities,
                                                  number_of_threads=number_of_threads)

        # print("Creating undirected graph..")

        lifted_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        lifted_graph.insertEdges(local_edges)
        lifted_graph.insertEdges(possibly_lifted_edges)
        total_nb_edges = lifted_graph.numberOfEdges

        is_local_edge = np.zeros(total_nb_edges, dtype=np.int8)
        is_local_edge[:nb_local_edges] = 1

        # print("Local edges:", nb_local_edges)
        # print("Lifted edges:", total_nb_edges - nb_local_edges)

        return lifted_graph, is_local_edge



def build_pixel_lifted_graph_from_offsets(image_shape,
                                          offsets,
                                          label_image=None,
                                          GT_label_image=None,
                                          offsets_probabilities=None,
                                          offsets_weights=None,
                                          strides=None,
                                          nb_local_offsets=3,
                                          downscaling_factor=None):
    """
    :param offsets: At the moment local offsets should be the first ones
    :param nb_local_offsets: UPDATE AND GENERALIZE!
    :param downscaling_factor: If a list [1,2,2] is given, then the image resolution is scaled down first
    """
    if downscaling_factor is not None:
        raise NotImplementedError()

    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape

    is_local_offset = np.zeros(offsets.shape[0], dtype='bool')
    is_local_offset[:nb_local_offsets] = True
    if label_image is not None:
        assert image_shape == label_image.shape
        if offsets_weights is not None:
            print("Offset weights ignored...!")


    # TODO: change name offsets_probabilities
    # print("Actually building graph...")
    tick = time.time()
    graph = ngraph.undirectedLongRangeGridGraph(image_shape, offsets, is_local_offset,
                        offsets_probabilities=offsets_probabilities,
                        labels=label_image,
                                         strides=strides)
    nb_nodes = graph.numberOfNodes
    # print(label_image)
    # print(is_local_offset)
    # print(offsets_probabilities)


    if label_image is None:
        # print("Getting edge index...")
        offset_index = graph.edgeOffsetIndex()
        # print(np.unique(offset_index, return_counts=True))
        is_local_edge = np.empty_like(offset_index, dtype='bool')
        w = np.where(offset_index < nb_local_offsets)
        is_local_edge[:] = 0
        is_local_edge[w] = 1
        # print("Nb. local edges: {} out of {}".format(is_local_edge.sum(), graph.numberOfEdges))
    else:
        # print("Took {} s!".format(time.time() - tick))
        # print("Checking edge locality...")
        is_local_edge = graph.findLocalEdges(label_image).astype('int32')


    if offsets_weights is None or label_image is not None:
        edge_weights = np.ones(graph.numberOfEdges, dtype='int32')
    else:
        if isinstance(offsets_weights,(list,tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == offsets.shape[0]

        if all([w>=1.0 for w in offsets_weights]):
            # Take the inverse:
            offsets_weights = 1. / offsets_weights
        else:
            assert all([w<=1.0 for w in offsets_weights]) and all([w>=0.0 for w in offsets_weights])

        # print("Edge weights...")
        edge_weights = offsets_weights[offset_index.astype('int32')]


    if GT_label_image is None:
        GT_labels_nodes = np.zeros(nb_nodes, dtype=np.int64)
    else:
        assert GT_label_image.shape == image_shape
        GT_labels_image = GT_label_image.astype(np.uint64)
        GT_labels_nodes = graph.nodeValues(GT_labels_image)

    return graph, is_local_edge, GT_labels_nodes, edge_weights