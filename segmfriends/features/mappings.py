import numpy as np
from nifty.graph import rag as nrag
from nifty import tools as ntools


def map_features_to_label_array(label_array, features, ignore_label=-1,
                                fill_value=0.,number_of_threads=-1):
    """

    :param label_array:
    :param features:
    :param ignore_label: the label in label_array that should be ignored in the mapping
    :param fill_value: the fill value used in the mapped array to replace the ignore_label
    :return:
    """
    # TODO: deprecate
    return ntools.mapFeaturesToLabelArray(label_array, features, ignore_label, fill_value, number_of_threads)


    # # if label_array.ndim != 3:
    # #     raise NotImplementedError("Bug in nifty function...!")
    #
    # if ignore_label is None:
    #     ignore_label = -1
    # if number_of_threads==1:
    #     try:
    #         from .mappings_CY import map_features_to_label_array as map_features_to_label_array_CY
    #     except ImportError:
    #         raise ImportError("Cython module should be compiled first")
    #     # Using faster cython version:
    #     return map_features_to_label_array_CY(label_array, features, ignore_label,
    #                             fill_value)
    # else:
    #     # Using multi-threaded nifty version:
    #     return nrag.mapFeaturesToLabelArray(label_array.astype(np.int64),
    #                                         features.astype(np.float64),
    #                                         ignore_label,
    #                                         fill_value,
    #                                         numberOfThreads=number_of_threads)


def map_edge_features_to_image(offsets, edge_features, rag=None, label_image=None, contractedRag=None,
                               channel_affs=-2, fillValue=0., number_of_threads=8):
    """
    Label image or rag should be passed. Using nifty rag.
    """
    raise DeprecationWarning()
    assert label_image is not None or rag is not None
    if contractedRag is not None:
        assert rag is not None

    if rag is None:
        rag = nrag.gridRag(label_image.astype(np.uint32))

    if contractedRag is None:
        image_map = nrag.mapFeaturesToBoundaries(rag, edge_features.astype(np.float32),
                                                       offsets.astype(np.int32), fillValue,
                                                 number_of_threads)
    else:
        assert number_of_threads == 1, "Multiple threads are currently not supported with a contracted graph!"
        image_map = nrag.mapFeaturesToBoundaries(rag, contractedRag,
                                                       edge_features.astype(np.float32),
                                                       offsets.astype(np.int32), fillValue)

    if channel_affs==0:
        ndim = image_map.ndim - 2
        dims = tuple(range(ndim))
        return np.transpose(image_map, (ndim,) + dims + (ndim+1,) )
    elif channel_affs!=-2:
        raise NotImplementedError()

    return image_map
