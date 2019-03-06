import numpy as np
from nifty.graph import rag as nrag


def compute_mask_boundaries_graph(offsets, graph=None, label_image=None, contractedRag=None,
                                  compress_channels=False, return_boundary_IDs=False, channel_axis=-1,
                                  use_undirected_graph=False,
                                  number_of_threads =6):
    """
    Label image or graph should be given. Using nifty graph or rag. Now there is a possibility
    (use_undirected_graph = True) to pass a general undirected graph together with a label image.

    The resulting mask will be based exactly on the edges of the graph (only neighboring segments).

    :param offsets: numpy array
    Example: [ [0,1,0], [0,0,1] ]

    :return: in bound_IDs inner affinities have ID = -1
    """
    assert label_image is not None or graph is not None
    if contractedRag is not None:
        assert graph is not None

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

    if not use_undirected_graph:
        if contractedRag is None:
            mask, bound_IDs = nrag.boundaryMaskLongRange(graph, offsets.astype(np.int32),
                                                         number_of_threads)
        else:
            print("Warning: multiple-threads not implemented atm")
            mask, bound_IDs = nrag.boundaryMaskLongRange(graph, contractedRag, offsets.astype(np.int32))
    else:
        assert contractedRag is None, "Not implemented atm"
        assert label_image is not None
        # Here 'graph' is actually a general undirected graph:
        undir_graph = graph
        mask, bound_IDs = nrag.boundaryMaskLongRange(undir_graph,
                                                     label_image,
                                                     offsets.astype(np.int32),
                                                     number_of_threads)

    ndim = mask.ndim - 1

    if compress_channels:
        compressed_mask = np.zeros(mask.shape[:ndim], dtype=np.int8)
        for ch_nb in range(mask.shape[-1]):
            compressed_mask = np.logical_or(compressed_mask, mask[...,ch_nb])
        return compressed_mask

    dims = tuple(range(ndim))

    if channel_axis==0:
        if return_boundary_IDs:
            return np.transpose(bound_IDs, (ndim,) + dims )
        else:
            return np.transpose(mask, (ndim,) + dims )
    elif channel_axis!=-1:
        raise NotImplementedError()

    if return_boundary_IDs:
        return bound_IDs
    else:
        return mask



def compute_boundary_mask_from_label_image(label_image,
                                           offsets,
                                           compress_channels=False,
                                           channel_affs=-1,
                                           pad_mode='edge',
                                           pad_constant_values=0,
                                           background_value=None,
                                           return_affinities=False):
    """
    Faster than the nifty version, but does not check the actual connectivity of the segments (no rag is
    built). A non-local edge could be cut, but it could also connect not-neighboring segments.
b
    It returns a boundary mask (1 on boundaries, 0 otherwise). To get affinities reverse it.

    :param offsets: numpy array
        Example: [ [0,1,0], [0,0,1] ]

    :param return_boundary_affinities:
        if True, the output shape is (len(axes, z, x, y)
        if False, the shape is       (z, x, y)

    :param channel_affs: accepted options are 0 or -1

    :param background_value: if either one of the two pixels is equal to background_value, then the edge is
            labelled as boundary
    """
    # TODO: use the version already implemented in the trasformations and using convolution kernels
    # TODO: use the standard convention and return affinities!
    assert label_image.ndim == 3
    ndim = 3

    padding = [[0,0] for _ in range(3)]
    for ax in range(3):
        # raise NotImplementedError("This is broken")
        # padding[ax][0] = np.abs(offsets[:, ax].min())
        # FIXME: there is something going wrong here because negative offsets are ignored in this step!
        padding[ax][1] = offsets[:,ax].max()

    if pad_mode == 'edge':
        padded_label_image = np.pad(label_image, pad_width=padding, mode=pad_mode)
    elif pad_mode == 'constant':
        padded_label_image = np.pad(label_image, pad_width=padding, mode=pad_mode, constant_values=pad_constant_values)
    else:
        raise NotImplementedError
    crop_slices = [slice(padding[ax][0], padded_label_image.shape[ax]-padding[ax][1]) for ax in range(3)]

    boundary_mask = []
    for offset in offsets:
        rolled_segm = padded_label_image
        for ax, offset_ax in enumerate(offset):
            if offset_ax!=0:
                rolled_segm = np.roll(rolled_segm, -offset_ax, axis=ax)
        if background_value is not None:
            boundary_mask.append(np.logical_or(np.logical_or(padded_label_image == background_value, rolled_segm == background_value),
                                           padded_label_image != rolled_segm)[crop_slices])
        else:
            boundary_mask.append((padded_label_image != rolled_segm)[crop_slices])

    boundary_affin = np.stack(boundary_mask)



    if compress_channels:
        compressed_mask = np.zeros(label_image.shape[:ndim], dtype=np.int8)
        for ch_nb in range(boundary_affin.shape[0]):
            compressed_mask = np.logical_or(compressed_mask, boundary_affin[ch_nb])
        out = compressed_mask
    elif channel_affs==0:
        out = boundary_affin
    else:
        assert channel_affs == -1
        out = np.transpose(boundary_affin, (1,2,3,0))

    if return_affinities:
        return np.logical_not(out)
    else:
        return out