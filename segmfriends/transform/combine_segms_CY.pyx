#!python
#cython: language_level=3

cimport numpy as np
import numpy as np

from ..features.mappings import map_features_to_label_array as map_features_to_label_array_nifty
from ..utils.various import cantor_pairing_fct
import vigra


'''
Types:

- int16  ---> int
- int32  ---> long
- int64  ---> long long  (actually both "long" and "long long" seems to work...)

- float16 ---> float
- float32 ---> double
- float64 ---> long double
'''



cdef np.ndarray[long, ndim=1] find_best_agglomeration_CY(np.ndarray[long, ndim=3] segm, np.ndarray[long, ndim=3] GT_segm,
                                                         double underSgm_rel_threshold,
                                                         long ignore_label):
    shape = segm.shape
    max_segm, max_GT  = (segm.max()+1).astype(np.uint64), (GT_segm.max()+1).astype(np.uint64)
    inter_matrix = np.zeros((max_segm, max_GT), dtype=np.uint32)
    flat_segm, flat_GT = segm.flatten().astype(np.uint64), GT_segm.flatten().astype(np.uint64)

    cdef unsigned long[::1] flat_segm_c = flat_segm
    cdef unsigned long[::1] flat_GT_c = flat_GT
    cdef unsigned int[:,::1] inter_matrix_c = inter_matrix
    cdef int dim0 = flat_GT.shape[0]

    for i in range(dim0):
        inter_matrix_c[flat_segm_c[i], flat_GT_c[i]] += 1

    best_labels = np.argmax(inter_matrix, axis=1)
    # Relative threshold:
    if underSgm_rel_threshold > 0.:
        # Superpixel size in initial segm: (avoid problems with zero-sized segments (only ignore label))
        inter_matrix[:, ignore_label] = 0
        segm_SP_sizes = inter_matrix.sum(axis=1) + 1
        assert segm_SP_sizes.shape[0] == inter_matrix.shape[0]

        ignore_mask = inter_matrix[np.arange(inter_matrix.shape[0]), best_labels].astype('float32') / segm_SP_sizes <= underSgm_rel_threshold
        print("NB undersegmeted segments: ", ignore_mask.sum())
        best_labels[ignore_mask] = ignore_label
    return best_labels




cdef np.ndarray[long, ndim=1] find_segmentation_mistakes_CY(
        np.ndarray[unsigned long, ndim=3] segm,
        unsigned long max_segm,
        np.ndarray[unsigned long, ndim=3] GT_segm,
        unsigned long max_GT,
        double ARAND_thresh,
        long ignore_label,
        long mode_axis,
):
    shape = segm.shape

    inter_matrix = np.zeros((max_segm, max_GT), dtype=np.uint32)
    flat_segm, flat_GT = segm.flatten().astype(np.uint64), GT_segm.flatten().astype(np.uint64)

    cdef unsigned long[::1] flat_segm_c = flat_segm
    cdef unsigned long[::1] flat_GT_c = flat_GT
    cdef unsigned int[:,::1] inter_matrix_c = inter_matrix
    cdef int dim0 = flat_GT.shape[0]

    for i in range(dim0):
        inter_matrix_c[flat_segm_c[i], flat_GT_c[i]] += 1


    # Mask ignore labels:
    inter_matrix[:, ignore_label] = 0

    segm_sizes = inter_matrix.sum(axis=mode_axis)
    squared_sum = (inter_matrix**2).sum(axis=mode_axis)

    valid_mask = segm_sizes > 0

    ARAND_scores_per_segm = np.ones_like(segm_sizes, dtype='float32')

    ARAND_scores_per_segm[valid_mask] = squared_sum[valid_mask] / (segm_sizes[valid_mask]**2)

    print(ARAND_scores_per_segm[valid_mask].mean())

    mistakes_mask = (ARAND_scores_per_segm < ARAND_thresh).astype('int64')

    return mistakes_mask


def find_segmentation_mistakes(segm, GT_segm, ARAND_thresh=None, ignore_label=None,
                            mode="undersegmentation"):
    assert segm.ndim == 3, "Only 3D at the moment"
    assert segm.shape == GT_segm.shape, "Segm and GT do not have the same shape"
    assert segm.min() >= 0 and GT_segm.min() >= 0, "Only positive labels are expected"

    if ARAND_thresh is None:
        ARAND_thresh = 1.
    else:
        assert (ARAND_thresh >= 0.) and (ARAND_thresh <= 1.)

    assert ignore_label == 0, "Because of vigra relabeling"

    segm, max_segm, _ = vigra.analysis.relabelConsecutive(segm.astype('uint64'))
    GT_segm, max_GT, _ = vigra.analysis.relabelConsecutive(GT_segm.astype('uint64'))
    max_segm = max_segm + 1
    max_GT = max_GT + 1



    mode_axis = 1
    if mode == "undersegmentation":
        segm_to_map = segm
        # For every segment, check GT in that segment:
        mode_axis = 1
    elif mode == "oversegmentation":
        segm_to_map = GT_segm
        # For every GT, check all segments in that GT:
        mode_axis = 0
    else:
        raise ValueError("The passed mode is not recognised")

    # Necessary to avoid cython compiling error (mistakes_mask could be None..)
    mistakes_mask = 0

    mistakes_mask = find_segmentation_mistakes_CY(
        segm.astype(np.uint64),
        max_segm,
        GT_segm.astype(np.uint64),
        max_GT,
        ARAND_thresh,
        ignore_label,
        mode_axis,

    )

    mistakes_mask_mapped = (
            map_features_to_label_array_nifty(
                segm_to_map,
                np.expand_dims(mistakes_mask, axis=-1),
                number_of_threads=3)
        ).astype(np.int64)[...,0]

    return (segm_to_map * mistakes_mask_mapped).astype('uint64')



def find_best_agglomeration(segm, GT_segm, undersegm_rel_threshold=None, ignore_label=None):
    assert segm.ndim == 3, "Only 3D at the moment"
    assert segm.shape == GT_segm.shape
    assert segm.min() >= 0 and GT_segm.min() >= 0, "Only positive labels are expected"

    if undersegm_rel_threshold is None:
        undersegm_rel_threshold = 0
    else:
        assert (undersegm_rel_threshold >= 0.) and (undersegm_rel_threshold <= 1.)
    if ignore_label is None:
        ignore_label = 0

    return find_best_agglomeration_CY(segm.astype(np.int64), GT_segm.astype(np.int64),
                                      undersegm_rel_threshold, ignore_label)



cdef np.ndarray[long, ndim=1] find_split_GT_CY(np.ndarray[long, ndim=3] finalSegm, np.ndarray[long, ndim=3] bestGT,
                                                         double size_ignored_SP_relative,
                                                         long ignore_label,
                                               long number_of_threads):
    shape = finalSegm.shape

    # Find intersection segm:
    print("Finding intersection segm...")
    finalSegm_mod, max_segm, _ = vigra.analysis.relabelConsecutive(finalSegm.astype('uint32'))
    intersect_segm, max_interSegm, _ = vigra.analysis.relabelConsecutive(cantor_pairing_fct(finalSegm_mod, bestGT).astype('uint32'))

    max_GT  = (bestGT.max()+1).astype(np.uint64)
    max_segm += 1
    inter_matrix = np.zeros((max_segm, max_GT), dtype=np.uint32)
    inter_matrix_interSect_indices = np.zeros((max_segm, max_GT), dtype=np.uint32)
    flat_segm, flat_GT = finalSegm_mod.flatten().astype(np.uint64), bestGT.flatten().astype(np.uint64)
    flat_interSegm = intersect_segm.flatten().astype(np.uint64)

    cdef unsigned long[::1] flat_segm_c = flat_segm
    cdef unsigned long[::1] flat_interSegm_c = flat_interSegm
    cdef unsigned long[::1] flat_GT_c = flat_GT
    cdef unsigned int[:,::1] inter_matrix_c = inter_matrix
    cdef unsigned int[:,::1] inter_matrix_interSect_indices_c = inter_matrix_interSect_indices
    cdef int dim0 = flat_GT.shape[0]

    print("Computing intersection matrix...")
    for i in range(dim0):
        inter_matrix_c[flat_segm_c[i], flat_GT_c[i]] += 1
        inter_matrix_interSect_indices_c[flat_segm_c[i], flat_GT_c[i]] = flat_interSegm_c[i]

    print("Finding undersegmented segments...")
    # Relative threshold:

    # Superpixel size in initial finalSegm: (avoid problems with zero-sized segments (only ignore label))
    inter_matrix[:, ignore_label] = 0
    segm_SP_sizes = inter_matrix.sum(axis=1) + 1
    print(segm_SP_sizes.mean())
    print(inter_matrix.max(axis=1).mean())
    debug = (inter_matrix / np.expand_dims(segm_SP_sizes, axis=1).astype('float64')).max(axis=1).mean()
    print(debug)

    large_segments_mask = ((inter_matrix.astype('float32') / np.expand_dims(segm_SP_sizes, axis=1)) >= size_ignored_SP_relative).astype('int16')

    # print(large_segments_mask.sum(axis=1).mean())
    # large_segments_mask[np.where(large_segments_mask.sum(axis=1) <= 1)] = np.zeros(inter_matrix.shape[1], dtype='int16')

    # Ignore segments that are not undersegmented (only one main GT label above the thresh):
    best_labels = np.argmax(inter_matrix, axis=1)
    ignore_mask = inter_matrix[np.arange(inter_matrix.shape[0]), best_labels].astype('float32') / segm_SP_sizes >= 0.9
    print("NB undersegmeted segments: ", ignore_mask.sum())
    large_segments_mask[ignore_mask] = np.zeros(inter_matrix.shape[1], dtype='int16')


    print("Associating GT ids to intersected undersegm segments...")
    GT_labels_undesegm_SP = np.ones((max_interSegm+1), dtype=np.uint64) * ignore_label
    cdef unsigned long[::1] GT_labels_undesegm_SP_c = GT_labels_undesegm_SP
    cdef short [:,::1] large_segments_mask_c = large_segments_mask

    for SP_id in range(large_segments_mask.shape[0]):
        for GT_id in range(large_segments_mask.shape[1]):
            if large_segments_mask_c[SP_id, GT_id] == 1:
                GT_labels_undesegm_SP_c[inter_matrix_interSect_indices_c[SP_id, GT_id]] = GT_id

    print("Mapping labels to image...")
    split_GT = (
            map_features_to_label_array_nifty(
                intersect_segm,
                np.expand_dims(GT_labels_undesegm_SP, axis=-1),
                number_of_threads=number_of_threads)
        ).astype(np.int64)[...,0]

    return split_GT

def find_split_GT(segm, bestGT_segm, size_small_segments_rel, ignore_label=None,
                  number_of_threads=8):
    assert segm.ndim == 3, "Only 3D at the moment"
    assert segm.shape == bestGT_segm.shape
    assert segm.min() >= 0 and bestGT_segm.min() >= 0, "Only positive labels are expected"

    assert (size_small_segments_rel>= 0.) and (size_small_segments_rel<= 1.)
    if ignore_label is None:
        ignore_label = 0

    if number_of_threads is None:
        number_of_threads = 8

    return find_split_GT_CY(segm.astype(np.int64), bestGT_segm.astype(np.int64),
                                      size_small_segments_rel, ignore_label, number_of_threads)



