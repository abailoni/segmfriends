import numpy as np


# def find_best_agglomeration(segm, GT_segm):
#     """DEPRECATED, IMPLEMENTED IN CYTHON"""
#     assert segm.ndim == 3, "Only 3D at the moment"
#     assert segm.shape == GT_segm.shape
#     assert segm.min() >= 0 and GT_segm.min() >= 0
#     shape = segm.shape
#     max_segm, max_GT  = (segm.max()+1).astype(np.uint64), (GT_segm.max()+1).astype(np.uint64)
#     inter_matrix = np.zeros((max_segm, max_GT), dtype=np.uint32)
#     flat_segm, flat_GT = segm.flatten().astype(np.uint64), GT_segm.flatten().astype(np.uint64)
#     for i in range(flat_GT.shape[0]):
#         inter_matrix[flat_segm[i], flat_GT[i]] += 1
#     return np.argmax(inter_matrix, axis=1)