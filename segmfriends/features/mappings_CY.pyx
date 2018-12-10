#!python
#cython: language_level=3

cimport numpy as np
import numpy as np


# TODO: find better way to combine the different versions...

cdef np.ndarray[long, ndim=4] map_features_to_label_array_CY_3D(np.ndarray[long, ndim=3] label_image,
                                                             np.ndarray[double, ndim=2] feature_array,
                                                             long ignore_label,
                                                             double fill_value):
    cdef:
        int dim0 = label_image.shape[0]
        int dim1 = label_image.shape[1]
        int dim2 = label_image.shape[2]
        int nb_feat = feature_array.shape[1]
    feature_image = np.ones((dim0, dim1, dim2, feature_array.shape[1])) * fill_value

    cdef double[:,:,:,::1] feature_image_c = feature_image
    cdef double[:,::1] feature_array_c = feature_array
    cdef long[:,:,::1] label_image_c = label_image


    cdef long label

    for k in range(dim0):
        for i in range(dim1):
            for j in range(dim2):
                label = label_image_c[k,i,j]
                if label!=ignore_label:
                    for f in range(nb_feat):
                        feature_image_c[k,i,j,f] = feature_array_c[label, f]

    return feature_image

cdef np.ndarray[long, ndim=5] map_features_to_label_array_CY_4D(np.ndarray[long, ndim=4] label_image,
                                                             np.ndarray[double, ndim=2] feature_array,
                                                             long ignore_label,
                                                             double fill_value):
    cdef:
        int dim0 = label_image.shape[0]
        int dim1 = label_image.shape[1]
        int dim2 = label_image.shape[2]
        int dim3 = label_image.shape[3]
        int nb_feat = feature_array.shape[1]

    feature_image = np.ones((dim0, dim1, dim2, dim3, feature_array.shape[1])) * fill_value


    cdef double[:,:,:,:,::1] feature_image_c = feature_image
    cdef double[:,::1] feature_array_c = feature_array
    cdef long[:,:,:,::1] label_image_c = label_image

    cdef long label

    for k in range(dim0):
        for i in range(dim1):
            for j in range(dim2):
                for t in range(dim3):
                    label = label_image_c[k,i,j,t]
                    if label!=ignore_label:
                        for f in range(nb_feat):
                            feature_image_c[k,i,j,t,f] = feature_array_c[label, f]

    return feature_image

def map_features_to_label_array(label_image, feature_array, ignore_label=None, fill_value=0.):
    """
    feature_array:

        - first dimension gives ID of the labels
        - second dimension represents the different features

    """
    # TODO: this could be expensive:
    label_image = label_image.copy(order='C')
    feature_array = feature_array.copy(order='C')

    ignore_label = -1 if ignore_label is None else ignore_label
    if label_image.ndim==3:
        return map_features_to_label_array_CY_3D(label_image.astype(np.int64), feature_array.astype(np.float64), ignore_label, fill_value)
    elif label_image.ndim==4:
        return map_features_to_label_array_CY_4D(label_image.astype(np.int64), feature_array.astype(np.float64), ignore_label, fill_value)
    else:
        raise NotImplementedError()