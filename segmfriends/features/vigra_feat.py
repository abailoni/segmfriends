import numpy as np
import vigra

from .mappings import map_features_to_label_array
from ..utils import find_first_index

def _get_SP_features_VIGRA_extractor(data, segmentation,
                          statistics=None, ignore_label=None):
    """
    If max_SP is None, the features are normalized to 1 and the max is returned as second output
    """
    # list of the region statistics, that we want to extract
    # you don't need the last two and you also should use the mean value
    if statistics is None:
        statistics = ["Count", "Kurtosis",  # "Histogram",
                  "Maximum", "Minimum", "Quantiles",
                  "RegionRadii", "Skewness", "Sum",
                  "Variance", "Mean"]

    extractor = vigra.analysis.extractRegionFeatures(
        data.astype(np.float32),
        segmentation.astype(np.uint32),
        features=statistics,
        ignoreLabel=ignore_label
    )
    return extractor


def accumulate_segment_features_vigra(data_list, segmentation_list,
                          statistics=None,
                          normalization_mode=None,
                          map_to_image=True,
                          ignore_label=None):
    # TODO: Rewrite this amazing spaghetti code
    """
    Remarks:
        - inputs can be numpy arrays or lists of arrays. The shape should be 2D or 3D.
        - if lists, the accepted options are:
            - len(data_list)==len(segmentation_list)
            - data_list or segmentation_list has length==1. This one will be used multiple times.


    @:param:normalization_mode: string or list of strings (for each statistic)
        The supported modes are:
            - None
            - 'affin_center': subtract 0.5
            - 'normalize'
            - 'zero_center'

        boolean or list of booleans (value for each feature. Remark: features with more values should be considered..)

    The output shape is:
        - node_features.shape = (number_IDs, number_features) if not map_to_image
        - image.shape = (z, x, y, number_features) if map_to_image

    """
    # tick = time.time()
    data_list = data_list if isinstance(data_list, list) else [data_list]
    segmentation_list = segmentation_list if isinstance(segmentation_list, list) else [segmentation_list]
    nb_data, nb_segm = len(data_list), len(segmentation_list)
    max_len = max(nb_data, nb_segm)


    # Trick to avoid problems with max-label and merging of different extractors:
    segmentation_list = [np.copy(segm) for segm in segmentation_list]
    first_indices = []
    if nb_segm!=1:
        max_labels = np.array([segmentation_list[i].max() for i in range(nb_segm)])
        new_max_label = max_labels.max() + 1
        for i in range(nb_segm):
            # Label 0 is not-boundary and always ignored. Add one pixel segment with max-label:
            first_indx = find_first_index(segmentation_list[i], 0)
            first_indices.append(first_indx)
            segmentation_list[i][first_indx] = new_max_label

    # Match the length of data/segmentations:
    if max_len!=1:
        if nb_data==1:
            data_list = [data_list[0] for _ in range(max_len)]
        elif nb_segm==1:
            segmentation_list = [segmentation_list[0] for _ in range(max_len)]
        else:
            assert nb_data==nb_segm

    # print "Time prep: {}".format(time.time() - tick)
    # tick = time.time()

    def get_node_features(stats):
        total_extractor = None
        for data, segmentation in zip(data_list, segmentation_list):
            new_extractor = _get_SP_features_VIGRA_extractor(data, segmentation, stats, ignore_label)
            if total_extractor is None:
                total_extractor = new_extractor
            else:
                total_extractor.merge(new_extractor)

        node_features = []
        nb_features = []
        for stat_name in stats:
            new_feat = total_extractor[stat_name].astype(np.float32)
            nb_IDs = new_feat.shape[0]
            new_feat = new_feat[:,None] if new_feat.ndim == 1 else new_feat.reshape((nb_IDs, -1))
            node_features.append(new_feat)
            nb_features.append(new_feat.shape[1])
        return np.concatenate(node_features, axis=-1), nb_features

    def weighted_avg_and_std(values, weights, axis=None):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights, axis=axis)
        variance = np.average((values - average) ** 2, weights=weights, axis=axis)  # Fast and numerically precise
        return (average, np.sqrt(variance))

    node_features, nb_features = get_node_features(statistics)
    # print "Time compute vigra: {}".format(time.time() - tick)
    # tick = time.time()

    # if len(statistics) == 2:
    #     print "Lengths: {}, {}. Max averages: {}".format(nb_data, nb_segm, node_features[1:, :].max(axis=0))
    #     if node_features[1:, 0].max() <= 0.:
    #         pass
    #
    # # Debug:
    # if np.isnan(node_features).sum()>1 and len(statistics)==2:
    #     assert np.isnan(node_features).sum()==0, "Some NaN values found"

    if normalization_mode is not None:
        # Compute counts (if not already present) to get an averaged normalization:
        count_stats, _ = get_node_features(['Count'])

        if not isinstance(normalization_mode, list):
            normalization_mode = [normalization_mode for _ in statistics]
        else:
            assert len(normalization_mode) == len(statistics)
        count_feat = 0
        for nb_feat, normalization in zip(nb_features, normalization_mode):
            if normalization is not None:
                feat_slice = (slice(None), slice(count_feat, count_feat+nb_feat))
                if normalization=='normalize':
                    avg, std = weighted_avg_and_std(node_features[feat_slice], np.tile(count_stats, (1,nb_feat)), axis=0)
                    node_features[feat_slice] -= avg[None, :]
                    node_features[feat_slice] /= std[None, :]
                elif normalization=='zero_center':
                    avg, std = weighted_avg_and_std(node_features[feat_slice], np.tile(count_stats, (1, nb_feat)),
                                                    axis=0)
                    node_features[feat_slice] -= avg[None, :]
                elif normalization=='affin_center':
                    node_features[feat_slice] -= 0.5
                else:
                    raise NotImplementedError('The passed normalization option is not supported.')
            count_feat += nb_feat

    node_features[np.where(np.isnan(node_features))] = 0.

    # print "Time normalization: {}".format(time.time() - tick)
    # tick = time.time()

    if map_to_image:
        output_images = []
        for segm in segmentation_list:
            # TODO: use faster cython implementation
            output_images.append(map_features_to_label_array(segm, node_features, ignore_label))
            # output_images.append(map_features_to_label_image(segm, node_features, ignore_label=ignore_label))

        # Fix the changes given by the previously introduced artificial max label:
        if nb_segm!=1:
            for i in range(nb_segm):
                output_images[i][first_indices[i]] = 0.

        output_images = output_images if len(output_images)!=1 else output_images[0]
        # print "Time remapping: {}".format(time.time() - tick)
        # time.sleep(10)
        return output_images
    else:
        # Ignore last added ID:
        if nb_segm != 1:
            return node_features[:-1]
        else:
            return node_features