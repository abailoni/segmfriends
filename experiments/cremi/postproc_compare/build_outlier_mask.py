import numpy as np
from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets
from segmfriends.utils.graph import convert_graph_to_metis_format
import os
from pathutils import get_trendytukan_drive_dir, get_home_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")


vieClus_result_file = os.path.join(get_home_dir(), "../hci_home/packages/VieClus/cremi_vieClus_test_3")

import numpy as np
import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors, mask_the_mask


# crop_slice = parse_data_slice("10:21,:200,:200")
crop_slice = parse_data_slice(":")
print(get_hdf5_inner_paths(dataset))
# cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice).astype('float32')
cremi_affs = readHDF5(dataset, "affinities_dice", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)
print(raw.shape)

# label_prop_segm = readHDF5("./label_prop_local_edges.h5", "segm")
#
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, mask_the_mask(label_prop_segm, value_to_mask=1), background=raw, alpha_boundary=0.05, alpha_labels=0.5, z_slice=3)
# save_plot(fig, "./plots/", "label_prop_local.png")


# segm_result_nodes = np.genfromtxt(vieClus_result_file ,delimiter=',')
# segmentation = segm_result_nodes.reshape(cremi_affs.shape[1:])
# offsets = [
#     [-1, 0, 0],
#     [0, -1, 0],
#     [0, 0, -1],
#     [0, -4, 0],
#     [0, 0, -4],
#     [0, -4, -4],
#     [0, 4, -4],
#     [-1, -4, 0],
#     [-1, 0, -4],
#     [-1, -4, -4],
#     [-1, 4, -4],
#     [-2, 0, 0],
#     [0, -8, -8],
#     [0, 8, -8],
#     [0, -12, 0],
#     [0, 0, -12]
# ]

offsets = [
  [-1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
  [0, -4, 0],
  [0, 0, -4],
  [0, -4, -4],
  [0, 4, -4],
  [-1, -4, 0],
  [-1, 0, -4],
  [-1, -4, -4],
  [-1, 4, -4],
  [-2, 0, 0],
  [-3, 0, 0],
  [-4, 0, 0],
  [0, -8, -8],
  [0, 8, -8],
  [0, -12, 0],
  [0, 0, -12]
]

# Reduce number of long-range edges:
offsets_prob = np.ones((len(offsets)), dtype='float32')
offsets_prob[3:] = 0.1
#
# # print("Done")
# #
# graph, is_local_edge, edge_sizes = build_pixel_long_range_grid_graph_from_offsets(
#     image_shape=cremi_affs.shape[1:],
#     offsets=offsets,
#     offsets_probabilities=offsets_prob
# )
#
# edge_weights =graph.edgeValues(np.rollaxis(cremi_affs, 0, start=4))
#
# edge_weights_uint = convert_array_from_float_to_uint(edge_weights, convert_to="uint8")
#
# # The algorithm complains if there are edges with weight zero
# edge_weights_uint[edge_weights_uint == 0] = 1
#

from GASP.segmentation.watershed_from_DT import WatershedOnDistanceTransformFromAffinities
from GASP.segmentation.watershed import WatershedFromAffinities


from GASP.segmentation.watershed import SeededWatershedOnAffinities

seeded_WS = SeededWatershedOnAffinities(offsets, hmap_kwargs={'used_offsets': [1,2]})

normal_WS_gen = WatershedFromAffinities(offsets,
                                        used_offsets=[1,2],
                                        stacked_2d=True,
                                        n_threads=6,
                                        )

normal_WS_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                            used_offsets=[1,2],
                                                            threshold=0.4,
                                                            min_segment_size=20,
                                                            preserve_membrane=True,
                                                            sigma_seeds=0.1,
                                                            stacked_2d=True,
                                                            n_threads=6,
                                                            )


gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
                          superpixel_generator=normal_WS_gen, n_threads=6)

WSDT_segm = normal_WS_gen(cremi_affs.astype('float32'))




def compute_outlier_score(label_image,
                                           offsets,
                                           compress_channels=False,
                                           channel_affs=-1,
                                           pad_mode='constant',
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
    assert label_image.ndim == 3
    ndim = 3

    padding = [[0,0] for _ in range(3)]
    for ax in range(3):
        padding[ax][0] = - np.minimum(offsets[:,ax].min(), 0)
        padding[ax][1] = np.maximum(offsets[:,ax].max(), 0)

    if pad_mode == 'edge':
        padded_label_image = np.pad(label_image, pad_width=padding, mode=pad_mode)
    elif pad_mode == 'constant':
        padded_label_image = np.pad(label_image, pad_width=padding, mode=pad_mode, constant_values=pad_constant_values)
    else:
        raise NotImplementedError
    crop_slices = [slice(padding[ax][0], padded_label_image.shape[ax]-padding[ax][1]) for ax in range(3)]

    mask = []
    accumulated_inner_affs = []
    for off_indx, offset in enumerate(offsets):
        rolled_segm = padded_label_image
        for ax, offset_ax in enumerate(offset):
            if offset_ax!=0:
                rolled_segm = np.roll(rolled_segm, -offset_ax, axis=ax)

        mask.append(((padded_label_image == rolled_segm))[crop_slices])
        # accumulated_inner_affs.append((affinities[off_indx] * (padded_label_image == rolled_segm)[crop_slices]))

    # accumulated_inner_affs = np.stack(accumulated_inner_affs)
    return np.stack(mask)



test_segm = np.array([
    [
        [2,2,3],
        [3,2,5]
    ]
])
# test = compute_outlier_score(test_segm, np.array(offsets[:3]))
mask = compute_outlier_score(WSDT_segm, np.array(offsets))
outlier_score = (cremi_affs*mask).sum(axis=0) / mask.sum(axis=0)

# FIXME: get rid of nan values
outlier_score[np.isnan(outlier_score)] = 0
OUTLIER_THRESH = 0.5
OUTLIER_WEIGHT = 0.5
outlier_foreground_mask = outlier_score > OUTLIER_THRESH

outlier_weightning = np.clip(outlier_score - OUTLIER_WEIGHT, 0, 1)
outlier_weightning /= outlier_weightning.max()

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, WSDT_segm, background=raw, alpha_boundary=0.0, alpha_labels=0.6)
save_plot(fig, "./plots/", "WS.png")
fig, axes = get_figure(1,1,figsize=(8,8))
cax = axes.matshow(mask_the_mask(outlier_weightning, value_to_mask=2)[0], cmap=plt.get_cmap('seismic'), vmin=0, vmax=1, interpolation='NONE')
save_plot(fig, "./plots/", "outlier_score.png")
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, mask_the_mask(outlier_foreground_mask, value_to_mask=1), background=raw, alpha_boundary=0.0, alpha_labels=0.6)
save_plot(fig, "./plots/", "outlier_mask.png")

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), outlier_foreground_mask, mask_used_edges=None,
                                  affinities_weights=np.stack([outlier_weightning for _ in range(len(offsets))], axis=0))
GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
#
#
# mask = mask.astype('int32')
#
print("GASP_on_WS with outliers weighting: ", cremi_score(GT*outlier_foreground_mask, GASP_segmentation, return_all_scores=True))
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_WS_with_outlier_mask_and_weight.png")


GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), outlier_foreground_mask, mask_used_edges=None)
GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
print("GASP_on_WS with mask: ", cremi_score(GT*outlier_foreground_mask, GASP_segmentation, return_all_scores=True))
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_WS_with_outlier_mask.png")

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'))
# GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
print("GASP_on_WS: ", cremi_score(GT*outlier_foreground_mask, GASP_segmentation, return_all_scores=True))
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_WS.png")





# from GASP.segmentation.GASP.run_from_affinities import SegmentationFeeder
# segm_feeder = SegmentationFeeder()
#
# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
#                           superpixel_generator=segm_feeder, n_threads=6)
#

# GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
# print(runtime)
#
#
#
# print("GASP_on_LP_masked: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))
#
# GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), label_prop_segm, mask_used_edges=None)
# print(runtime)
#
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
# save_plot(fig, "./plots/", "GASP_on_LP.png")
#
# print("GASP_on_LP: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))
#
#
# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
#                           superpixel_generator=superpixel_gen, n_threads=6)
#
# GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), mask_used_edges=None)
# print(runtime)
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
# save_plot(fig, "./plots/", "GASP_on_WSDT.png")
#
#
# print("GASP_on_WSDT: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))
#
# # gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
# #
# # GASP_segmentation, runtime = gasp(affinities=cremi_affs)
# # print(runtime)
# #
# # print("GASP from pixels:", cremi_score(GT*mask, GASP_segmentation*mask, return_all_scores=True))
#
#
# #
# #
# # #
# # #
# # # convert_graph_to_metis_format(graph, edge_weights_uint, "crop_pixel_graph_cremi.graph")
