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
cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)
print(raw.shape)

label_prop_segm = readHDF5("./label_prop_local_edges.h5", "segm")

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, mask_the_mask(label_prop_segm, value_to_mask=1), background=raw, alpha_boundary=0.05, alpha_labels=0.5, z_slice=3)
save_plot(fig, "./plots/", "label_prop_local.png")
raise ValueError

# segm_result_nodes = np.genfromtxt(vieClus_result_file ,delimiter=',')
# segmentation = segm_result_nodes.reshape(cremi_affs.shape[1:])
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
    [0, -8, -8],
    [0, 8, -8],
    [0, -12, 0],
    [0, 0, -12]
]

sum_affs = cremi_affs.sum(axis=0)
max_affs = cremi_affs.max(axis=0)
min_local = cremi_affs[[1,2]].min(axis=0)
sum_affs_2 = cremi_affs[[1,2,3,4,5,6,12,13,14,15]].sum(axis=0)
max_affs_2 = cremi_affs[[1,2,3,4,5,6,12,13,14,15]].max(axis=0)

mask = (max_affs>0.8).astype('int32')

# Reduce number of long-range edges:
offsets_prob = np.ones((len(offsets)), dtype='float32')
offsets_prob[3:] = 1
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


superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
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

WSDT_segm = superpixel_gen(cremi_affs.astype('float32'), mask)

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), mask, mask_used_edges=None)
print(runtime)

GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
#



fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, mask_the_mask(mask, value_to_mask=1), background=raw, alpha_boundary=0.05, alpha_labels=0.5, z_slice=3)
save_plot(fig, "./plots/", "mask_boundary.png")
#
# print(cremi_score(GT, segmentation, return_all_scores=True))
#
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_on_WS_masked.png")

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, WSDT_segm, background=raw, alpha_boundary=0.05, alpha_labels=0.5)
save_plot(fig, "./plots/", "WS_segm.png")


mask = mask.astype('int32')

print("GASP_on_WSDT masked: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))



from GASP.segmentation.GASP.run_from_affinities import SegmentationFeeder
segm_feeder = SegmentationFeeder()

gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
                          superpixel_generator=segm_feeder, n_threads=6)

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), label_prop_segm, mask, mask_used_edges=None)
GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
print(runtime)

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_on_LP_masked.png")


print("GASP_on_LP_masked: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), label_prop_segm, mask_used_edges=None)
print(runtime)

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_on_LP.png")

print("GASP_on_LP: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))


gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
                          superpixel_generator=superpixel_gen, n_threads=6)

GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), mask_used_edges=None)
print(runtime)
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./plots/", "GASP_on_WSDT.png")


print("GASP_on_WSDT: ", cremi_score(GT, GASP_segmentation, return_all_scores=True))

# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# GASP_segmentation, runtime = gasp(affinities=cremi_affs)
# print(runtime)
#
# print("GASP from pixels:", cremi_score(GT*mask, GASP_segmentation*mask, return_all_scores=True))


#
#
# #
# #
# # convert_graph_to_metis_format(graph, edge_weights_uint, "crop_pixel_graph_cremi.graph")
