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



import numpy as np
import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors


crop_slice = parse_data_slice("10:11,:200,:200")
print(get_hdf5_inner_paths(dataset))
cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice)
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)

label_prop_segm_file = os.path.join(get_home_dir(), "repositories/signed_labelpropagation/out_segm/first_test_2.segm")
segm_result_nodes = np.genfromtxt(label_prop_segm_file, delimiter=',')
label_prop_segm = segm_result_nodes.reshape(cremi_affs.shape[1:])
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

mask = (max_affs<0.8).astype('int32')
mask = (min_local>0.9).astype('int32')

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
# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# GASP_segmentation, runtime = gasp(affinities=cremi_affs)
# print(runtime)
#

from GASP.segmentation.watershed import WatershedFromAffinities

# ws_genereator = WatershedFromAffinities(offsets=offsets, used_offsets=[1,2], stacked_2d=True)
ws_genereator = WatershedFromAffinities(offsets=offsets, used_offsets=[1,2,3,4,5,6,12,13,14,15],
                                        offset_weights=[1,1,0.7,0.7,0.7,0.7,0.3,0.3,0.3,0.3],
                                        stacked_2d=True)

ws_segm = ws_genereator(cremi_affs.astype('float32'))

#
#
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, label_prop_segm, background=raw, alpha_boundary=0.05, alpha_labels=0.99, z_slice=0)
save_plot(fig, "./plots/", "label_prop.png")
#
print(cremi_score(GT, label_prop_segm, return_all_scores=True))

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, ws_segm, background=raw, alpha_boundary=0.05, alpha_labels=0.99, z_slice=0)
save_plot(fig, "./plots/", "ws_segm.png")
print(cremi_score(GT, ws_segm, return_all_scores=True))



#
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.05, alpha_labels=0.5)
# save_plot(fig, "./", "GASP_outSegm.png")
#
# print(cremi_score(GT, GASP_segmentation, return_all_scores=True))
#
#
# #
# #
# # convert_graph_to_metis_format(graph, edge_weights_uint, "crop_pixel_graph_cremi.graph")
