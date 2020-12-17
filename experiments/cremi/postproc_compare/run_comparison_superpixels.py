import numpy as np
import yaml

from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets
from segmfriends.utils.graph import convert_graph_to_metis_format
import os
from pathutils import get_trendytukan_drive_dir, get_home_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")

save_scores = False
METHOD_NAME = "diceAffs_GASP"
# label_pro_name = "dice_local_edges"
label_pro_name = "dice_all_edges"
sp_method = "LP"


import numpy as np
import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors, mask_the_mask


crop_slice = parse_data_slice("10:15,:200,:200")
# crop_slice = parse_data_slice(":")
print(get_hdf5_inner_paths(dataset))

inner_path = "affinities_mask_average" if "maskAffs" in METHOD_NAME else "affinities_dice"
cremi_affs = readHDF5(dataset, inner_path, crop_slice=(slice(None),) + crop_slice).astype('float32')
# cremi_affs = readHDF5(dataset, "affinities_dice", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)
print(raw.shape)

label_prop_segm = readHDF5("./label_prop_segm/{}.h5".format(label_pro_name), "segm", crop_slice=crop_slice)
#
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, mask_the_mask(label_prop_segm, value_to_mask=1), background=raw, alpha_boundary=0.05, alpha_labels=0.5, z_slice=3)
# save_plot(fig, "./plots/", "label_prop_local.png")


# segm_result_nodes = np.genfromtxt(vieClus_result_file ,delimiter=',')
# segmentation = segm_result_nodes.reshape(cremi_affs.shape[1:])
if "maskAffs" in METHOD_NAME:
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
else:
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
offsets_prob[3:] = 1.
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

from GASP.segmentation.GASP.run_from_affinities import SegmentationFeeder
segm_feeder = SegmentationFeeder()

seeded_WS = SeededWatershedOnAffinities(offsets, hmap_kwargs={'used_offsets': [1,2]})

normal_WS_gen = WatershedFromAffinities(offsets,
                                        used_offsets=[1,2],
                                        stacked_2d=True,
                                        n_threads=6,
                                        )
DTWS_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                            used_offsets=[1,2],
                                                            threshold=0.4,
                                                            min_segment_size=20,
                                                            preserve_membrane=True,
                                                            sigma_seeds=0.1,
                                                            stacked_2d=True,
                                                            n_threads=6,
                                                            )


SP_generator = None
SP_segm = None
if sp_method == "LP":
    SP_generator = segm_feeder
    SP_segm = label_prop_segm
elif sp_method == "WSDT":
    SP_generator = DTWS_gen
    SP_segm = DTWS_gen(cremi_affs.astype('float32'))
elif sp_method == "WS":
    SP_generator = normal_WS_gen
    SP_segm = normal_WS_gen(cremi_affs.astype('float32'))



gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
                          n_threads=8,
                          superpixel_generator=SP_generator,
                          )

args = [cremi_affs.astype('float32')]
if sp_method == "LP":
    args += [label_prop_segm]

import time
tick = time.time()
GASP_segmentation, runtime = gasp(*args)
runtime_total = time.time() - tick
# GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
#
#
# mask = mask.astype('int32')
#

from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
size_thresh_ws = SizeThreshAndGrowWithWS(200, offsets, hmap_kwargs={'used_offsets': [1,2]})
GASP_segmentation = size_thresh_ws(cremi_affs.astype('float32'), GASP_segmentation)

score = cremi_score(GT, GASP_segmentation, return_all_scores=True)
score['runtime'] = runtime_total
score['runtime_agglo'] = runtime

full_name = METHOD_NAME
if sp_method == "LP":
    full_name = "{}_LP_{}".format(full_name, label_pro_name)
else:
    full_name = "{}_{}".format(full_name, sp_method)


print("{}: ".format(full_name), score)
fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
save_plot(fig, "./new_plots/", "{}.png".format(full_name))

if save_scores:
    config_file_path = "./scores/{}.yml".format(full_name)

    with open(config_file_path, 'w') as f:
        yaml.dump(score, f)


# GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'), outlier_foreground_mask, mask_used_edges=None)
# GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
# print("GASP_on_WS with mask: ", cremi_score(GT*outlier_foreground_mask, GASP_segmentation, return_all_scores=True))
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
# save_plot(fig, "./plots/", "GASP_WS_with_outlier_mask.png")
#
# GASP_segmentation, runtime = gasp(cremi_affs.astype('float32'))
# # GASP_segmentation = seeded_WS(cremi_affs.astype('float32'), GASP_segmentation)
# print("GASP_on_WS: ", cremi_score(GT*outlier_foreground_mask, GASP_segmentation, return_all_scores=True))
# fig, axes = get_figure(1,1,figsize=(8,8))
# plot_segm(axes, GASP_segmentation, background=raw, alpha_boundary=0.5, alpha_labels=0.5)
# save_plot(fig, "./plots/", "GASP_WS.png")





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
