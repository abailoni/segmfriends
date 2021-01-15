import numpy as np

from GASP.affinities import AccumulatorLongRangeAffs
from GASP.affinities.utils import probs_to_costs
from segmfriends.utils.graph import convert_graph_to_metis_format
from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets, build_lifted_graph_from_rag, get_rag
import os
from pathutils import get_trendytukan_drive_dir, get_home_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint, writeHDF5, cantor_pairing_fct

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


METHOD_NAME = "verySmall_cremi_crop_forceLocal_doubleLP"
LP_method = "sum"

dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")

crop_slice = parse_data_slice("10:11,:100,:100")
# crop_slice = parse_data_slice(":")
# print(get_hdf5_inner_paths(dataset))
# cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice).astype('float32')
cremi_affs = readHDF5(dataset, "affinities_dice", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)

label_pro_name = "dice_all_edges_forceLocal"
loaded_label_prop_segm = readHDF5("./label_prop_segm/{}.h5".format(label_pro_name), "segm", crop_slice=crop_slice)

from vigra.analysis import labelVolume, relabelConsecutive
loaded_label_prop_segm = relabelConsecutive(loaded_label_prop_segm.astype('uint32'))[0]




# offsets = [
#     [0, 1, 0],
#     [0, 0, 1],
# ]
#
# affinities_y = np.array([[
#     [1,1,1,1,1],
#     [1,0,1,0,1],
#     [1,0,1,0,1],
#     [1,0,1,0,1],
#     [1,1,1,1,1],
# ]])
#
# affinities_x = np.array([[
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,1,1,1,1],
# ]])
#
# # cremi_affs = np.stack([affinities_x, affinities_y]).astype('float32')
#
# cremi_affs = np.ones((2,1,25,25), dtype='float32')
#
# # Add x-boundary:
# cremi_affs[0,:,7,7:14] = 0
# cremi_affs[0,:,13,7:14] = 0
#
# # Add y-boundary:
# cremi_affs[1,:,7:14,7] = 0
# cremi_affs[1,:,7:14,13] = 0
# # cremi_affs[:] = 1
#
# # Add some random noise:
# cremi_affs += np.random.normal(0,0.01,size=cremi_affs.shape)
# # cremi_affs = np.clip(cremi_affs, 0, 1)
#
#
# print("test")

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
if "onlyLocal" in METHOD_NAME:
    offsets_prob[3:] = 0.

# print("Done")
#

import time
tick = time.time()
shape_2D = (1,) + cremi_affs.shape[2:]
# rag, dict = get_rag(label_prop_segm, nb_threads=8)
# graph, is_local_edge = build_lifted_graph_from_rag(rag, offsets, add_lifted_edges=True)
# graph, is_local_edge, edge_sizes = build_pixel_long_range_grid_graph_from_offsets(
#     image_shape=shape_2D,
#     offsets=offsets,
#     offsets_probabilities=offsets_prob
# )

accumulator = AccumulatorLongRangeAffs(offsets, return_dict=True, offset_probabilities=offsets_prob, n_threads=-1)


def lp_python(z):
    graph_data_dict = accumulator(cremi_affs[:,[z]], loaded_label_prop_segm[[z]])

    graph = graph_data_dict["graph"]
    is_local_edge = graph_data_dict["is_local_edge"]
    edge_indicators = graph_data_dict["edge_indicators"]

    signed_edge_weights = edge_indicators - 0.5

    if "forceLocal" in METHOD_NAME:
        local_edges = is_local_edge
    else:
        local_edges = None

    from nifty.graph import run_label_propagation
    from nifty import tools as ntools
    import time
    tick = time.time()
    print("Check")

    segm_result_nodes = run_label_propagation(graph, signed_edge_weights, nb_iter=150, local_edges=local_edges)
    print("Label prop took ", time.time() - tick)

    label_prop_segm = ntools.mapFeaturesToLabelArray(
        loaded_label_prop_segm[[z]],
        np.expand_dims(segm_result_nodes, axis=-1),
        nb_threads=8,
        fill_value=-1.,
        ignore_label=-1,
    )[..., 0].astype(np.uint32)

    from affogato.affinities import compute_affinities
    active_edges, valid_mask = compute_affinities(label_prop_segm.astype('uint64'), offsets, False, 0)

    MC_energy = (cremi_affs[:,[z]][np.logical_and(active_edges==0, valid_mask==1)] - 0.5).sum()
    if MC_energy > 0:
        print(graph.numberOfNodes, np.unique(segm_result_nodes).shape)
    print("MC energy: ", MC_energy)

    max_label = label_prop_segm.max()

    return label_prop_segm, max_label

METHOD_NAME += "_{}".format(LP_method)

gasp = GaspFromAffinities(offsets,
                          offsets_probabilities=offsets_prob,
                          verbose=True,
                          n_threads=8,
                          run_GASP_kwargs={'linkage_criteria': 'max', 'add_cannot_link_constraints': False, 'use_efficient_implementations': True},
                          )
# connected_components, runtime = gasp(cremi_affs)

label_prop_segm = np.zeros(cremi_affs.shape[1:], dtype='uint32')
max_label = 0
for z in range(cremi_affs.shape[1]):
    print(z)
    label_prop_segm_z, max_label_z = lp_python(z)
    # label_prop_segm_z, max_label_z = compute_stacked_sp(z)
    label_prop_segm[z] = label_prop_segm_z + max_label
    max_label += max_label_z


# from vigra.analysis import labelVolume, relabelConsecutive
#
# segm_relabelled = labelVolume(label_prop_segm.astype('uint32'))
# segm_relabelled = relabelConsecutive(segm_relabelled)[0]
# intersect_segm = cantor_pairing_fct(connected_components, segm_relabelled)
# intersect_segm = relabelConsecutive(intersect_segm)[0]
#
# has_mistake = not np.allclose(intersect_segm, segm_relabelled)
# if has_mistake:
#     raise ValueError

# print("Done")
# writeHDF5(label_prop_segm, "./{}.h5".format(METHOD_NAME), "segm")

import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors, mask_the_mask

fig, axes = get_figure(1,2,figsize=(8,16))
# axes.matshow(mask_the_mask(cremi_affs[0,0], value_to_mask=1), cmap='gray', alpha=0.9, interpolation='None')
# axes.matshow(mask_the_mask(cremi_affs[1,0], value_to_mask=1), cmap='gray', alpha=0.9, interpolation='None')
plot_segm(axes[0], loaded_label_prop_segm, alpha_boundary=0.0, z_slice=0, background=raw)
plot_segm(axes[1], label_prop_segm, alpha_boundary=0.0, z_slice=0, background=raw)
# axes.matshow(mask_the_mask(cremi_affs.min(axis=0)[0], value_to_mask=1), cmap='gray', alpha=0.6, interpolation='None')
# plot_segm(axes, segm, alpha_boundary=0.0, alpha_labels=0.9, z_slice=0)
save_plot(fig, "./new_plots_2/", "{}.png".format(METHOD_NAME))



writeHDF5(label_prop_segm, "./{}.h5".format(METHOD_NAME), "segm")
print(METHOD_NAME)


# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# segmentation, runtime = gasp(affinities=cremi_affs)

# print(cremi_score(GT, segmentation, return_all_scores=True))
