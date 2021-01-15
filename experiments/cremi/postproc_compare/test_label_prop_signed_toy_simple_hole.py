import numpy as np
from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets
from segmfriends.utils.graph import convert_graph_to_metis_format
import os
from pathutils import get_trendytukan_drive_dir, get_home_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint, writeHDF5, cantor_pairing_fct

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


METHOD_NAME = "simple_hole_forceLocal"
LP_method = "sum"

dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")

crop_slice = parse_data_slice("10:12,:200,:200")
# crop_slice = parse_data_slice(":")
print(get_hdf5_inner_paths(dataset))
# cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice).astype('float32')
# cremi_affs = readHDF5(dataset, "affinities_dice", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)

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

GT_segm = np.ones((1,44,44), dtype='uint64')
GT_segm[:,:22] = 2

from affogato.affinities import compute_affinities

cremi_affs = compute_affinities(GT_segm, offsets, False, 0)[0]

cremi_affs[1,0,22,22] = 1

# cremi_affs = np.stack([affinities_x, affinities_y]).astype('float32')


# Add some random noise:
# cremi_affs += np.random.normal(0,0.01,size=cremi_affs.shape)
# cremi_affs = np.clip(cremi_affs, 0, 1)


print("test")

# offsets = [
#   [-1, 0, 0],
#   [0, -1, 0],
#   [0, 0, -1],
#   [0, -4, 0],
#   [0, 0, -4],
#   [0, -4, -4],
#   [0, 4, -4],
#   [-1, -4, 0],
#   [-1, 0, -4],
#   [-1, -4, -4],
#   [-1, 4, -4],
#   [-2, 0, 0],
#   [-3, 0, 0],
#   [-4, 0, 0],
#   [0, -8, -8],
#   [0, 8, -8],
#   [0, -12, 0],
#   [0, 0, -12]
# ]

# Reduce number of long-range edges:
offsets_prob = np.ones((len(offsets)), dtype='float32')
if "onlyLocal" in METHOD_NAME:
    offsets_prob[3:] = 0.

# print("Done")
#

import time
tick = time.time()
shape_2D = (1,) + cremi_affs.shape[2:]
graph, is_local_edge, edge_sizes = build_pixel_long_range_grid_graph_from_offsets(
    image_shape=shape_2D,
    offsets=offsets,
    offsets_probabilities=offsets_prob
)
print(time.time() - tick)

uv_ids = graph.uvIds()
nb_edges = graph.numberOfEdges
nb_nodes = graph.numberOfNodes



# Mask everything that is below 0.5:

path_label_prop_dir = os.path.join(get_home_dir(), "repositories/signed_labelpropagation")
graph_path = os.path.join(path_label_prop_dir, "signed_cremi.graph")
out_segm_path = os.path.join(path_label_prop_dir, "out_segm/signed_cremi.segm")
label_prp_exe = os.path.join(path_label_prop_dir, "deploy/label_propagation")

run_label_prop_command = "{} {} --cluster_upperbound=400 --label_propagation_iterations=100 --seed=42 --output_filename={}".format(label_prp_exe, graph_path, out_segm_path)

import subprocess

# np.random.seed(42)


def run_label_prop(graph, edge_values, get_priority, nb_iter=1, local_edges=None, size_constr=None):
    print("Start")
    if local_edges is not None:
        assert edge_values.shape == local_edges.shape
        local_edges = np.require(local_edges, dtype='bool')
    else:
        local_edges = np.ones_like(edge_values).astype('bool')
    nb_nodes = graph.numberOfNodes
    labels = np.arange(0, nb_nodes)
    sizes = np.ones((nb_nodes,))

    MAX_label = nb_nodes - 1

    iter = 0
    while iter < nb_iter:
        random_order = np.arange(0, nb_nodes)
        np.random.shuffle(random_order)
        for n in random_order:
            old_label = labels[n]
            old_size = sizes[old_label]
            stats = {}

            for neigh, edge in graph.nodeAdjacency(n):
                neigh_label = labels[neigh]
                neigh_size = sizes[neigh_label]
                if neigh_label == old_label:
                    neigh_size -= 1
                if size_constr is not None:
                    if neigh_size >= size_constr:
                        continue
                edge_is_local = local_edges[edge]
                acc_value, acc_size, neigh_is_local = stats.get(neigh_label, (0., 0., False))
                # If edge is local, save as local neighbor:
                neigh_is_local = edge_is_local or neigh_is_local
                stats[neigh_label] = (acc_value + edge_values[edge], acc_size+1, neigh_is_local)

            if old_size == 1:
                # If we had a singleton before, then I can re-use the old label
                # (if all connections turn out to be repulsive)
                max_labels_local = [old_label]
            else:
                # Otherwise, if all neighbors are repulsive, I should assign a label that was never used so far:
                max_labels_local = [MAX_label + 1]

            max = 0
            for label in stats:
                acc_value, acc_size, neigh_is_local = stats[label]
                if neigh_is_local:
                    prio = get_priority(acc_value, acc_size)
                    if prio > max:
                        max = prio
                        max_labels_local = [label]
                    elif prio == max:
                        max_labels_local.append(label)

            # Save the max label:
            if len(max_labels_local) > 1:
                selected_max_label = np.random.choice(max_labels_local, size=1)
            else:
                selected_max_label = max_labels_local[0]
            labels[n] = selected_max_label
            sizes[old_label] -= 1
            sizes[selected_max_label] += 1

            # If we introduced a new label, then increase the global max:
            if selected_max_label == MAX_label + 1:
                MAX_label += 1

        iter += 1

    return labels


def get_avg_prio(acc_value, acc_size):
    if acc_size == 0:
        return 0
    else:
        return acc_value / acc_size

def get_sum_prio(acc_value, acc_size):
    return acc_value

def compute_stacked_sp(z):
    print(z)
    edge_weights = graph.edgeValues(np.rollaxis(cremi_affs[:,[z]], 0, start=4))
    signed_edge_weights = edge_weights - 0.5

    # Convert graph:
    print("Converting graph...")
    convert_graph_to_metis_format(graph, signed_edge_weights, graph_path)

    print("Running label prop...")
    process = subprocess.Popen(run_label_prop_command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    stdout = process.communicate()[0]
    print(stdout)

    print("Saving...")
    segm_result_nodes = np.genfromtxt(out_segm_path, delimiter=',')
    label_prop_segm = segm_result_nodes.reshape(shape_2D).astype("uint32")
    max_label = label_prop_segm.max()

    return label_prop_segm, max_label

def lp_python(z, method='sum'):
    if "schulz" in METHOD_NAME:
        return compute_stacked_sp(z)
    edge_weights = graph.edgeValues(np.rollaxis(cremi_affs[:,[z]], 0, start=4))
    signed_edge_weights = edge_weights - 0.5

    if method == 'sum':
        get_prio = get_sum_prio
    elif method == 'avg':
        get_prio = get_avg_prio
    else:
        raise ValueError

    if "forceLocal" in METHOD_NAME:
        local_edges = is_local_edge
    else:
        local_edges = None
    segm_result_nodes = run_label_prop(graph, signed_edge_weights, get_prio, nb_iter=100, local_edges=local_edges)

    label_prop_segm = segm_result_nodes.reshape(shape_2D).astype("uint32")
    max_label = label_prop_segm.max()

    return label_prop_segm, max_label

METHOD_NAME += "_{}".format(LP_method)

gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True,
                          n_threads=8,
                          run_GASP_kwargs={'linkage_criteria': 'max', 'add_cannot_link_constraints': False, 'use_efficient_implementations': True},
                          )
connected_components, runtime = gasp(cremi_affs)

label_prop_segm = np.zeros(cremi_affs.shape[1:], dtype='uint32')
max_label = 0
for z in range(cremi_affs.shape[1]):
    label_prop_segm_z, max_label_z = lp_python(z, method=LP_method)
    # label_prop_segm_z, max_label_z = compute_stacked_sp(z)
    label_prop_segm[z] = label_prop_segm_z + max_label
    max_label += max_label_z

from vigra.analysis import labelVolume, relabelConsecutive

segm_relabelled = label_prop_segm
# segm_relabelled = labelVolume(label_prop_segm.astype('uint32'))
segm_relabelled = relabelConsecutive(segm_relabelled)[0]
intersect_segm = cantor_pairing_fct(connected_components, segm_relabelled)
intersect_segm = relabelConsecutive(intersect_segm)[0]

has_mistake = not np.allclose(intersect_segm, segm_relabelled)
if has_mistake:
    print("HAS MISTAKE")

segm_masked = np.zeros_like(segm_relabelled)
for label in np.unique(segm_relabelled[GT_segm==2]):
    segm_masked[segm_relabelled==label] = label

# print("Done")
# writeHDF5(label_prop_segm, "./{}.h5".format(METHOD_NAME), "segm")

import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors, mask_the_mask

fig, axes = get_figure(3,1,figsize=(24,8))
# axes.matshow(mask_the_mask(cremi_affs[0,0], value_to_mask=1), cmap='gray', alpha=0.9, interpolation='None')
# axes.matshow(mask_the_mask(cremi_affs[1,0], value_to_mask=1), cmap='gray', alpha=0.9, interpolation='None')
plot_segm(axes[0], segm_relabelled, alpha_boundary=0.0, alpha_labels=0.9, z_slice=0)
plot_segm(axes[2], mask_the_mask(segm_masked, value_to_mask=0), alpha_boundary=0.0, alpha_labels=0.9, z_slice=0)
plot_segm(axes[1], connected_components, alpha_boundary=0.0, alpha_labels=0.9, z_slice=0)
import matplotlib.patches as patches
# rect = patches.Rectangle((17.5,17.5),9,9,linewidth=1,edgecolor='black',facecolor='none')
# Add the patch to the Axes
# axes[0].add_patch(rect)
# axes.matshow(mask_the_mask(cremi_affs.min(axis=0)[0], value_to_mask=1), cmap='gray', alpha=0.6, interpolation='None')
# plot_segm(axes, segm, alpha_boundary=0.0, alpha_labels=0.9, z_slice=0)
save_plot(fig, "./new_plots_2/", "{}.png".format(METHOD_NAME))







# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# segmentation, runtime = gasp(affinities=cremi_affs)

# print(cremi_score(GT, segmentation, return_all_scores=True))
