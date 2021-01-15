import numpy as np
from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets
from segmfriends.utils.graph import convert_graph_to_metis_format
import os
from pathutils import get_trendytukan_drive_dir, get_home_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint, writeHDF5

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


METHOD_NAME = "maskAffs_all_edges_test"

dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")

crop_slice = parse_data_slice("10:12,:200,:200")
# crop_slice = parse_data_slice(":")
print(get_hdf5_inner_paths(dataset))
cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice).astype('float32')
# cremi_affs = readHDF5(dataset, "affinities_dice", crop_slice=(slice(None),) + crop_slice).astype('float32')
GT = readHDF5(dataset, "GT", crop_slice=crop_slice)
raw = readHDF5(dataset, "raw", crop_slice=crop_slice)

# Add some random noise:
cremi_affs += np.random.normal(0,0.01,size=cremi_affs.shape)
cremi_affs = np.clip(cremi_affs, 0, 1)

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
offsets_prob[3:] = 1.

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

run_label_prop_command = "{} {} --cluster_upperbound=400 --label_propagation_iterations=100 --output_filename={}".format(label_prp_exe, graph_path, out_segm_path)

import subprocess

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


label_prop_segm = np.zeros(cremi_affs.shape[1:], dtype='uint32')
max_label = 0
for z in range(cremi_affs.shape[1]):
    label_prop_segm_z, max_label_z = compute_stacked_sp(z)
    label_prop_segm[z] = label_prop_segm_z + max_label
    max_label += max_label_z


# from segmfriends.utils.multi_threads import ThreadPoolExecutorStackTraced
#
# with ThreadPoolExecutorStackTraced(1) as tp:
#     tasks = [tp.submit(compute_stacked_sp, z) for z in range(cremi_affs.shape[1])]
#     results = [t.result() for t in tasks]

print("Done")
writeHDF5(label_prop_segm, "./{}.h5".format(METHOD_NAME), "segm")

import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors, mask_the_mask

fig, axes = get_figure(1,1,figsize=(8,8))
plot_segm(axes, label_prop_segm, background=raw, alpha_boundary=0.05, alpha_labels=0.5, z_slice=0)
save_plot(fig, "./new_plots_2/", "{}.png".format(METHOD_NAME))







# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# segmentation, runtime = gasp(affinities=cremi_affs)

# print(cremi_score(GT, segmentation, return_all_scores=True))
