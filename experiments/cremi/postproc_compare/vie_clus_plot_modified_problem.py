import numpy as np
from GASP.utils.graph import build_pixel_long_range_grid_graph_from_offsets
from segmfriends.utils.graph import convert_graph_to_metis_format
import os
from pathutils import get_trendytukan_drive_dir
from segmfriends.utils.various import readHDF5, get_hdf5_inner_paths, cremi_score, parse_data_slice, \
    convert_array_from_float_to_uint

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities
from GASP.segmentation.GASP.core import run_GASP


dataset = os.path.join(get_trendytukan_drive_dir(), "datasets/CREMI/crop_mask_emb_predictions/crop_maskEmb_affs_cremi_val_sample_C.h5")

crop_slice = parse_data_slice("10:11,:200,:200")
print(get_hdf5_inner_paths(dataset))
cremi_affs = readHDF5(dataset, "affinities_mask_average", crop_slice=(slice(None),) + crop_slice)
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
    [0, -8, -8],
    [0, 8, -8],
    [0, -12, 0],
    [0, 0, -12]
]

# Reduce number of long-range edges:
offsets_prob = np.ones((len(offsets)), dtype='float32')
offsets_prob[3:] = 1

# print("Done")
#

import time
tick = time.time()
graph, is_local_edge, edge_sizes = build_pixel_long_range_grid_graph_from_offsets(
    image_shape=cremi_affs.shape[1:],
    offsets=offsets,
    offsets_probabilities=offsets_prob
)
print(time.time() - tick)

edge_weights = graph.edgeValues(np.rollaxis(cremi_affs, 0, start=4))
nb_edges = graph.numberOfEdges
nb_nodes = graph.numberOfNodes
uv_ids = graph.uvIds()


# Mask everything that is below 0.5:
positive_edges_mask = edge_weights > 0.5
import nifty.graph as ngraph
graph = ngraph.UndirectedGraph(nb_nodes)
graph.insertEdges(uv_ids[positive_edges_mask])
edge_weights = (edge_weights[positive_edges_mask] - 0.5) * 2

edge_weights_uint = convert_array_from_float_to_uint(edge_weights, convert_to="uint8", rescale=False)

# The algorithm complains if there are edges with weight zero
edge_weights_uint[edge_weights_uint == 0] = 1

unique_ids, counts = np.unique(edge_weights_uint, return_counts=True)

# Compute equivalent multicut problem (go back to signed graph):
NODE_COORDS = (100, 100)
SELECTED_NODE = NODE_COORDS[0]*200 + NODE_COORDS[1]

wE = edge_weights_uint.sum()

# weight_matrix = np.zeros((nb_nodes, nb_nodes), dtype='uint')
# weight_matrix[uv_ids[:,0], uv_ids[:,1]] = edge_weights_uint

node_degree = np.zeros((graph.numberOfNodes), dtype='uint')
weight_matrix_row = np.zeros((graph.numberOfNodes), dtype='uint')
for node in graph.nodes():
    for nghID, edgeID in graph.nodeAdjacency(node):
        node_degree[node] += edge_weights_uint[edgeID]
        if node == SELECTED_NODE:
            weight_matrix_row[nghID] = edge_weights_uint[edgeID]


# Compute actual signed weight matrix:
signed_weighted_matrix_row = weight_matrix_row - (node_degree*node_degree[SELECTED_NODE]) / (2.*wE)
signed_weights_selected_node = signed_weighted_matrix_row.reshape(cremi_affs.shape[1:])
node_mask = np.zeros((nb_nodes))
node_mask[SELECTED_NODE] = 1
node_mask = node_mask.reshape(cremi_affs.shape[1:])

# Plot:
import matplotlib.pyplot as plt
from segmfriends.vis import plot_segm, get_figure, save_plot, plot_output_affin, plot_affs_divergent_colors
plt.rcParams.update({'font.size': 22})

fig, axes = get_figure(3,1,figsize=(24,8))
plot_segm(axes[0], GT, background=raw)
cax = axes[1].matshow(signed_weights_selected_node[0], cmap=plt.get_cmap('seismic'), interpolation='NONE')
fig.colorbar(cax,ax=axes[1])
cax = axes[2].matshow(signed_weights_selected_node[0], cmap=plt.get_cmap('seismic'), interpolation='NONE', vmin=-1, vmax=1)
fig.colorbar(cax,ax=axes[2])
# fig.colorbar(im, cax=cax, orientation='horizontal')

# axes[0,1].matshow(node_mask[0], cmap=plt.get_cmap('seismic'), interpolation='NONE')

save_plot(fig, "./", "VieClusTest_modified.png")

print("Done")

# # Convert graph:
# convert_graph_to_metis_format(graph, edge_weights_uint, "crop_pixel_graph_cremi_p01.graph")






# gasp = GaspFromAffinities(offsets, offsets_probabilities=offsets_prob, verbose=True)
#
# segmentation, runtime = gasp(affinities=cremi_affs)

# print(cremi_score(GT, segmentation, return_all_scores=True))
