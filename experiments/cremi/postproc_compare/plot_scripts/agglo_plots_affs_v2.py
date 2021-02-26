import numpy as np
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from pathutils import get_home_dir, get_trendytukan_drive_dir

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils import various as segm_utils
import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import collect_score_configs
from segmfriends.vis import mask_the_mask

from nifty.ufd import Ufd_UInt64

import numpy as np
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from GASP.segmentation import SizeThreshAndGrowWithWS
from pathutils import get_home_dir, get_trendytukan_drive_dir
from copy import deepcopy
import vigra

from segmfriends.features import map_features_to_label_array
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create, cremi_score
from segmfriends.utils import various as segm_utils
import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import collect_score_configs

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities, SegmentationFeeder

from affogato.affinities import compute_affinities


from matplotlib import rc
# rc('text', usetex=True)

project_dir = os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare")
# exp_name = "export_agglo_data_LR0"
exp_name = "plot_agglo_with_noisy_affs"

data_dir = os.path.join(project_dir, exp_name, "out_segms")


# -------------------
# LOAD DATA:
# -------------------
collected = {}
GT = None
for filename in os.listdir(data_dir):
    data_file = os.path.join(data_dir, filename)
    if os.path.isfile(data_file):
        if not filename.endswith('.h5') or filename.startswith(".") or "exported_data" not in filename:
            continue


        if GT is None:
            # Here we assume that the affs and GT are equal for all GASP runs collected in this foldeer:
            GT = segm_utils.readHDF5(data_file, "GT")

        agglo_type = filename.split("__")[2]
        print("Loading ", agglo_type)
        collected[agglo_type] = data_agglo = {}

        # data_agglo["final_node_labels"] = segm_utils.readHDF5(data_file, "node_labels")
        data_agglo["action_data"] = segm_utils.readHDF5(data_file, "action_data")
        # data_agglo["constraint_stats"] = segm_utils.readHDF5(data_file, "constraint_stats")
        # data_agglo["edge_ids"] = segm_utils.readHDF5(data_file, "edge_ids")
        # data_agglo["node_stats"] = segm_utils.readHDF5(data_file, "node_stats")


font = {
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

agglo_types = ["MEAN", "SUM", "Mutex"]
plots_dir = os.path.join(project_dir, exp_name, "plots")
segm_utils.check_dir_and_create(plots_dir)
f, axes = plt.subplots(ncols=1, nrows=len(agglo_types), figsize=(36, 45))
for a in f.get_axes():
    a.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

all_clusters = False
plot_every_n_iter = 5
infra_cluster_space = 10

GT = GT.flatten()

selected_GT_labels = [1,6]

assert GT.min() > 0
# Map GT labels that are not used to zero:
GT_mask = np.zeros_like(GT, dtype='bool')
for used_gt in selected_GT_labels:
    GT_mask = np.logical_or(GT_mask, GT == used_gt)
GT[np.logical_not(GT_mask)] = 0

for idx_agglo, agglo_type in enumerate(agglo_types):
    agglo_data = collected[agglo_type]
    N = GT.shape[0]
    action_data = agglo_data["action_data"].astype('uint64')
    merged_uv_ids = action_data[action_data[:,3]==1, :2] # Select only the edges that were merged

    # Create union find data-structure to re-run agglo:
    uf = Ufd_UInt64(N)

    """
    Maximum width of the plotted matrix:
        All clusters have 2 elements (plus one with 3 if odd), so a total of int(N/2) clusters, which requires
        N + int(N/2) - 1 additional spaces
    The number of rows instead, are simply equal to to the numbers of agglo_iterations
    
    For the moment, the elements will be:
        - 0 and 2 for an element that belongs on one of the two clusters
        - 1 simbolize the space between clusters 
    """
    if all_clusters:
        matrix_width = 2*N - 1
    else:
        matrix_width = N + int(N/2)-1

    matrix_height = int(merged_uv_ids.shape[0] / plot_every_n_iter) + 2
    plot_matrix = []

    # TODO: generalize to multiple labels
    # This array will contain what are the GT labels in each cluster: (in the first rqo, we simply remeber if the cluster is still alive)
    GT_stats = np.zeros((N, 3), dtype='int')
    GT_stats[:, 0] = 1
    GT_stats[GT == selected_GT_labels[0], 1] = 1
    GT_stats[GT == selected_GT_labels[1], 2] = 1


    segment_masks = np.eye(N, dtype='bool')
    cluster_roots = np.arange(N)
    linkage_matrix_Z = []

    for indx_iter, merged_edge in enumerate(merged_uv_ids):
        # Merge nodes and statistics
        u = uf.find(merged_edge[0])
        v = uf.find(merged_edge[1])
        uf.merge(u,v)
        alive_node = uf.find(u)
        dead_node = u if alive_node == v else v

        # Update GT stats
        GT_stats[alive_node] = GT_stats[u] + GT_stats[v]
        GT_stats[dead_node] = 0

        # Update masks:
        segment_masks[alive_node] = np.logical_or(segment_masks[alive_node], segment_masks[dead_node])
        # Build Z matrix to plot dendrogram (cluster1, cluster2, distance, size_new_cluster):
        linkage_matrix_Z.append([cluster_roots[dead_node], cluster_roots[alive_node], indx_iter+1, segment_masks[alive_node].sum()])
        cluster_roots[alive_node] = N + indx_iter

        if indx_iter%plot_every_n_iter == 0 or indx_iter == merged_uv_ids.shape[0]-1:
            # idx_plot_matrix = int(indx_iter / plot_every_n_iter) if indx_iter != merged_uv_ids.shape[0]-1 else int(indx_iter / plot_every_n_iter) + 1

            # Simply plot only segment that was merged, using GT labels:
            GT_merged_segment = np.where(segment_masks[alive_node], GT, 0)
            plot_matrix.append(GT_merged_segment)
            # last_row = plot_matrix[idx_plot_matrix]

    # TODO: we need to merge everything in one cluster now...
    total_nb_iterations = merged_uv_ids.shape[0]
    final_segmentation = uf.find(np.arange(N))
    alive_clusters = np.unique(final_segmentation)
    assert alive_clusters.shape[0] == N - total_nb_iterations

    # Keep merging in random order until one cluster is left:
    for indx_iter, to_be_merged in enumerate(alive_clusters[1:]):
        # Merge nodes and statistics
        u = uf.find(alive_clusters[0])
        v = uf.find(to_be_merged)
        uf.merge(u,v)
        alive_node = uf.find(u)
        dead_node = u if alive_node == v else v

        # Update masks:
        segment_masks[alive_node] = np.logical_or(segment_masks[alive_node], segment_masks[dead_node])
        # Build Z matrix to plot dendrogram (cluster1, cluster2, distance, size_new_cluster):
        linkage_matrix_Z.append([cluster_roots[dead_node], cluster_roots[alive_node], total_nb_iterations+indx_iter+1, segment_masks[alive_node].sum()])
        cluster_roots[alive_node] = N + indx_iter + total_nb_iterations

    linkage_matrix_Z = np.array(linkage_matrix_Z)

    # Compute re-ordering of the leaves and re-order the plot matrix to get a dendrogram-like structure:
    from scipy.cluster.hierarchy import leaves_list
    leaves_order = leaves_list(linkage_matrix_Z.astype(np.double))
    plot_matrix = np.array(plot_matrix)
    plot_matrix = plot_matrix[:, leaves_order]

    # Remap colors in final plot matrix:
    assert len(selected_GT_labels) == 2
    zero_mask = plot_matrix == 0
    plot_matrix[plot_matrix == selected_GT_labels[0]] = 0
    plot_matrix[plot_matrix == selected_GT_labels[1]] = 2
    plot_matrix[zero_mask] = 1


    # At the very end, repeat the last line for 10% of the matrix height, to make the last clustering clear
    break_space = int(matrix_height*0.02)
    last_clustering_reps = int(matrix_height*0.05)

    # axes.matshow(test, cmap='seismic')
    # mask = mask_the_mask(plot_matrix, value_to_mask=1)
    axes[idx_agglo].matshow(plot_matrix, cmap='bwr', interpolation=None)
    # axes[idx_agglo].matshow(plot_matrix, cmap='gray', interpolation=None)
    axes[idx_agglo].set_title(agglo_type)
    # f.savefig(os.path.join(plots_dir, 'agglo_order_{}.pdf'.format(agglo_type)), format='pdf')
f.savefig(os.path.join(plots_dir, 'new_agglo_order.png'), format='png')

