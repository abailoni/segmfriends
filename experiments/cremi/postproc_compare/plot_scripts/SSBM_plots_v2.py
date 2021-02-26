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

# Build random color map:
__MAX_LABEL__ = 1000
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(__MAX_LABEL__, 3))


from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities, SegmentationFeeder

from affogato.affinities import compute_affinities


from matplotlib import rc
# rc('text', usetex=True)

project_dir = os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare_SSBM")
# exp_name = "export_agglo_data_LR0"
exp_name = "SSBM_export_data"

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
            uv_ids = segm_utils.readHDF5(data_file, "uv_ids")
            GT = segm_utils.readHDF5(data_file, "GT_labels")

        agglo_type = filename.split("__")[1]
        print("Loading ", agglo_type)
        collected[agglo_type] = data_agglo = {}

        data_agglo["final_node_labels"] = segm_utils.readHDF5(data_file, "node_labels")
        data_agglo["action_data"] = segm_utils.readHDF5(data_file, "action_data")
        # data_agglo["constraint_stats"] = segm_utils.readHDF5(data_file, "constraint_stats")
        # data_agglo["edge_ids"] = segm_utils.readHDF5(data_file, "edge_ids")
        # data_agglo["node_stats"] = segm_utils.readHDF5(data_file, "node_stats")


# -------------------
# Start making the plot:
# -------------------

font = {
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

plots_dir = os.path.join(project_dir, exp_name, "plots")
segm_utils.check_dir_and_create(plots_dir)
f, axes = plt.subplots(ncols=3, nrows=1, figsize=(48, 16))
for a in f.get_axes():
    a.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

all_clusters = False
plot_every_n_iter = 1
nb_rows_per_plotted_iter = 1
extra_spaces_between_clusters = 1


selected_GT_labels = [1,2]

GT = GT+1
assert GT.min() > 0
# Map GT labels that are not used to zero:
GT_mask = np.zeros_like(GT, dtype='bool')
for used_gt in selected_GT_labels:
    GT_mask = np.logical_or(GT_mask, GT == used_gt)
GT[np.logical_not(GT_mask)] = 0

for idx_agglo, agglo_type in enumerate(["MEAN", "SUM", "Mutex"]):
    agglo_data = collected[agglo_type]
    N = GT.shape[0]
    action_data = agglo_data["action_data"].astype('uint64')
    merged_uv_ids = action_data[action_data[:, 3] == 1, :2]  # Select only the edges that were merged

    # ------------------
    # STEP 1:
    # Create union find data-structure and re-run agglomeration using exported data:
    # ------------------
    uf = Ufd_UInt64(N)

    plot_matrix = []
    plot_matrix_iter = []

    # TODO: generalize to multiple labels
    # This array will contain what are the GT labels in each cluster: (in the first row, we simply remeber if the cluster is still alive)
    GT_stats = np.zeros((N, 3), dtype='int')
    GT_stats[:, 0] = 1
    GT_stats[GT == selected_GT_labels[0], 1] = 1
    GT_stats[GT == selected_GT_labels[1], 2] = 1
    GT_stats_per_iter = []

    segment_masks = np.eye(N, dtype='bool')
    cluster_roots = np.arange(N)
    linkage_matrix_Z = []

    for indx_iter, merged_edge in enumerate(merged_uv_ids):
        # Merge nodes and statistics
        u = uf.find(merged_edge[0])
        v = uf.find(merged_edge[1])
        uf.merge(u, v)
        alive_node = uf.find(u)
        dead_node = u if alive_node == v else v

        # Update GT stats
        GT_stats[alive_node] = GT_stats[u] + GT_stats[v]
        GT_stats[dead_node] = 0
        GT_stats_per_iter.append([GT_stats[alive_node, 1], GT_stats[alive_node, 2]])

        # Update masks:
        segment_masks[alive_node] = np.logical_or(segment_masks[alive_node], segment_masks[dead_node])
        # Build Z matrix to plot dendrogram (cluster1, cluster2, distance, size_new_cluster):
        linkage_matrix_Z.append(
            [cluster_roots[dead_node], cluster_roots[alive_node], indx_iter + 1, segment_masks[alive_node].sum()])
        cluster_roots[alive_node] = N + indx_iter

        if indx_iter % plot_every_n_iter == 0 or indx_iter == merged_uv_ids.shape[0] - 1:
            # Plot only the labels of the GT of the segment that was merged:
            GT_merged_segment = np.where(segment_masks[alive_node], GT, 0)
            plot_matrix.append(GT_merged_segment)
            # Remember the iteration number associated to each row:
            plot_matrix_iter.append(indx_iter)

    total_nb_iterations = merged_uv_ids.shape[0]
    final_segmentation = uf.find(np.arange(N))
    alive_clusters = np.unique(final_segmentation)
    assert alive_clusters.shape[0] == N - total_nb_iterations

    # ------------------
    # STEP 2:
    # Keep merging in random order until one cluster is left (for scipy hierarchy, we need full dendrogram)
    # ------------------
    for indx_iter, to_be_merged in enumerate(alive_clusters[1:]):
        # Merge nodes and statistics
        u = uf.find(alive_clusters[0])
        v = uf.find(to_be_merged)
        uf.merge(u, v)
        alive_node = uf.find(u)
        dead_node = u if alive_node == v else v

        # Update masks:
        segment_masks[alive_node] = np.logical_or(segment_masks[alive_node], segment_masks[dead_node])
        # Build Z matrix to plot dendrogram (cluster1, cluster2, distance, size_new_cluster):
        linkage_matrix_Z.append(
            [cluster_roots[dead_node], cluster_roots[alive_node], total_nb_iterations + indx_iter + 1,
             segment_masks[alive_node].sum()])
        cluster_roots[alive_node] = N + indx_iter + total_nb_iterations

    linkage_matrix_Z = np.array(linkage_matrix_Z)

    # ------------------
    # STEP 2b:
    # Compute the reordering of the leaves/nodes and re-order the columns of the plot matrix
    # to get a dendrogram-like structure:
    # ------------------
    from scipy.cluster.hierarchy import leaves_list

    leaves_order = leaves_list(linkage_matrix_Z.astype(np.double))
    plot_matrix = np.array(plot_matrix)
    plot_matrix = plot_matrix[:, leaves_order]

    GT_stats_per_iter = np.array(GT_stats_per_iter)

    # ------------------
    # STEP 3 (the messy part)
    #   - Recolor the clusters using "clustered" GT labels (blue on left, red on right)
    #   - Fill gaps in the dendrograms by looking at previous rows and only rewriting stuff associated to the new
    #         formed cluster
    #   - Plot some additional dendrogram lines
    # ------------------

    assert len(selected_GT_labels) == 2
    current_segm_inflated = np.zeros(GT.shape[0] + extra_spaces_between_clusters*(GT.shape[0]-1))
    current_segm_inflated[::extra_spaces_between_clusters+1] = GT[leaves_order]
    new_nb_rows = (plot_matrix.shape[0])*(nb_rows_per_plotted_iter+1)
    plot_matrix_inflated = np.zeros( (new_nb_rows, current_segm_inflated.shape[0]))
    plot_dendr_lines = np.ones_like(plot_matrix_inflated)
    for row_idx in range(plot_matrix.shape[0]):
        iter_index = plot_matrix_iter[row_idx]
        row = plot_matrix[row_idx]
        # Find where we have something:
        # TODO: not very efficient...
        non_zero_mask = row != 0
        non_zero_pixels = np.argwhere(non_zero_mask)
        min_non_zero_index = non_zero_pixels.min()
        max_non_zero_index = non_zero_pixels.max()
        # TODO: generalize to multiple GT
        stats = GT_stats_per_iter[iter_index]
        new_row = np.copy(non_zero_mask).astype('int') * selected_GT_labels[1]
        # Blue on the left:
        new_row[min_non_zero_index: min_non_zero_index + stats[0]] = selected_GT_labels[0]

        # Now inflate the previous segment accordingly, putting white spaces between segments:
        new_segment_size = max_non_zero_index - min_non_zero_index + 1
        inflated_segment_size = (new_segment_size - 1)*(extra_spaces_between_clusters+1) + 1
        min_inflated_non_zero_index = min_non_zero_index*(extra_spaces_between_clusters+1)
        max_inflated_non_zero_index = max_non_zero_index*(extra_spaces_between_clusters+1)
        padding = int((inflated_segment_size - new_segment_size) / 2)

        current_segm_inflated[min_inflated_non_zero_index:min_inflated_non_zero_index+padding] = 0
        current_segm_inflated[min_inflated_non_zero_index+padding:min_inflated_non_zero_index+padding+stats[0]] = selected_GT_labels[0]
        current_segm_inflated[min_inflated_non_zero_index+padding+stats[0]:min_inflated_non_zero_index+padding+new_segment_size] = selected_GT_labels[1]
        pad_left_indx = min_inflated_non_zero_index + padding + new_segment_size
        current_segm_inflated[pad_left_indx:max_inflated_non_zero_index+1] = 0


        # Deduce dendrogram lines by looking at the previous line of the plot_matrix
        if row_idx > 0:
            prev_segm_inflated = plot_matrix_inflated[(row_idx-1)*(nb_rows_per_plotted_iter+1)]
            cluster_left = np.argwhere(prev_segm_inflated[min_inflated_non_zero_index:min_inflated_non_zero_index + padding] != 0)
            if cluster_left.shape[0] > 0:
                dendgr_line_left_indx = int((cluster_left.max() - cluster_left.min())/2) + min_inflated_non_zero_index + cluster_left.min()
            else:
                dendgr_line_left_indx = min_inflated_non_zero_index

            cluster_right = np.argwhere(prev_segm_inflated[pad_left_indx:max_inflated_non_zero_index+1] != 0)
            if cluster_right.shape[0] > 0:
                dendgr_line_right_indx = int((cluster_right.max() - cluster_right.min())/2) + pad_left_indx + cluster_right.min()
            else:
                dendgr_line_right_indx = pad_left_indx

            plot_dendr_lines[row_idx*(nb_rows_per_plotted_iter+1), dendgr_line_left_indx:dendgr_line_right_indx] = 0.

        for i in range(nb_rows_per_plotted_iter+1):
            plot_matrix_inflated[row_idx*(nb_rows_per_plotted_iter+1)+i] = current_segm_inflated

    # Repeat the last segmentation some times:
    # break_space = int(plot_matrix_inflated.shape[0]*0.0)
    last_clustering_reps = int(plot_matrix_inflated.shape[0]*0.08)
    plot_matrix_inflated = np.concatenate([plot_matrix_inflated, np.tile(current_segm_inflated, reps=(last_clustering_reps,1))], axis =0)
    plot_dendr_lines = np.concatenate([plot_dendr_lines, np.ones((last_clustering_reps,plot_dendr_lines.shape[1]))], axis =0)

    # Remap colors in final plot matrix:
    # TODO: fix this nonsense
    zero_mask = plot_matrix_inflated == 0
    plot_matrix_inflated[plot_matrix_inflated == selected_GT_labels[0]] = 0
    plot_matrix_inflated[plot_matrix_inflated == selected_GT_labels[1]] = 2
    plot_matrix_inflated[zero_mask] = 1

    # Flip row order to make it look like standard dendrograms:
    plot_matrix_inflated = plot_matrix_inflated[::-1]
    plot_dendr_lines = plot_dendr_lines[::-1]

    plot_matrix_inflated_masked = mask_the_mask(plot_matrix_inflated, value_to_mask=1)

    axes[idx_agglo].matshow(np.flipud(plot_dendr_lines), cmap="gray", interpolation="None", vmin=0, vmax=1, origin="lower")
    axes[idx_agglo].matshow(np.flipud(plot_matrix_inflated_masked), cmap="bwr", interpolation="None", origin="lower")
    axes[idx_agglo].set_title(agglo_type)
f.savefig(os.path.join(plots_dir, 'new_agglo_order.png'), format='png')
