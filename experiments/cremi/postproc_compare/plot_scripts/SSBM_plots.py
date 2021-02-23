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


font = {
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

plots_dir = os.path.join(project_dir, exp_name, "plots")
segm_utils.check_dir_and_create(plots_dir)
f, axes = plt.subplots(ncols=1, nrows=3, figsize=(36, 45))
for a in f.get_axes():
    a.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

all_clusters = False
plot_every_n_iter = 1


for idx_agglo, agglo_type in enumerate(["MEAN", "SUM", "Mutex"]):
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
    plot_matrix = np.ones((matrix_height, matrix_width))
    # This array will contain what are the GT labels in each cluster: (in the first rqo, we simply remeber if the cluster is still alive)
    GT_stats = np.zeros((N, 3), dtype='int')
    GT_stats[:, 0] = 1
    GT_stats[GT == 0, 1] = 1
    GT_stats[GT == 1, 2] = 1

    for indx_iter, merged_edge in enumerate(merged_uv_ids):
        # Merge nodes and statistics
        u = uf.find(merged_edge[0])
        v = uf.find(merged_edge[1])
        uf.merge(u,v)
        alive_node = uf.find(u)
        dead_node = u if alive_node == v else v
        GT_stats[alive_node] = GT_stats[u] + GT_stats[v]
        GT_stats[dead_node] = 0

        if indx_iter%plot_every_n_iter == 0 or indx_iter == merged_uv_ids.shape[0]-1:
            # Find all clusters that are not singletons:
            cluster_sizes = GT_stats[:,1:].sum(axis=1)
            max_GT_label = np.argmax(GT_stats[:,1:], axis=1)
            if all_clusters:
                cluster_mask = np.ones_like(cluster_sizes, dtype='bool')
            else:
                cluster_mask = cluster_sizes > 1
            # TODO: generalize to more labels...?
            NB_labels = 2
            matrix_rows = np.ones((NB_labels, matrix_width))
            for GT_lab in range(NB_labels):
                # Find and sort clusters assigned to this label:
                found_clusters = GT_stats[np.logical_and(cluster_mask, max_GT_label == GT_lab), 1:]
                found_clusters = found_clusters[np.argsort(found_clusters.sum(axis=1))[::-1]]

                # Fill the plot matrix accordingly
                col_indx = 0
                for cluster in found_clusters:
                    # First, correct assignments:
                    matrix_rows[GT_lab, col_indx:col_indx+cluster[GT_lab]] = GT_lab*2
                    col_indx += cluster[GT_lab]
                    # Next, wrong assignments:
                    # TODO: generalize to more labels
                    matrix_rows[GT_lab, col_indx:col_indx+cluster[1-GT_lab]] = (1-GT_lab)*2
                    # Now increase column index, including a space:
                    col_indx += cluster[1-GT_lab] + 1

            # Now flip the second row and merge them into the plot_matrix:
            idx_plot_matrix = int(indx_iter / plot_every_n_iter) if indx_iter != merged_uv_ids.shape[0]-1 else int(indx_iter / plot_every_n_iter)

            plot_matrix[idx_plot_matrix][matrix_rows[0]!=1] = matrix_rows[0][matrix_rows[0]!=1]
            flipped = np.flip(matrix_rows[1])
            plot_matrix[idx_plot_matrix][flipped!=1] = flipped[flipped!=1]
            last_row = plot_matrix[idx_plot_matrix]

    # At the very end, repeat the last line for 10% of the matrix height, to make the last clustering clear
    break_space = int(matrix_height*0.05)
    last_clustering_reps = int(matrix_height*0.08)

    plot_matrix = np.concatenate([plot_matrix, np.ones((break_space, plot_matrix.shape[1]))], axis=0)
    plot_matrix = np.concatenate([plot_matrix, np.tile(last_row, reps=(last_clustering_reps,1))], axis =0)



    # axes.matshow(test, cmap='seismic')
    # mask = mask_the_mask(plot_matrix, value_to_mask=1)
    axes[idx_agglo].matshow(plot_matrix, cmap='bwr', interpolation=None)
    axes[idx_agglo].set_title(agglo_type)
    # f.savefig(os.path.join(plots_dir, 'agglo_order_{}.pdf'.format(agglo_type)), format='pdf')
f.savefig(os.path.join(plots_dir, 'agglo_order.png'), format='png')
print(plots_dir)

