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
exp_name = "export_agglo_data_LR1_noLocalEnforce"

data_dir = os.path.join(project_dir, exp_name, "out_segms")

offset_file = os.path.join(get_home_dir(), "pyCharm_projects/segmfriends/experiments/cremi/postproc_compare/offsets/SOA_offsets.json")

import json
with open(offset_file, 'r') as f:
    offsets = json.load(f)

# -------------------
# LOAD DATA:
# -------------------
collected = {}
affs = None
GT = None
for filename in os.listdir(data_dir):
    data_file = os.path.join(data_dir, filename)
    if os.path.isfile(data_file):
        if not filename.endswith('.h5') or filename.startswith(".") or "exported_data" not in filename:
            continue


        if affs is None:
            # Here we assume that the affs and GT are equal for all GASP runs collected in this foldeer:
            affs = segm_utils.readHDF5(data_file, "affinities")
            GT = segm_utils.readHDF5(data_file, "GT")

        agglo_type = filename.split("__")[2]
        print("Loading ", agglo_type)
        collected[agglo_type] = data_agglo = {}

        data_agglo["segm"] = segm_utils.readHDF5(data_file, "segm")
        print(data_agglo["segm"].shape)
        data_agglo["merge_stats"] = segm_utils.readHDF5(data_file, "merge_stats")
        data_agglo["constraint_stats"] = segm_utils.readHDF5(data_file, "constraint_stats")
        data_agglo["edge_ids"] = segm_utils.readHDF5(data_file, "edge_ids")
        data_agglo["node_stats"] = segm_utils.readHDF5(data_file, "node_stats")

        print(cremi_score(GT,data_agglo["segm"], return_all_scores=True))

# -------------------
# Plot some stats:
# -------------------
plots_dir = os.path.join(project_dir, exp_name, "plots")
segm_utils.check_dir_and_create(plots_dir)


GT_affs, valid_GT_affs = compute_affinities(GT.astype('uint64'), offsets, False, 0)


mask_actual_used_edges = collected[list(collected.keys())[0]]["edge_ids"] != -1

def mask_data_and_flatten(data, masks=None):
    data = data.flatten()
    if masks is not None:
        masks = [masks] if not isinstance(masks, (tuple, list)) else masks
        # Combine all the masks:
        combined_masks = np.ones_like(data, dtype='bool')
        for mask in masks:
            mask = mask.flatten()
            combined_masks = np.logical_and(combined_masks, mask)
        # Do the actual masking:
        data = data[combined_masks]

    return data


# -------------------
# Make two-by-two comparisons:
# -------------------
comparisons_pairs = [
    ["MEAN", "MEANconstr"],
    ["Mutex_constr", "MEANconstr"],
    ["MEAN", "Mutex"],
    # ["MEAN", "SUM"]
]

threshold_intervals = [
    [1., 0.5],
    [0.5, 0.],
    [1., 0.9],
    [0.9, 0.7],
    [0.7, 0.5],
    [0.5, 0.3],
    [0.3, 0.1],
    [0.1, 0.],
]

colors = ['green', 'lime', 'red', 'orange']
labels = ['Correctly merged', 'Wrongly merged (under-clustering)', 'Correctly not merged', 'Wrongly not merged (over-clustering)']

for idx_thresh, thresh_pair in enumerate(threshold_intervals):
    mask_selected_edges = np.logical_and(affs<=thresh_pair[0],affs>=thresh_pair[1])
    for pair in comparisons_pairs:
        def collect_merge_stuff(data):
            was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0], [mask_selected_edges, mask_actual_used_edges])
            merge_weights = mask_data_and_flatten(data["merge_stats"][..., 1], [mask_selected_edges, mask_actual_used_edges])
            merge_sizes = mask_data_and_flatten(data["merge_stats"][..., 2], [mask_selected_edges, mask_actual_used_edges])
            return was_edge_merged, merge_weights, merge_sizes

        data0 = collect_merge_stuff(collected[pair[0]])
        data1 = collect_merge_stuff(collected[pair[1]])

        # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        fig, axs = plt.subplots(2, 2, sharey='row')
        fig.suptitle(r"$\bf{" + "Stats\ for\ edges\ with\ initial\ values\ in\ [{:.1f}, {:.1f}]".format(thresh_pair[0]-0.5, thresh_pair[1]-0.5) + r"}$")
        n_bins = 30

        # Compare merged edges:
        sum0 = np.logical_and(data0[0], np.logical_not(data1[0])).sum()
        sum1 = np.logical_and(data1[0], np.logical_not(data0[0])).sum()
        total = data0[0].shape[0]

        GT_edges = mask_data_and_flatten(GT_affs, [mask_selected_edges, mask_actual_used_edges])

        def distinguish_merged_and_not(data):
            return [[item[np.logical_and(data[0] == is_merged, GT_edges == GT_merged)] for is_merged in [1., 0.] for GT_merged in [1., 0.]] for item in data[1:]]


        data0 = distinguish_merged_and_not(data0)
        data1 = distinguish_merged_and_not(data1)

        # Histogram 1 and 2: weight of the final (or merged) edges
        axs[0,0].set_title("Value final edges for {}. \n Edges only merged here: {:.4f} %".format(pair[0], sum0/total))
        axs[0,0].hist(data0[0], bins=n_bins, stacked=True, color=colors, label=labels)
        axs[0,0].legend(prop={'size': 5})
        axs[0,1].set_title("Value final edges for {}. \n Edges only merged here: {:.4f} %".format(pair[1], sum1/total))
        axs[0,1].hist(data1[0], bins=n_bins, stacked=True, color=colors, label=labels)
        axs[1,0].set_title("Size final edges for {}.".format(pair[0]))
        axs[1,0].hist(data0[1], bins=n_bins, stacked=True, color=colors, label=labels)
        axs[1,1].set_title("Size final edges for {}.".format(pair[1]))
        axs[1,1].hist(data1[1], bins=n_bins, stacked=True, color=colors, label=labels)

        # fig.savefig(os.path.join(plots_dir, 'merge_stats_{}_{}_{:.1f}_{:.1f}.pdf'.format(pair[0], pair[1], thresh_pair[0]-0.5, thresh_pair[1]-0.5)), format='pdf')
        fig.savefig(os.path.join(plots_dir, 'merge_stats_{}_{}_{}.pdf'.format(pair[0], pair[1], idx_thresh)), format='pdf')
        plt.close(fig)


        if "constr" in pair[0] and "constr" in pair[1]:
            def collect_constr_stuff(data):
                was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0],
                                                        [mask_selected_edges, mask_actual_used_edges,data["constraint_stats"][..., 0]>0])
                constrained_value = mask_data_and_flatten(data["constraint_stats"][..., 1],
                                                      [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                constrained_sum = mask_data_and_flatten(data["constraint_stats"][..., 0],
                                                    [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                GT_edges = mask_data_and_flatten(GT_affs, [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                return [constrained_value, constrained_sum, was_edge_merged, GT_edges]

            data0 = collect_constr_stuff(collected[pair[0]])
            data1 = collect_constr_stuff(collected[pair[1]])

            # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
            fig, axs = plt.subplots(2, 2, sharey='row', sharex='row')
            fig.suptitle(
                r"$\bf{" + "Stats\ for\ CONSTRAINED\ edges\ with\ initial\ values\ in\ [{:.1f}, {:.1f}]".format(thresh_pair[
                                                                                                              0] - 0.5,
                                                                                                          thresh_pair[
                                                                                                              1] - 0.5) + r"}$")

            def distinguish_merged_and_not(data):
                return [[item[np.logical_and(data[2] == is_merged, data[3] == GT_merged)] for is_merged in [1., 0.] for
                         GT_merged in [1., 0.]] for item in data[:2]]


            # Compute average:
            data0[0] = data0[0]/data0[1]
            data1[0] = data1[0]/data1[1]

            data0 = distinguish_merged_and_not(data0)
            data1 = distinguish_merged_and_not(data1)

            axs[0,0].set_title("Avg constraint values for {}".format(pair[0]))
            axs[0,0].hist(data0[0], bins=n_bins, histtype='bar', stacked=True, color=colors, label=labels)
            axs[0,0].legend(prop={'size': 5})
            axs[0,1].set_title("Avg constraint values for {}".format(pair[1]))
            axs[0,1].hist(data1[0], bins=n_bins, histtype='bar', stacked=True, color=colors, label=labels)
            axs[1,0].set_title("Size constraints for {}".format(pair[0]))
            axs[1,0].hist(data0[1], bins=n_bins, histtype='bar', stacked=True, color=colors, label=labels)
            axs[1,1].set_title("Size constraints for {}".format(pair[1]))
            axs[1,1].hist(data1[1], bins=n_bins, histtype='bar', stacked=True, color=colors, label=labels)


            # fig.savefig(os.path.join(plots_dir, 'merge_stats_{}_{}_{:.1f}_{:.1f}.pdf'.format(pair[0], pair[1], thresh_pair[0]-0.5, thresh_pair[1]-0.5)), format='pdf')
            fig.savefig(os.path.join(plots_dir, 'constr_stats_{}_{}_{}.pdf'.format(pair[0], pair[1], idx_thresh)),
                        format='pdf')
            plt.close(fig)


#################
# DIFFERENCE PLOTS:
#################

for idx_thresh, thresh_pair in enumerate(threshold_intervals):
    mask_selected_edges = np.logical_and(affs<=thresh_pair[0],affs>=thresh_pair[1])
    for pair in comparisons_pairs:
        def collect_merge_stuff(data):
            was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0], [mask_selected_edges, mask_actual_used_edges])
            merge_weights = mask_data_and_flatten(data["merge_stats"][..., 1], [mask_selected_edges, mask_actual_used_edges])
            merge_sizes = mask_data_and_flatten(data["merge_stats"][..., 2], [mask_selected_edges, mask_actual_used_edges])
            return [was_edge_merged, merge_weights, merge_sizes]

        data0 = collect_merge_stuff(collected[pair[0]])
        data1 = collect_merge_stuff(collected[pair[1]])

        original_weights = mask_data_and_flatten(affs-0.5, [mask_selected_edges, mask_actual_used_edges])
        GT_edges = mask_data_and_flatten(GT_affs, [mask_selected_edges, mask_actual_used_edges])

        # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        fig, axs = plt.subplots(2, 1, sharey='all', sharex='all')
        fig.suptitle(r"$\bf{" + "Stats\ for\ edges\ with\ initial\ values\ in\ [{:.1f}, {:.1f}]".format(thresh_pair[0]-0.5, thresh_pair[1]-0.5) + r"}$")
        n_bins = 30

        # Compare merged edges:
        sum0 = np.logical_and(data0[0], np.logical_not(data1[0])).sum()
        sum1 = np.logical_and(data1[0], np.logical_not(data0[0])).sum()
        total = data0[0].shape[0]


        def distinguish_merged_and_not(data):
            return [[item[np.logical_and(data[0] == is_merged, GT_edges == GT_merged)] for is_merged in [1., 0.] for
                     GT_merged in [1., 0.]] for item in data[1:2]]


        data0[1] = data0[1] - original_weights
        data1[1] = data1[1] - original_weights
        data0 = distinguish_merged_and_not(data0)
        data1 = distinguish_merged_and_not(data1)


        # Histogram 1 and 2: weight of the final (or merged) edges
        axs[0].set_title("Did final edge values for {} increase wrt original?".format(pair[0]))
        axs[0].hist(data0[0], bins=n_bins, stacked=True, color=colors, label=labels)
        axs[0].legend(prop={'size': 5})
        axs[1].set_title("Did final edge values for {} increase wrt original?".format(pair[1]))
        axs[1].hist(data1[0], bins=n_bins, stacked=True, color=colors, label=labels)
        # axs[1,0].set_title("Difference value comparison: Did {} increase more wrt {}?".format(pair[0], pair[1]))
        # axs[1,0].hist(data0[1]-data1[1], bins=n_bins)

        # fig.savefig(os.path.join(plots_dir, 'merge_stats_{}_{}_{:.1f}_{:.1f}.pdf'.format(pair[0], pair[1], thresh_pair[0]-0.5, thresh_pair[1]-0.5)), format='pdf')
        fig.savefig(os.path.join(plots_dir, 'relative_merge_stats_{}_{}_{}.pdf'.format(pair[0], pair[1], idx_thresh)), format='pdf')
        plt.close(fig)


        if "constr" in pair[0] and "constr" in pair[1]:
            def collect_constr_stuff(data):
                was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0],
                                                        [mask_selected_edges, mask_actual_used_edges,
                                                         data["constraint_stats"][..., 0] > 0])
                constrained_value = mask_data_and_flatten(data["constraint_stats"][..., 1],
                                                      [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                constrained_sum = mask_data_and_flatten(data["constraint_stats"][..., 0],
                                                    [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                original_weights_constr = mask_data_and_flatten(affs-0.5, [mask_selected_edges, mask_actual_used_edges, data["constraint_stats"][..., 0]>0])
                GT_edges = mask_data_and_flatten(GT_affs, [mask_selected_edges, mask_actual_used_edges,
                                                           data["constraint_stats"][..., 0] > 0])
                return [constrained_value, constrained_sum, original_weights_constr, was_edge_merged, GT_edges]

            data0 = collect_constr_stuff(collected[pair[0]])
            data1 = collect_constr_stuff(collected[pair[1]])

            def distinguish_merged_and_not(data):
                return [[item[np.logical_and(data[3] == is_merged, data[4] == GT_merged)] for is_merged in [1., 0.] for
                         GT_merged in [1., 0.]] for item in data[:2]]

            # Compute average:
            data0[0] = data0[0]/data0[1]-data0[2]
            data1[0] = data1[0]/data1[1]-data1[2]

            data0 = distinguish_merged_and_not(data0)
            data1 = distinguish_merged_and_not(data1)


            # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
            fig, axs = plt.subplots(2, 1, sharey='all', sharex='all')
            fig.suptitle(
                r"$\bf{" + "Stats\ for\ CONSTRAINED\ edges\ with\ initial\ values\ in\ [{:.1f}, {:.1f}]".format(thresh_pair[
                                                                                                              0] - 0.5,
                                                                                                          thresh_pair[
                                                                                                              1] - 0.5) + r"}$")
            axs[0].set_title("How much more negative was the avg constraint value wrt the original edge value (for {})?".format(pair[0]))
            axs[0].hist(data0[0], bins=n_bins, stacked=True, color=colors, label=labels)
            axs[0].legend(prop={'size': 5})
            axs[1].set_title("How much more negative was the avg constraint value wrt the original edge value (for {})?".format(pair[1]))
            axs[1].hist(data1[0], bins=n_bins, stacked=True, color=colors, label=labels)


            # fig.savefig(os.path.join(plots_dir, 'merge_stats_{}_{}_{:.1f}_{:.1f}.pdf'.format(pair[0], pair[1], thresh_pair[0]-0.5, thresh_pair[1]-0.5)), format='pdf')
            fig.savefig(os.path.join(plots_dir, 'relative_constr_stats_{}_{}_{}.pdf'.format(pair[0], pair[1], idx_thresh)),
                        format='pdf')
            plt.close(fig)


# # Check how many of the strong edges are constrained/merged and when:
# HIGH_MERGE_THRESH = 0.99
# strong_merge_affs_mask = affs > HIGH_MERGE_THRESH
#
# for agglo_type in collected:
#     data = collected[agglo_type]
#     merge_weights = data["merge_stats"][..., 1]
#     merge_sizes = data["merge_stats"][..., 2]
#
#     masks = [mask_actual_used_edges, strong_merge_affs_mask, data["merge_stats"][..., 0] == 1.]
#     merge_weights = mask_data_and_flatten(merge_weights, masks)
#     merge_sizes = mask_data_and_flatten(merge_sizes, masks)
#
#     was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0], [mask_actual_used_edges, strong_merge_affs_mask])
#     ratio_merged_edges = was_edge_merged.sum() / was_edge_merged.shape[0]
#
#     fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#     fig.suptitle("{} percent of edges with initial value >{} were merged".format(ratio_merged_edges*100, HIGH_MERGE_THRESH))
#     n_bins= 20
#     axs[0].hist(merge_weights, bins=n_bins)
#     axs[1].hist(merge_sizes, bins=n_bins)
#
#     fig.savefig(os.path.join(plots_dir, 'merge_stats_{}.pdf'.format(agglo_type)), format='pdf')
#
#
# In the constrained version, check how many constraints became positive:
# for agglo_type in collected:
#     if "constr" in agglo_type:
#         data = collected[agglo_type]
#         print("\n",agglo_type)
#         print("Among all constrained edges:")
#         was_edge_merged = mask_data_and_flatten(data["merge_stats"][..., 0] > 0, [mask_actual_used_edges])
#         was_edge_constrained = mask_data_and_flatten(data["constraint_stats"][..., 0] > 0, [mask_actual_used_edges])
#         edge_became_positive = mask_data_and_flatten(data["constraint_stats"][..., 4] > 0, [mask_actual_used_edges])
#
#         constr_size = mask_data_and_flatten(data["constraint_stats"][..., 0], [mask_actual_used_edges])
#         constr_value = mask_data_and_flatten(data["constraint_stats"][..., 1], [mask_actual_used_edges, data["constraint_stats"][..., 0] > 0])
#         positive_value = mask_data_and_flatten(data["constraint_stats"][..., 5],
#                                              [mask_actual_used_edges, data["constraint_stats"][..., 4] > 0])
#
#
#         was_edge_constrained_and_merged = np.logical_and(was_edge_constrained, was_edge_merged)
#         print("Percentage of edges that was constrained and then merged: {}".format(was_edge_constrained_and_merged.sum()/ was_edge_constrained.sum()))
#
#         was_positive_but_not_merged = np.logical_and(np.logical_not(was_edge_merged), edge_became_positive)
#         print("Percentage of edges that was constrained, became positive and then negative again: {}".format(was_positive_but_not_merged.sum()/ was_edge_constrained.sum()))
#         print("(The remaining ones were all constrained, not merged and stayed always negative)")

