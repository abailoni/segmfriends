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



def make_plots(project_directory, exp_name):
    scores_path = os.path.join(project_directory, exp_name, "scores")
    results_collected = collect_score_configs(
        scores_path,
        score_files_have_IDs=True,
        organize_configs_by=(('postproc_config', 'GASP_kwargs', 'run_GASP_kwargs', 'linkage_criteria'), ('postproc_config', 'GASP_kwargs', 'run_GASP_kwargs', 'add_cannot_link_constraints'),
                             ('postproc_config', 'noise_factor'))
    )

    USE_LATEX = True

    colors = {'mutex_watershed': {False: 'C0'},
              'mean': {False: 'C1',
                       True: 'C8'},
              'sum': {False: 'C2',
                      True: 'C3'},
              }

    # key_y = ['score_WS', 'vi-split']
    key_y = ['score_WS', 'adapted-rand']
    key_x = ['postproc_config', 'noise_factor']
    # key_y = ['score_WS', 'vi-split']
    # key_y = ['energy']
    # key_x = ['runtime']
    key_value = ['run_GASP_runtime']

    list_all_keys = [
        ['score_WS', 'adapted-rand'],
        # ['score_WS', "vi-merge"],
        # ['score_WS', "vi-split"],
        # ['energy'],
        # ['run_GASP_runtime']
    ]

    legend_labels = {
        'vi-merge': "VI-merge",
        'vi-split': "VI-split",
        'adapted-rand': "Rand-Score",
        'noise_factor': "\\textbf{Over-clustering} noise added to edge weights",
        'energy': 'Multicut energy'

    }

    update_rule_names = {
        'sum': "GASP Sum", 'mutex_watershed': "GASP Abs Max", 'mean': "GASP Average"
    }

    axis_ranges = {
        'vi-merge': None,
        'vi-split': None,
        'adapted-rand': [0.65, 0.98],
    }

    for all_keys in list_all_keys:
        if USE_LATEX:
            rc('text', usetex=True)
        matplotlib.rcParams.update({'font.size': 12})
        # f, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 7))

        for k, selected_edge_prob in enumerate([1., 0.1]):
            f, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 3.8))

            ax = axes
            for agglo_type in [ty for ty in ['sum', 'mutex_watershed', 'mean'] if
                               ty in results_collected]:
                for non_link in [ty for ty in [False, True] if
                                 ty in results_collected[agglo_type]]:
                    sub_dict = results_collected[agglo_type][non_link]

                    probs = []
                    VI_split_median = []
                    VI_merge = []
                    runtimes = []
                    split_max = []
                    split_min = []
                    split_q_0_25 = []
                    split_q_0_75 = []
                    error_bars_merge = []
                    counter_per_type = 0
                    for noise_factor in sub_dict:
                        multiple_VI_split = []
                        multiple_VI_merge = []
                        multiple_runtimes = []
                        for ID in sub_dict[noise_factor]:
                            data_dict = sub_dict[noise_factor][ID]

                            # TODO: update this shit
                            off_prob_1 = data_dict["postproc_config"]["GASP_kwargs"].get("offsets_probabilities", None)
                            off_prob_2 = data_dict["postproc_config"].get("edge_prob", None)
                            off_prob = off_prob_1 if off_prob_1 is not None else off_prob_2
                            if off_prob != selected_edge_prob:
                                continue

                            multiple_VI_split.append(
                                return_recursive_key_in_dict(data_dict, key_y))
                            multiple_VI_merge.append(
                                return_recursive_key_in_dict(data_dict, key_x))
                            multiple_runtimes.append(
                                return_recursive_key_in_dict(data_dict, key_value))
                            if key_y[-1] == 'adapted-rand':
                                multiple_VI_split[-1] = 1 - multiple_VI_split[-1]

                            counter_per_type += 1
                        if len(multiple_VI_split) == 0:
                            continue
                        probs.append(float(noise_factor))

                        multiple_VI_split = np.array(multiple_VI_split)
                        VI_split_median.append(np.median(multiple_VI_split))
                        split_max.append(multiple_VI_split.max())
                        split_min.append(multiple_VI_split.min())
                        split_q_0_25.append(np.percentile(multiple_VI_split, 25))
                        split_q_0_75.append(np.percentile(multiple_VI_split, 75))

                        multiple_VI_merge = np.array(multiple_VI_merge)
                        VI_merge.append(multiple_VI_merge.mean())
                        error_bars_merge.append(multiple_VI_merge.std())

                        multiple_runtimes = np.array(multiple_runtimes)
                        runtimes.append(multiple_runtimes.mean())

                        # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*edge_prob * 500,
                        #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                        #            alpha=0.3)

                    if len(probs) == 0:
                        continue
                    probs = np.array(probs)

                    split_max = np.array(split_max)
                    split_min = np.array(split_min)
                    VI_split_median = np.array(VI_split_median)
                    split_q_0_25 = np.array(split_q_0_25)
                    split_q_0_75 = np.array(split_q_0_75)

                    error_bars_merge = np.array(error_bars_merge)
                    VI_merge = np.array(VI_merge)

                    runtimes = np.array(runtimes)

                    # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                    #     continue

                    # Compose plot label:
                    plot_label_1 = update_rule_names[agglo_type]
                    plot_label_2 = " + Constraints" if non_link else " "

                    if all_keys[-1] == 'runtime':
                        error_bars_split = None

                    # if all_keys[-1] == 'energy':
                    #     values = -values

                    # print(runtimes.min(), runtimes.max())
                    # runtimes -= 0.027
                    # runtimes /= 0.2
                    # runtimes = (1 - runtimes) * 500
                    # print(runtimes.min(), runtimes.max())

                    print("Found in {} - : {} ({})".format(agglo_type, non_link, counter_per_type, k))

                    label = plot_label_1 + plot_label_2

                    argsort = np.argsort(VI_merge)
                    ax.fill_between(VI_merge[argsort], split_q_0_25[argsort],
                                    split_q_0_75[argsort],
                                    alpha=0.32,
                                    facecolor=colors[agglo_type][non_link],
                                    label=label)
                    ax.errorbar(VI_merge, VI_split_median,
                                # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                                fmt='.',
                                color=colors[agglo_type][non_link], alpha=0.5,
                                )

                    ax.plot(VI_merge[argsort], VI_split_median[argsort], '-',
                            color=colors[agglo_type][non_link], alpha=0.8)

                    # ax.plot(VI_merge[argsort], VI_split[argsort] - error_bars_split[argsort]*0.8, '-',
                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7, label=label)
                    # ax.plot(VI_merge[argsort], VI_split[argsort] + error_bars_split[argsort]*0.8, '-',
                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                    # ax.plot(VI_merge[argsort], error_bars_split[argsort],
                    #         '-',
                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7,
                    #         label=label)
                    # ax.plot(VI_merge[argsort], VI_split[argsort],
                    #         '-',
                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                    # ax.fill_between(VI_merge[argsort],
                    #                 error_bars_split[argsort],
                    #                 VI_split[argsort],
                    #                 alpha=0.4,
                    #                 facecolor=colors[agglo_type][non_link][local_attraction])

                    # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)

                    # title = "Without long-range connections" if k == 0 else "With 10% long-range connections"
                    # # ax.set_title(title)
                    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                    # ax.text(0.6, 0.1, title, transform=ax.transAxes, fontsize=10, fontweight='bold',
                    #         verticalalignment='top', bbox=props)

            # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')

            ax.set_xlim([2, 10])


            # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')

            # ax.set_xscale("log")

            # ax.set_xticks(np.arange(0, 1, step=0.1))

            # Reorder legend:
            handles, labels = ax.get_legend_handles_labels()
            # Original order: 0:Sum, 1:SumCLC, 2:MWS, 3:Mean, 4:MeanCLC
            new_ordering = [3, 0, 1, 2, 4] if k == 0 else [3, 0, 1, 2, 4]

            if len(handles) == 5:
                handles = [handles[new_indx] for new_indx in new_ordering]
                labels = [labels[new_indx] for new_indx in new_ordering]

            loc = 'best' if k == 1 else "lower left"
            ax.legend(handles, labels, loc=loc)

            # if k == 1:
            ax.set_xlabel(legend_labels[key_x[-1]])
            ax.set_ylabel(legend_labels[key_y[-1]])

            if key_x[-1] in axis_ranges:
                ax.set_xlim(axis_ranges[key_x[-1]])
            if key_y[-1] in axis_ranges:
                ax.set_ylim(axis_ranges[key_y[-1]])

            # ax.set_xlim([0.15, 0.35])

            plot_dir = os.path.join(project_directory, exp_name, "plots")
            check_dir_and_create(plot_dir)
            plt.subplots_adjust(bottom=0.15)

            # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
            f.savefig(os.path.join(plot_dir,
                                   'noise_plots_{}_{}.pdf'.format(key_y[-1], k)),
                      format='pdf')




make_plots(os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare"), "merge_biased_noise_LR01")
