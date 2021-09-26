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
        organize_configs_by=(('postproc_config', 'presets_collected'),
                             ('postproc_config', 'SSBM_kwargs', 'etain'))
    )

    key_x = ['eta']
    key_y = ['RAND_score']
    key_value = ['scores', 'adapted-rand']

    legend_axes = {
        'eta': "Amount of flip-noise (flip-probability $\eta$)",
        'RAND_score': "ARAND Error",
        'runtime': "runtime",
    }

    # Find best values for every crop:
    from matplotlib import rc
    # rc('font', **{'family': 'serif', 'serif': ['Times']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    # matplotlib.rcParams['mathtext.fontset'] = 'stix'

    matplotlib.rcParams.update({'font.size': 24})
    ncols, nrows = 1, 1
    f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 6.5))
    ax = all_ax

    label_names = []

    all_method_descriptors = ["SUM", "SPONGE_sym", "MEAN", "Mutex", "SPONGE"]

    colors = {'SPONGE_sym': 'C7',
              'SUM': 'C2',
              # 'sumTrue': 'C3',
              'MEAN': 'C1',
              'Mutex': 'C0',
              # 'L-sym': 'C4',
              'SPONGE': 'C5',
              # 'BNC': 'C6',
              # 'kernighanLin': 'C2',
              }

    # methods = {'SPONGE-sym': 'spectral',
    #            'sumFalse': 'GASP',
    #            'sumTrue': 'GASP',
    #            'meanFalse': 'GASP',
    #            'abs_maxFalse': 'GASP',
    #            'L-sym': 'spectral',
    #            'SPONGE': 'spectral',
    #            'BNC': 'spectral',
    #            'kernighanLin': 'multicut'
    #            }

    labels = {'SPONGE_sym': 'SPONGE$_{sym}$',
              'SUM': 'GAEC',
              # 'sumTrue': 'GASP Sum + CLC',
              'MEAN': 'HC-Avg',
              'Mutex': 'MWS',
              # 'L-sym': '$L_{sym}$',
              'SPONGE': 'SPONGE',
              # 'BNC': 'BNC ',
              # 'kernighanLin': 'MC',
              }

    type_counter = 0
    for method_descriptor in all_method_descriptors:
        if method_descriptor not in results_collected:
            continue
        sub_dict = results_collected[method_descriptor]
        values = []
        etas = []
        nb_iterations = []

        print(method_descriptor)
        for eta in sub_dict:
            multiple_values = []
            for ID in sub_dict[eta]:
                data_dict = sub_dict[eta][ID]
                multiple_values.append(return_recursive_key_in_dict(data_dict, key_value))
            if len(multiple_values) == 0:
                continue
            multiple_values = np.array(multiple_values)
            median = np.median(multiple_values)
            p_25 = np.percentile(multiple_values, 25)
            p_75 = np.percentile(multiple_values, 75)
            values.append([median, p_25, p_75])
            etas.append(eta)
            nb_iterations.append(multiple_values.shape[0])

        print(np.array(nb_iterations).mean())
        # Sort keys:
        etas = np.array(etas)
        values = np.array(values)
        argsort = np.argsort(etas, axis=0)

        ax.fill_between(etas[argsort], values[:, 1][argsort],
                        values[:, 2][argsort],
                        alpha=0.32,
                        facecolor=colors[method_descriptor],
                        label=labels[method_descriptor])

        ax.errorbar(etas, values[:, 0],
                    # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                    fmt='.',
                    color=colors[method_descriptor], alpha=0.5,
                    )

        ax.plot(etas[argsort], values[:, 0][argsort], '-',
                color=colors[method_descriptor], alpha=0.8)
        type_counter += 0

    ax.set_xlabel(legend_axes[key_x[-1]])
    ax.set_ylabel(legend_axes[key_y[-1]])
    if "k50" in exp_name:
        lgnd = ax.legend(prop={'size': 17}, loc="lower right")

    # ax.set_yscale("log", nonposy='clip')
    # f.subplots_adjust(bottom=0.2)

    # for i in range(10):
    #     try:
    #         lgnd.legendHandles[i]._sizes = [30]
    #     except IndexError:
    #         break

    # ax.set_title("CREMI training sample {}".format(sample))

    # if sample == "B":
    #     ax.set_ylim([0.080, 0.090])
    # else:
    # ax.autoscale(enable=True, axis='both')
    # ax.set_ylim([2e-4, 1.])

    # if all_keys[-1] == 'runtime':
    #     ax.set_yscale("log", nonposy='clip')

    # plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

    plot_dir = os.path.join(project_directory, exp_name, "plots")
    check_dir_and_create(plot_dir)

    f.tight_layout(rect=[0, 0, 1, 1])
    pdf_name = "summary_SSBM_experiments_k50.pdf" if exp_name == "SSBM_spectral_compare_k50_c" else "summary_SSBM_experiments_k20.pdf"

    # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
    f.savefig(os.path.join(plot_dir,
                           pdf_name),
              format='pdf')




make_plots(os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare_SSBM"), "SSBM_spectral_compare_k50_c")
make_plots(os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare_SSBM"), "SSBM_spectral_compare_k20")
