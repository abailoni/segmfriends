import os
from copy import deepcopy

# TODO: get rid of paths
from pathutils import get_trendytukan_drive_dir

from segmfriends.utils.config_utils import assign_color_to_table_value, return_recursive_key_in_dict
import json
import numpy as np
from segmfriends.utils.various import yaml2dict
from segmfriends.utils.config_utils import collect_score_configs
# -----------------------
# Script options:
# -----------------------

project_dir = os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare_general_graphs")

used_agglo_types = \
    ["SUM", "SUMconstr", "MutexGraphEff", "MEAN", "MEANconstr", "MAX", "MAXconstr", "MIN", "MINconstr"]
    # ["SUM", "SUMconstr", "MutexGraphEff", "MEAN", "MEANconstr", "MAX", "MIN", "MINconstr"]
    # ["SUM", "SUMconstr", "MutexGraphEff", "MEAN", "MEANconstr"]

EXP_NAMES = [
    "compare_energies",
]

REQUIRED_STRINGS = [
    # "_mergedGlia",
    # "affs_withLR_z"
]

EXCLUDE_STRINGS = [
    # "_mergedGlia",
    # "multicut_kerLin",
    # "multicut_exact",
    # "affs_noLR",
    # "plusGliaMask2",
    # "MEAN_affs",
]

INCLUDE_STRINGS = [
]

POSTFIX_FILE = "_all"

COMPUTE_REL_DIFF= False

LATEX_OUTPUT = True

# Options for coloring:
COLOR_SCORES = False
GOOD_THRESH = 100
BAD_THRESH = 1000

# Index of the sorting column (ignoring the first columns given by exp-name and name of the score file
sorting_column_idx = 0

INCLUDE_EXP_NAME = False
INCLUDE_SCORE_FILENAME = False

# -------------------------------------------------------


# In order, each element is a tuple containing:
#   - name of the key in the config file
#   - type of the data (how to print it in the table
#   - number of floating digits (optional)
key_to_collect = \
    (['multicut_energy'], 'f', 0)
    # (['runtime'], 'f', 4)
    # (['postproc_config', 'sample'], 'string'),
    # (['postproc_config', 'crop'], 'string'),
    # (['postproc_config', 'presets_collected'], 'string'),
    # (['score_WS', 'cremi-score'], 'f', 3),
    # (['score_WS', 'adapted-rand'], 'f', 3),
    # (['score_WS', 'vi-merge'], 'f', 3),
    # (['score_WS', 'vi-split'], 'f', 3),
    # (['run_GASP_runtime'], 'f', 1),
    # (['full_GASP_pipeline_runtime'], 'f', 1),
    # (['multicut_energy'], 'f', 0),
# ]


# label_names = {
#     'MutexWatershed': "Abs Max",
#     'mean': "Average",
#     "max": "Max",
#     "min": "Min",
#     "sum": "Sum",
# }



rows_table_collected = []
# energies, ARAND = [], []
# SEL_PROB = 0.1


# for exp_name in EXP_NAMES:
#     os.path.join(project_dir, exp_name)
#     scores_path = os.path.join(project_dir, exp_name, "out_segms")
#     import shutil
#
#     # Get all the configs:
#     for item in os.listdir(scores_path):
#         if os.path.isfile(os.path.join(scores_path, item)):
#             filename = item
#             if not filename.endswith(".h5") or filename.startswith("."):
#                 continue
#             result_file = os.path.join(scores_path, filename)
#             new_filename = result_file.replace("_fullGT.", "__fullGT.")
#             new_filename = new_filename.replace("_ignoreGlia.", "__ignoreGlia.")
#             shutil.move(result_file, new_filename)

max_nb_columns = 1

for exp_name in EXP_NAMES:
    exp_dir = os.path.join(project_dir, exp_name)
    table_header_built = False
    for dataset_name in os.listdir(exp_dir):
        sub_directory = os.path.join(exp_dir, dataset_name)

        if dataset_name == "fruitfly-large":
            continue


        if os.path.isdir(sub_directory):
            scores_path = os.path.join(sub_directory, "scores")
            if not os.path.exists(scores_path):
                continue
            results_collected = collect_score_configs(
                scores_path,
                score_files_have_IDs=False,
                organize_configs_by=(('postproc_config', 'presets_collected'),
                                     ('problem_name',))
            )

            if len(results_collected) ==0:
                continue

            # Build headers:
            if not table_header_built:
                if used_agglo_types is not None:
                    all_agglo_types = used_agglo_types
                else:
                    all_agglo_types = list(results_collected.keys())
                nb_agglos = len(all_agglo_types)
                first_row = [""] + all_agglo_types
                if COMPUTE_REL_DIFF:
                    # In this case add a last column
                    first_row += ["Min value"]
                rows_table_collected.append(first_row)
                table_header_built = True


            new_table_entrance = []
            number_problems = None
            for agglo_type in all_agglo_types:
                if number_problems is None:
                    number_problems = len(results_collected[agglo_type])
                else:
                    assert number_problems == len(results_collected[agglo_type]) , "Some problems were solved for some agglo types but not for others!"

                # Take the average over the collected values:
                values_collected = []
                for problem in results_collected[agglo_type]:
                    config = results_collected[agglo_type][problem]
                    config = config[list(config.keys())[0]]
                    new_value = return_recursive_key_in_dict(config, key_to_collect[0])
                    values_collected.append(new_value)

                values_collected = np.array(values_collected).mean()
                new_table_entrance += [values_collected]

            if COMPUTE_REL_DIFF:
                all_values = np.array(new_table_entrance)
                min_value = all_values.min()
                new_table_entrance = [vl - min_value for vl in new_table_entrance]
                # Add last column with minimum value:
                new_table_entrance += [min_value]

            # Format the numbers as strings:
            if LATEX_OUTPUT and COLOR_SCORES:
                # In this case we color the results:
                new_table_entrance = [
                    assign_color_to_table_value(cell_value, good_thresh=GOOD_THRESH,
                                                bad_thresh=BAD_THRESH,
                                                nb_flt=key_to_collect[2])
                    for cell_value in new_table_entrance
                ]
            else:
                new_table_entrance = [
                    "{0:.{prec}{type}}".format(cell_value, prec=key_to_collect[2],
                                               type=key_to_collect[1]) for cell_value in new_table_entrance
                ]

            new_table_entrance = [dataset_name] + new_table_entrance
            rows_table_collected.append(new_table_entrance)

# nb_col, nb_rows = len(collected_results[0]), len(collected_results)
#
# collected_array = np.empty((nb_rows, nb_col), dtype="str")
# for r in range(nb_rows):
#     for c in range(nb_col):
#         collected_array[r, c] = collected_results[r][c]

# collected_results = np.array([np.array(item, dtype="str") for item in collected_results], dtype="str")

collected_results = np.array(rows_table_collected, dtype="str")
# collected_results = collected_results[collected_results[:, sorting_column_idx + max_nb_columns].argsort()]
ID = np.random.randint(255000)
print(ID)
from segmfriends.utils.various import check_dir_and_create
if len(EXP_NAMES) == 1:
    os.path.join(project_dir, EXP_NAMES[0])
    export_dir = os.path.join(project_dir, EXP_NAMES[0], "collected_scores")
else:
    export_dir = os.path.join(project_dir, "collected_scores")
check_dir_and_create(export_dir)
# print(collected_results)
write_path = os.path.join(export_dir, "summarized_{}{}.csv".format(ID, POSTFIX_FILE))
print(write_path)
if LATEX_OUTPUT:
    np.savetxt(write_path, collected_results, delimiter=' & ',
           fmt='%s',
           newline=' \\\\\n')
else:
    np.savetxt(write_path, collected_results, delimiter=',',
               fmt='%s',
               newline=' \n')
