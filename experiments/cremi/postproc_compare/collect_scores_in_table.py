import os
from copy import deepcopy

# TODO: get rid of paths
from pathutils import get_trendytukan_drive_dir

from segmfriends.utils.config_utils import assign_color_to_table_value, return_recursive_key_in_dict
import json
import numpy as np
from segmfriends.utils.various import yaml2dict
# -----------------------
# Script options:
# -----------------------

project_dir = os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare")

EXP_NAMES = [
    "subcrop_train_samples_3",
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

POSTFIX_FILE = "_mean_and_meanConstr"

LATEX_OUTPUT = False

# Index of the sorting column (ignoring the first columns given by exp-name and name of the score file
sorting_column_idx = 3

INCLUDE_EXP_NAME = False
INCLUDE_SCORE_FILENAME = False

# -------------------------------------------------------


# In order, each element is a tuple containing:
#   - name of the key in the config file
#   - type of the data (how to print it in the table
#   - number of floating digits (optional)
keys_to_collect = [
    (['postproc_config', 'sample'], 'string'),
    (['postproc_config', 'crop'], 'string'),
    (['postproc_config', 'presets_collected'], 'string'),
    (['score_WS', 'cremi-score'], 'f', 3),
    (['score_WS', 'adapted-rand'], 'f', 3),
    (['score_WS', 'vi-merge'], 'f', 3),
    (['score_WS', 'vi-split'], 'f', 3),
    (['run_GASP_runtime'], 'f', 1),
    (['full_GASP_pipeline_runtime'], 'f', 1),
    (['multicut_energy'], 'f', 0),
]


# label_names = {
#     'MutexWatershed': "Abs Max",
#     'mean': "Average",
#     "max": "Max",
#     "min": "Min",
#     "sum": "Sum",
# }

collected_results = []
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

max_nb_columns = 0

for exp_name in EXP_NAMES:
    os.path.join(project_dir, exp_name)
    scores_path = os.path.join(project_dir, exp_name, "scores")

    # Get all the configs:
    for item in os.listdir(scores_path):
        if os.path.isfile(os.path.join(scores_path, item)):
            filename = item
            if not filename.endswith(".yml") or filename.startswith("."):
                continue
            skip = False
            for char in REQUIRED_STRINGS:
                if char not in filename:
                    skip = True
                    break
            if not skip:
                for excl_string in EXCLUDE_STRINGS:
                    if excl_string in filename:
                        skip = True
                        break
                for excl_string in INCLUDE_STRINGS:
                    if excl_string in filename:
                        skip = False
                        break
            if skip:
                continue
            result_file = os.path.join(scores_path, filename)
            config = yaml2dict(result_file)

            # print(filename.replace(".yml", "").split("__"))
            # new_table_entrance = [exp_name + "__" + filename.replace(".yml", "")]
            new_table_entrance = []
            if INCLUDE_EXP_NAME:
                new_table_entrance = [exp_name]
            if INCLUDE_SCORE_FILENAME:
                 new_table_entrance += \
                                 ["{}".format(spl) for spl in filename.replace(".yml", "").split("__")]

            nb_first_columns = len(new_table_entrance)
            if nb_first_columns > max_nb_columns:
                # Add empty columns to all previous rows:
                cols_to_add = nb_first_columns - max_nb_columns
                for i, row in enumerate(collected_results):
                    collected_results[i] = row[:max_nb_columns] + ["" for _ in range(cols_to_add)] + \
                                           row[max_nb_columns:]
                max_nb_columns = nb_first_columns
            elif nb_first_columns < max_nb_columns:
                # Add empty columns only to this row:
                cols_to_add = max_nb_columns - nb_first_columns
                new_table_entrance += ["" for _ in range(cols_to_add)]

            for j, key in enumerate(keys_to_collect):
                # print(result_file)
                cell_value = return_recursive_key_in_dict(config, key[0])
                if key[1] == 'string':
                    new_table_entrance.append("{0}".format(cell_value))
                else:
                    assert len(key) == 3, "Precision is expected"
                    new_table_entrance.append("{0:.{prec}{type}}".format(cell_value, prec=key[2],
                                                                     type=key[1]))

            collected_results.append(new_table_entrance)

# nb_col, nb_rows = len(collected_results[0]), len(collected_results)
#
# collected_array = np.empty((nb_rows, nb_col), dtype="str")
# for r in range(nb_rows):
#     for c in range(nb_col):
#         collected_array[r, c] = collected_results[r][c]

# collected_results = np.array([np.array(item, dtype="str") for item in collected_results], dtype="str")

if len(collected_results) == 0:
    raise ValueError("No scores collected")
assert all(len(row) == len(collected_results[0]) for row in collected_results)
# if any(len(row) != len(collected_results[0]) for row in collected_results):
#     # Collapse first columns?
#     for i, row in enumerate(collected_results):
#         collected_results[i] = ['__'.join(row[:nb_first_columns])] + row[nb_first_columns:]
#     nb_first_columns = 1

collected_results = np.array(collected_results, dtype="str")
collected_results = collected_results[collected_results[:, sorting_column_idx + max_nb_columns].argsort()]
ID = np.random.randint(255000)
print(ID)
from segmfriends.utils.various import check_dir_and_create
if len(EXP_NAMES) == 1:
    os.path.join(project_dir, EXP_NAMES[0])
    export_dir = os.path.join(project_dir, EXP_NAMES[0], "scores", "collected_scores")
else:
    export_dir = os.path.join(project_dir, "collected_scores")
check_dir_and_create(export_dir)
# print(collected_results)
if LATEX_OUTPUT:
    np.savetxt(os.path.join(export_dir, "collected_{}{}.csv".format(ID, POSTFIX_FILE)), collected_results, delimiter=' & ',
           fmt='%s',
           newline=' \\\\\n')
else:
    np.savetxt(os.path.join(export_dir, "collected_{}{}.csv".format(ID, POSTFIX_FILE)), collected_results, delimiter='\t',
               fmt='%s',
               newline=' \n')
