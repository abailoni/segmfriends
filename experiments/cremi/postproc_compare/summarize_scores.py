import os
from copy import deepcopy


from segmfriends.utils.config_utils import assign_color_to_table_value, return_recursive_key_in_dict
import json
import numpy as np
from segmfriends.utils.various import yaml2dict
from segmfriends.utils.config_utils import collect_score_configs
# -----------------------
# Script options:
# -----------------------

project_dir = "/scratch/bailoni/projects/gasp/"

EXP_NAMES = [
    # "train_samples_LR01_pixels",
    "train_samples_SP_3"
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

POSTFIX_FILE = "_average"

LATEX_OUTPUT = False

# Index of the sorting column (ignoring the first columns given by exp-name and name of the score file
sorting_column_idx = 0

INCLUDE_EXP_NAME = False
INCLUDE_SCORE_FILENAME = False

# -------------------------------------------------------


# In order, each element is a tuple containing:
#   - name of the key in the config file
#   - type of the data (how to print it in the table
#   - number of floating digits (optional)
keys_to_collect = [
    # (['postproc_config', 'sample'], 'string'),
    # (['postproc_config', 'crop'], 'string'),
    # (['postproc_config', 'presets_collected'], 'string'),
    # (['score_WS', 'cremi-score'], 'f', 3),
    (['score', 'adapted-rand'], 'f', 4),
    (['score', 'vi-split'], 'f', 3),
    (['score', 'vi-merge'], 'f', 3),
    # (['run_GASP_runtime'], 'f', 1),
    (['full_GASP_pipeline_runtime'], 'f', 0),
    # (['multicut_energy'], 'f', 0),
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

max_nb_columns = 1

for exp_name in EXP_NAMES:

    scores_path = os.path.join(project_dir, exp_name, "scores")
    results_collected = collect_score_configs(
        scores_path,
        score_files_have_IDs=True,
        organize_configs_by=(
            # ('postproc_config', 'sample'),
            ('postproc_config', 'presets_collected'),
                              # ('postproc_config', 'nb_nodes')
                             ),
        # files_to_be_exlcuded=["MINconstr"]
        # restrict_files_to=['MINconstr']
    )

    # results_collected = results_collected["C"]
    # Build headers:
    new_table_entrance = ["Agglo type"]
    for j, key in enumerate(keys_to_collect):
        new_table_entrance.append(key[0][-1])
    collected_results.append(new_table_entrance)
    for agglo_name in results_collected:
        new_table_entrance = [agglo_name]
        values = [[] for _ in keys_to_collect]
        for ID in results_collected[agglo_name]:
            config = results_collected[agglo_name][ID]

            for j, key in enumerate(keys_to_collect):
                # print(result_file)
                cell_value = return_recursive_key_in_dict(config, key[0])
                values[j].append(cell_value)

        values = np.array(values)
        mean_values = values.mean(axis=1)
        new_table_entrance += [
            "{0:.{prec}{type}}".format(mean_values[j], prec=key[2],
                                       type=key[1])
             for j, key in enumerate(keys_to_collect)]
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
