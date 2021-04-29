import os
from copy import deepcopy

# TODO: get rid of paths
from pathutils import get_trendytukan_drive_dir

from segmfriends.utils.config_utils import assign_color_to_table_value, return_recursive_key_in_dict
import json
import h5py
import numpy as np
from segmfriends.utils.various import yaml2dict
# -----------------------
# Script options:
# -----------------------

project_dir = os.path.join(get_trendytukan_drive_dir(), "Desktop/cremi_experiments/")

EXP_NAMES = [
    "dice"
]

# require/exclude files with names that include certain strings
REQUIRED_STRINGS = [
]

EXCLUDE_STRINGS = [
]

INCLUDE_STRINGS = [
]

POSTFIX_FILE = "stat"
LATEX_OUTPUT = False

# Index of the sorting column (ignoring the first columns given by exp-name and name of the score file
sorting_column_idx = 0

INCLUDE_EXP_NAME = False
INCLUDE_SCORE_FILENAME = False

# -------------------------------------------------------

inner_paths_to_collect = []
collected_results = []

max_nb_columns = 0

for exp_name in EXP_NAMES:
    os.path.join(project_dir, exp_name)
    scores_path = os.path.join(project_dir, exp_name, "scores")

    # Get all the configs:
    for item in os.listdir(scores_path):
        if os.path.isfile(os.path.join(scores_path, item)):
            filename = item
            if not filename.endswith(".h5") or filename.startswith("."):
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
            f = h5py.File(result_file)

            # print(filename.replace(".yml", "").split("__"))
            # new_table_entrance = [exp_name + "__" + filename.replace(".yml", "")]
            new_table_entrance = []

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
                cell_value = return_recursive_key_in_dict(config, key[0])
                if key[1] == 'string':
                    new_table_entrance.append("{0}".format(cell_value))
                else:
                    assert len(key) == 3, "Precision is expected"
                    new_table_entrance.append("{0:.{prec}{type}}".format(cell_value, prec=key[2],
                                                                     type=key[1]))

            collected_results.append(new_table_entrance)

if len(collected_results) == 0:
    raise ValueError("No scores collected")
assert all(len(row) == len(collected_results[0]) for row in collected_results)

collected_results = np.array(collected_results, dtype="str")
collected_results = collected_results[collected_results[:, sorting_column_idx + max_nb_columns].argsort()]
ID = np.random.randint(255000)
print(ID)
from segmfriends.utils.various import check_dir_and_create
if len(EXP_NAMES) == 1:
    os.path.join(project_dir, EXP_NAMES[0])
    export_dir = os.path.join(project_dir, EXP_NAMES[0], "collected_data")
else:
    export_dir = os.path.join(project_dir, "collected_data")
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

print(os.path.join(export_dir, "collected_{}{}.csv".format(ID, POSTFIX_FILE)))
