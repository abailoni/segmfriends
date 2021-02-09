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



project_dir = os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare")
exp_name = "debug_MWS_1"

data_dir = os.path.join(project_dir, exp_name, "out_segms")

# -------------------
# LOAD DATA:
# -------------------
collected = {}
affs = None
GT = None
for filename in os.listdir(data_dir):
    data_file = os.path.join(data_dir, filename)
    if os.path.isfile(data_file):
        if not filename.endswith('.h5') or filename.startswith("."):
            continue


        agglo_type = filename.split("__")[2]
        print("Loading ", agglo_type)
        collected[agglo_type] = data_agglo = {}

        data_agglo["segm"] = segm_utils.readHDF5(data_file, "segm")


print("DONE")
print(cremi_score(collected["Mutex_constr"]["segm"]+1,collected["Mutex"]["segm"]+1, return_all_scores=True))
print(cremi_score(collected["Mutex_constr"]["segm"]+1,collected["Mutex"]["segm"]+1, return_all_scores=True, run_connected_components=False))
print(cremi_score(collected["Mutex"]["segm"]+1,collected["MutexEfficient"]["segm"]+1, return_all_scores=True))
print(cremi_score(collected["Mutex"]["segm"]+1,collected["MutexEfficient"]["segm"]+1, return_all_scores=True, run_connected_components=False))
