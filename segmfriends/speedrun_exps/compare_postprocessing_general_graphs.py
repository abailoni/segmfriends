
from copy import deepcopy

import os
import torch
import yaml

import sys

import json
import numpy as np
import vigra

from GASP.segmentation import run_GASP
from speedrun import BaseExperiment

from segmfriends.utils import writeHDF5
from segmfriends.utils.paths import get_vars_from_argv, change_paths_config_file
from segmfriends.features import map_features_to_label_array

import segmfriends.utils.various as segm_utils
from segmfriends.utils.config_utils import adapt_configs_to_model_v2
from segmfriends.algorithms import get_segmentation_pipeline

from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
import time

try:
    import signet.block_models as block_models
    import signet.cluster as signet_cluster
except ImportError:
    block_models, signet_cluster = None, None

from sklearn.metrics import adjusted_rand_score
from scipy import sparse
from nifty.graph import UndirectedGraph


from segmfriends.speedrun_exps.compare_postprocessing_SSBM_graphs import SSBMPostProcessingExperiment



from multiprocessing.pool import ThreadPool, Pool
from itertools import repeat
from segmfriends.utils.various import starmap_with_kwargs

torch.backends.cudnn.benchmark = True


class GeneralGraphPostProcessingExperiment(SSBMPostProcessingExperiment):

    def get_kwargs_for_each_run(self):
        nb_iterations = self.get("postproc_config/nb_iterations", 1)

        kwargs_collected = []

        postproc_config = self.get("postproc_config", ensure_exists=True)
        iterated_options = self.get("postproc_config/iterated_options", {})

        # Initialize default iterated options:
        iterated_options.setdefault("preset", [postproc_config.get("preset", None)])

        # Make sure to have lists:
        for iter_key in iterated_options:
            if isinstance(iterated_options[iter_key], dict):
                for dict_key in iterated_options[iter_key]:
                    iterated_options[iter_key][dict_key] = iterated_options[iter_key][dict_key] \
                        if isinstance(iterated_options[iter_key][dict_key], list) \
                        else [iterated_options[iter_key][dict_key]]
            else:
                iterated_options[iter_key] = iterated_options[iter_key] if isinstance(iterated_options[iter_key], list) \
                    else [iterated_options[iter_key]]


        # Load the data:
        print("Loading and building graphs...")
        datasets_root_dir = self.get("general_graphs_datasets_root_dir", ensure_exists=True)
        for (dirpath, dirnames, filenames) in os.walk(datasets_root_dir):
            skip = False
            if self.get("restrict_data_to") is not None:
                restrict_data_to = self.get("restrict_data_to")
                restrict_data_to = restrict_data_to if isinstance(restrict_data_to, list) else [restrict_data_to]
                for str in restrict_data_to:
                    if str not in dirpath:
                        skip = True
                        break
            if self.get("data_to_be_ignored") is not None:
                data_to_be_ignored = self.get("data_to_be_ignored")
                data_to_be_ignored = data_to_be_ignored if isinstance(data_to_be_ignored, list) else [data_to_be_ignored]
                for str in data_to_be_ignored:
                    if str in dirpath:
                        skip = True
                        break
            if skip:
                continue
            for filename in filenames:
                if filename.endswith('.h5'):
                    data_path = os.path.join(dirpath, filename)
                    uv_ids = segm_utils.readHDF5(data_path, "edges")
                    edges_weights = segm_utils.readHDF5(data_path, "edge_weights")

                    problem_name = filename.replace(".h5", "")
                    check = datasets_root_dir
                    if not check.endswith("/"):
                        check += "/"
                    subdir = dirpath.split(check)[1]

                    # Build graph
                    nb_nodes = int(uv_ids.max() + 1)
                    graph = UndirectedGraph(nb_nodes)
                    graph.insertEdges(uv_ids.astype('uint64'))


                    # Sometimes there are double edges, so here we find them and sum the corresponding edge values:
                    if graph.numberOfEdges != edges_weights.shape[0]:
                        actual_uv_ids = graph.findEdges(uv_ids)
                        edges_weights = np.bincount(actual_uv_ids, weights=edges_weights)

                    # if update_rule == "mean" and CONVERT_TO_AFFINITIES:
                    #     used_edge_weights = costs_to_affinities(edge_weights) - 0.5
                    # else:
                    #     used_edge_weights = edge_weights

                    for _ in range(nb_iterations):
                        for preset in iterated_options['preset']:
                            new_kwargs = {
                                'preset': preset,
                                'signed_edge_weights': edges_weights,
                                'graph': graph,
                                'problem_subdir': subdir,
                                'problem_name': problem_name
                            }
                            kwargs_collected.append(new_kwargs)

        return kwargs_collected



if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, '../../experiments/cremi/postproc_compare/postproc_configs')
    experiments_path = os.path.join(source_path, 'runs')

    path_types = ["DATA_HOME", "LOCAL_DRIVE"]
    collected_paths, sys.argv = get_vars_from_argv(sys.argv, vars=path_types)

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]), path_types, collected_paths)
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]), path_types, collected_paths)
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = change_paths_config_file(os.path.join(config_path, sys.argv[ind]), path_types, collected_paths)
            i += 1
        else:
            break
    cls = GeneralGraphPostProcessingExperiment
    cls().run()
