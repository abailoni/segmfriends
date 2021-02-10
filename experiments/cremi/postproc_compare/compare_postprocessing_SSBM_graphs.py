
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
from nifty.tools import fromAdjMatrixToEdgeList, fromEdgeListToAdjMatrix
from nifty.graph import UndirectedGraph



from multiprocessing.pool import ThreadPool, Pool
from itertools import repeat
from segmfriends.utils.various import starmap_with_kwargs

torch.backends.cudnn.benchmark = True

def from_adj_matrix_to_edge_list(sparse_adj_matrix):
    # sparse_adj_matrix.setdiag(np.zeros(sparse_adj_matrix.shape[0], sparse_adj_matrix.dtype))
    nb_edges = sparse_adj_matrix.count_nonzero()
    if not isinstance(sparse_adj_matrix, np.ndarray):
        sparse_adj_matrix = sparse_adj_matrix.toarray()
    # sh = sparse_adj_matrix.shape[0]
    # nb_edges = int((sh*sh - sh) / 2)
    edge_list = np.empty((nb_edges, 3))

    # Set diagonal elements to zero, we don't care about them:
    real_nb_edges = fromAdjMatrixToEdgeList(sparse_adj_matrix, edge_list, 1)
    edge_list = edge_list[:real_nb_edges]
    uvIds = edge_list[:,:2].astype('uint64')
    edge_weights = edge_list[:,2].astype('float32')
    return uvIds, edge_weights


def from_edge_list_to_adj_matrix(uvIds, edge_weights):
    edge_list = np.concatenate((uvIds, np.expand_dims(edge_weights, axis=-1)),
                               axis=1)
    nb_nodes = int(edge_list[:,:2].max()+1)
    A_p = np.zeros((nb_nodes, nb_nodes))
    A_n = np.zeros((nb_nodes, nb_nodes))
    fromEdgeListToAdjMatrix(A_p, A_n, edge_list, 1)
    return A_p, A_n


class SSBMPostProcessingExperiment(BaseExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(SSBMPostProcessingExperiment, self).__init__(experiment_directory)
        # Privates
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.auto_setup()

        assert signet_cluster is not None, "signet module is needed for SSBM experiments " \
                                   "https://github.com/alan-turing-institute/SigNet"

        # Where the segmentation will be saved: (each experiment is in a sub-folder)
        main_output_dir = self.get('main_output_dir', ensure_exists=True)
        self.set("main_output_dir", main_output_dir)

        # Load the file with all the presets of postprocessing:
        postproc_presets_path = self.get('postproc_config/postproc_presets_file_path', ensure_exists=True)
        self.set("postproc_presets", segm_utils.yaml2dict(postproc_presets_path))

        self.build_experiment()

    def build_experiment(self):
        # Create dirs if not existing:
        exp_name = self.get("experiment_name", ensure_exists=True)
        exp_path = os.path.join(self.get("main_output_dir", ensure_exists=True), exp_name)
        self.set("exp_path", exp_path)
        exp_path_exists = segm_utils.check_dir_and_create(exp_path)

    def run(self):
        # Load data for all the runs:
        nb_thread_pools = self.get("postproc_config/nb_thread_pools")
        print("Pools: ", nb_thread_pools)
        kwargs_iter = self.get_kwargs_for_each_run()
        print("Total number of runs: {}".format(len(kwargs_iter)))



        # Create Pool and run post-processing:
        # TODO: replace with pool, but we need to pass a function, not a method
        pool = ThreadPool(processes=nb_thread_pools)
        starmap_with_kwargs(pool, self.run_method_on_graph, args_iter=repeat([]),
                            kwargs_iter=kwargs_iter)
        pool.close()
        pool.join()

    def run_method_on_graph(self, preset,
                            GT_labels,
                            p=None,
                            signed_edge_weights=None,
                            graph=None,
                            uv_ids=None,
                            A_p=None,
                            A_n=None,
                            eta=None,
                            gauss_sigma=None):
        post_proc_config, _ = self.apply_presets_to_postproc_config([preset])

        segm_pipeline_type = post_proc_config.get("segm_pipeline_type")

        # Run clustering:
        tick = time.time()
        print(preset)
        if segm_pipeline_type == "GASP":
            run_GASP_kwargs = post_proc_config.get("GASP_kwargs").get("run_GASP_kwargs")
            node_labels, _ = run_GASP(graph, signed_edge_weights,
                                      use_efficient_implementations=False,
                                      **run_GASP_kwargs)
        elif segm_pipeline_type == "spectral":
            c = signet_cluster.Cluster((A_p, A_n))
            spectral_method_name = post_proc_config.get("spectral_method_name")
            k = post_proc_config.get("SSBM_kwargs").get("k")
            try:
                if spectral_method_name == "BNC":
                    node_labels = c.spectral_cluster_bnc(k=k, normalisation='sym')
                elif spectral_method_name == "L-sym":
                    node_labels = c.spectral_cluster_laplacian(k=k, normalisation='sym')
                elif spectral_method_name == "SPONGE":
                    # FIXME: not sure about this...
                    # node_labels = c.geproblem_laplacian(k = k, normalisation='additive')
                    node_labels = c.SPONGE(k=k)
                elif spectral_method_name == "SPONGE-sym":
                    # node_labels = c.geproblem_laplacian(k = k, normalisation='multiplicative')
                    node_labels = c.SPONGE_sym(k=k)
                else:
                    raise NotImplementedError
            except np.linalg.LinAlgError:
                print("#### LinAlgError ({}) ####".format(spectral_method_name))
                return
        else:
            raise NotImplementedError

        runtime = time.time() - tick

        # Compute scores and stats:
        # Convert to "volume" array to compute cremi score:
        scores = segm_utils.cremi_score(np.expand_dims(np.expand_dims(GT_labels+1, axis=0), axis=0),
                               np.expand_dims(np.expand_dims(node_labels, axis=0), axis=0),
                               return_all_scores=True, run_connected_components=False)
        ARAND_score = adjusted_rand_score(node_labels, GT_labels)
        counts = np.bincount(node_labels.astype('int64'))
        nb_clusters = (counts > 0).sum()
        biggest_clusters = np.sort(counts)[::-1][:10]

        print(runtime, ARAND_score, scores)


        # ------------------------------
        # SAVING RESULTS:
        # ------------------------------
        strings_to_include_in_filenames = [segm_pipeline_type] + [preset]
        config_file_path, _ = \
            self.get_valid_out_paths(strings_to_include_in_filenames,
                                     overwrite_previous=post_proc_config.get("overwrite_prev_files", False),
                                     filename_postfix=post_proc_config.get('filename_postfix', None)
                                     )

        print(config_file_path)
        config_to_save = deepcopy(self._config)
        config_to_save.pop("postproc_presets")

        post_proc_config.pop("iterated_options")
        config_to_save['postproc_config'] = post_proc_config

        # Save configuration of the iterated kwargs:
        config_to_save["postproc_config"]["SSBM_kwargs"]["eta"] = eta
        config_to_save["postproc_config"]["SSBM_kwargs"]["p"] = p
        config_to_save["postproc_config"]["SSBM_kwargs"]["gauss_sigma"] = gauss_sigma

        # Include scores in config:
        config_to_save["runtime"] = runtime
        config_to_save["RAND_score"] = ARAND_score
        config_to_save['scores'] = scores
        config_to_save["nb_clusters"] = int(nb_clusters)
        config_to_save["biggest_clusters"] = [int(size) for size in biggest_clusters]

        with open(config_file_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, sort_keys=True)


    def get_kwargs_for_each_run(self):
        # TODO: add option to load GT and affs directly in run_agglomeration? (More memory efficient)
        nb_iterations = self.get("postproc_config/nb_iterations", 1)

        kwargs_collected = []

        postproc_config = self.get("postproc_config", ensure_exists=True)
        iterated_options = self.get("postproc_config/iterated_options", {})

        SSBM_kwargs = self.get("postproc_config/SSBM_kwargs", ensure_exists=True)

        # Initialize default iterated options:
        iterated_options.setdefault("preset", [postproc_config.get("preset", None)])
        iterated_options.setdefault("eta", [SSBM_kwargs.get("eta", 0.1)])
        iterated_options.setdefault("p", [SSBM_kwargs.get("p", 0.1)])
        iterated_options.setdefault("gaussian_sigma", [SSBM_kwargs.get("gaussian_sigma", 0.1)])

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
        for _ in range(nb_iterations):
            n = SSBM_kwargs.get("n")
            k = SSBM_kwargs.get("k")
            for p in iterated_options['p']:
                for eta in iterated_options['eta']:
                    for gauss_sigma in iterated_options['gaussian_sigma']:
                        print("Creating SSBM model...")


                        (A_p, A_n), GT_labels = block_models.SSBM(n=n, k=k, pin=p, etain=eta, values='gaussian',
                                                                 guassian_sigma=gauss_sigma)

                        # Symmetrize matrices:
                        # why was this necessary at all...?
                        grid = np.indices((n, n))
                        matrix_mask = grid[0] > grid[1]
                        A_p = matrix_mask * A_p.toarray()
                        A_n = matrix_mask * A_n.toarray()
                        A_p = A_p + np.transpose(A_p)
                        A_n = A_n + np.transpose(A_n)
                        A_p = sparse.csr_matrix(A_p)
                        A_n = sparse.csr_matrix(A_n)

                        A_signed = A_p - A_n
                        uv_ids, signed_edge_weights = from_adj_matrix_to_edge_list(A_signed)

                        print("Building nifty graph...")
                        graph = UndirectedGraph(n)
                        graph.insertEdges(uv_ids)
                        nb_edges = graph.numberOfEdges
                        assert nb_edges == uv_ids.shape[0]

                        # Test connected components:
                        from nifty.graph import components
                        components = components(graph)
                        components.build()
                        print("Nb. connected components in graph:", np.unique(components.componentLabels()).shape)

                        for preset in iterated_options['preset']:
                            new_kwargs = {
                                'preset': preset,
                                'GT_labels': GT_labels,
                                'A_p': A_p,
                                'A_n': A_n,
                                'eta': eta,
                                'gauss_sigma': gauss_sigma,
                                'p': p,
                                'signed_edge_weights': signed_edge_weights,
                                'graph': graph,
                                'uv_ids': uv_ids
                            }
                            kwargs_collected.append(new_kwargs)


        return kwargs_collected


    def get_valid_out_paths(self, strings_to_include_in_filenames,
                            sub_dirs=('scores', 'out_segms'),
                            file_extensions=('.yml', '.h5'),
                            filename_postfix=None,
                            overwrite_previous=False):
        """
        This function takes care of creating a valid names for the output config/score and segmentation files.

        By default only two paths are generated:
           - one for the score/config file, that will be stored in the 'scores' subfolder and with the `.yml` extension
           - one for the output segmentations, stored in the `out_segms` subfolder with the `.h5` extension.

        Sometimes, we want to run the same setup/agglomeration multiple times (when there is randomness or noise involved
        for example, so we collect statistics afterward). In this case, set overwrite_previous to False and each output
        file name will contain a randomly assigned ID between 0 and 1000000000.

        In order to make the names of the saved files more readable, the name will include all the strings included in
        the parameter `strings_to_include_in_filenames`
        """
        experiment_dir = self.get("exp_path")

        # Compose output file name:
        filename = ""
        for i, string in enumerate(strings_to_include_in_filenames):
            if i == 0:
                filename += "{}".format(string)
            else:
                filename += "__{}".format(string)
        if filename_postfix is not None:
            filename = filename + "__" + filename_postfix

        ID = str(np.random.randint(1000000000))
        out_file_paths = []
        for file_ext, dir_type in zip(file_extensions, sub_dirs):
            # Create directories:
            dir_path = os.path.join(experiment_dir, dir_type)
            segm_utils.check_dir_and_create(dir_path)

            candidate_file = os.path.join(dir_path, filename + file_ext)
            # If necessary, add random ID to filename:
            if not overwrite_previous:
                candidate_file = candidate_file.replace(file_ext, "__{}{}".format(ID, file_ext))
            out_file_paths.append(candidate_file)

        return out_file_paths

    def apply_presets_to_postproc_config(self,
                            list_of_presets_to_be_applied=None):
        # Get all presets:
        post_proc_config = deepcopy(self.get("postproc_config"))
        presets_collected = [] if list_of_presets_to_be_applied is None else list_of_presets_to_be_applied

        if post_proc_config.get("extra_presets", False):
            presets_from_config = post_proc_config.get("extra_presets")
            assert isinstance(presets_from_config, list)
            presets_collected += presets_from_config

        # Adapt the original config:
        adapted_config = adapt_configs_to_model_v2(presets_collected,
                                  config={"postproc": post_proc_config},
                                  all_presets=self.get("postproc_presets"),
                                  debug=True)
        return adapted_config["postproc"], presets_collected


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'postproc_configs')
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
    cls = SSBMPostProcessingExperiment
    cls().run()
