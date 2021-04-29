from segmfriends.utils import writeHDF5

from GASP.utils.various import find_indices_direct_neighbors_in_offsets
from segmfriends.utils.paths import get_vars_from_argv, change_paths_config_file
from segmfriends.utils.opensimplex_noise import add_opensimplex_noise_to_affs
from segmfriends.features import map_features_to_label_array

from speedrun import BaseExperiment

from copy import deepcopy

import os
import torch
import yaml

import sys

import json
import numpy as np
import vigra
import segmfriends.utils.various as segm_utils
from segmfriends.utils.config_utils import adapt_configs_to_model_v2
from segmfriends.algorithms import get_segmentation_pipeline

from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
import time
import shutil

from multiprocessing.pool import ThreadPool, Pool
from itertools import repeat
from segmfriends.utils.various import starmap_with_kwargs

torch.backends.cudnn.benchmark = True


class PostProcessingExperiment(BaseExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(PostProcessingExperiment, self).__init__(experiment_directory)
        # Privates
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.auto_setup()

        # Where the segmentation will be saved: (each experiment is in a sub-folder)
        main_output_dir = self.get('main_output_dir', ensure_exists=True)
        self.set("main_output_dir", main_output_dir)

        # Load the file with all the presets of postprocessing:
        postproc_presets_path = self.get('postproc_config/postproc_presets_file_path', ensure_exists=True)
        self.set("postproc_presets", segm_utils.yaml2dict(postproc_presets_path))

        self.build_experiment()
        self.build_offsets()

    def build_experiment(self):
        # Create dirs if not existing:
        exp_name = self.get("experiment_name", ensure_exists=True)
        exp_path = os.path.join(self.get("main_output_dir", ensure_exists=True), exp_name)
        self.set("exp_path", exp_path)
        exp_path_exists = segm_utils.check_dir_and_create(exp_path)

        # Check paths where to get affinities, if a path is not already specified:
        if self.get("affinities/path") is None:
            # Check if an experiment folder is given:
            if self.get("get_affinities_from_experiment_named") is not None:
                affs_dir_path = os.path.join(self.get("main_output_dir"), self.get("get_affinities_from_experiment_named"))
            else:
                assert exp_path_exists, "None affs path has been passed and none were found in exp folder!"
                affs_dir_path = exp_path
            self.set("affinities_dir_path", affs_dir_path)

    def build_offsets(self):
        if self.get("offsets_file_name") is not None:
            print("Loading offsets from file...")
            offset_dir = self.get("offsets_dir_path", ensure_exists=True)
            offsets_path = os.path.join(offset_dir, self.get("offsets_file_name", ensure_exists=True))
            assert os.path.exists(offsets_path)
            with open(offsets_path, 'r') as f:
                self.set("offsets", json.load(f))
        else:
            print("Loading offsets from inference config...")
            offset_path_in_infer_config = self.get("offset_path_in_infer_config", ensure_exists=True)
            affs_dir_path = self.get("affinities_dir_path", ensure_exists=True)
            prediction_config = segm_utils.yaml2dict(os.path.join(affs_dir_path, "prediction_config.yml"))

            # Recursively look into the prediction_config:
            paths = offset_path_in_infer_config.split("/")
            data = prediction_config
            for path in paths:
                assert path in data
                data = data.get(path)
            offsets = data
            assert isinstance(offsets, list)
            self.set("offsets", offsets)

    def run(self):
        # Load data for all the runs:
        nb_thread_pools = self.get("postproc_config/nb_thread_pools")
        print("Pools: ", nb_thread_pools)
        kwargs_iter = self.get_kwargs_for_each_run()
        print("Total number of runs: {}".format(len(kwargs_iter)))



        # Create Pool and run post-processing:
        # TODO: replace with pool, but we need to pass a function, not a method
        pool = ThreadPool(processes=nb_thread_pools)
        starmap_with_kwargs(pool, self.run_clustering, args_iter=repeat([]),
                            kwargs_iter=kwargs_iter)
        pool.close()
        pool.join()

    def run_clustering(self, affinities, GT, sample, crop_slice, sub_crop_slice, preset,
                       edge_prob,
                       local_attraction,
                       noise_factor,
                       mask_used_edges):
        affinities = affinities.copy()
        sample = str(sample)

        # -----------------------------------
        # Temporary fix to crop affinities if the tensor happens to have padding with invalid values (usually with
        # value -1), for example because a portion of the dataset was not predicted by the model.
        def find_first_index(array, min, max):
            for idx, val in np.ndenumerate(array):
                if val >= min and val <= max:
                    return idx
            return None

        global_pad = find_first_index(affinities, 0., 1.)

        global_crop_slc = tuple(slice(pad, -pad) if pad!= 0 else slice(None) for pad in global_pad)

        affinities = affinities[global_crop_slc]
        if mask_used_edges is not None:
            mask_used_edges = mask_used_edges[global_crop_slc]
        GT = GT[global_crop_slc[1:]]
        # -----------------------------------

        # ------------------------------
        # Build segmentation pipeline:
        # ------------------------------
        print(sample, preset, crop_slice) #, sub_crop_slice)

        offsets = self.get("offsets")

        # TODO: generalize so that we can pass a list of list of presets, and iterate all the combinations!
        post_proc_config, presets_collected = self.apply_presets_to_postproc_config([preset])

        assert not post_proc_config.get("start_from_given_segmentation", False), "Starting from a given segmentation is not suppoerted with the current setup"

        # Make some adjustments to config:
        post_proc_config['GASP_kwargs']['return_extra_outputs'] = True
        if mask_used_edges is None:
            post_proc_config['GASP_kwargs']['offsets_probabilities'] = edge_prob
            post_proc_config['multicut_kwargs']['offsets_probabilities'] = edge_prob
        else:
            post_proc_config['GASP_kwargs'].pop('offsets_probabilities', None)
            post_proc_config['multicut_kwargs'].pop('offsets_probabilities', None)

        nb_threads = post_proc_config.get('nb_threads')
        invert_affinities = post_proc_config.get('invert_affinities', False)
        assert 'segm_pipeline_type' in post_proc_config, "Segmentation pipeline was not specified"
        segm_pipeline_type = post_proc_config['segm_pipeline_type']

        # Build the segmentation pipeline:
        segmentation_pipeline = get_segmentation_pipeline(
            offsets=offsets,
            return_fragments=False,
            **post_proc_config
        )

        # ------------------------------
        # Run segmentation pipeline:
        # ------------------------------
        run_kwargs = {}
        # Pass mask of used edges if necessary:
        if mask_used_edges is not None:
            assert segm_pipeline_type == "GASP", "Only GASP supports mask of edges at the moment!"
            run_kwargs = {"mask_used_edges": mask_used_edges}

        run_args = []
        if post_proc_config.get("restrict_to_GT_bbox", False):
            assert post_proc_config.get("from_superpixels", False), "Restricting to GT box is only supported from superpixels at the moment."
            run_args.append(GT != 0)

        print("Starting prediction...")
        tick = time.time()
        outputs = segmentation_pipeline(affinities, *run_args, **run_kwargs)
        comp_time = time.time() - tick
        print("Post-processing took {} s".format(comp_time))

        # Analyze outputs:
        if isinstance(outputs, tuple):
            pred_segm, out_dict = outputs
        else:
            pred_segm = outputs
            out_dict = {'multicut_energy': np.array([0]), 'runtime': 0.}
        multicut_energy = out_dict['multicut_energy']

        # ------------------------------
        # Post-process the output segmentation:
        # ------------------------------
        # Convert to 2D segmentation, if necessary:
        if post_proc_config.get("return_2D_segmentation", False):
            segm_2D = np.empty_like(pred_segm)
            max_label = 0
            for z in range(pred_segm.shape[0]):
                segm_2D[z] = pred_segm[z] + max_label
                max_label += pred_segm[z].max() + 1
            pred_segm = vigra.analysis.labelVolume(segm_2D.astype('uint32'))

        # Relabel consecutive:
        pred_segm = vigra.analysis.relabelConsecutive(pred_segm.astype('uint64'))[0]

        # Some algorithms could return clusters with multiple connected compoents in the image plane
        # (e.g. MWS (efficient implementation) with long-range attractive connections), so we may want
        # to run connected components:
        if post_proc_config.get("connected_components_on_final_segm", False):
            # Vigra cannot handle numbers higher than uint32:
            if pred_segm.max() > np.uint32(-1):
                raise ValueError("uint32 limit reached!")
            else:
                pred_segm = vigra.analysis.labelVolumeWithBackground(pred_segm.astype('uint32'))

        # If necessary, get rid of small segments with seeded watershed:
        grow_WS = post_proc_config.get('thresh_segm_size', 0) != 0 and post_proc_config.get("WS_growing", False)
        if grow_WS:
            grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                                           offsets,
                                           hmap_kwargs=post_proc_config['prob_map_kwargs'],
                                           apply_WS_growing=True,
                                           size_of_2d_slices=False,
                                           invert_affinities=invert_affinities)
            pred_segm_WS = grow(affinities.astype('float32'), pred_segm)



        # TODO: is local and global crop really necessary...? I should handle different samples properly......
        #  That was the main deal of the subcrop I think (I could apply it independtly to each CREMI sample)
        # TODO: separate presets files
        # TODO: more iterable presets
        # TODO: save sp segmentation?

        # ------------------------------
        # SAVING RESULTS:
        # ------------------------------
        strings_to_include_in_filenames = [sample, segm_pipeline_type, str(edge_prob)] + presets_collected
        config_file_path, segm_file_path = \
            self.get_valid_out_paths(strings_to_include_in_filenames,
                                     overwrite_previous=post_proc_config.get("overwrite_prev_files", False),
                                     filename_postfix=post_proc_config.get('filename_postfix', None)
                                     )

        print(segm_file_path)
        config_to_save = deepcopy(self._config)
        config_to_save.pop("postproc_presets")

        post_proc_config.pop("iterated_options")
        config_to_save['postproc_config'] = post_proc_config

        # Save configuration of the iterated kwargs:
        config_to_save["postproc_config"]["presets_collected"] = presets_collected
        config_to_save["postproc_config"]["sample"] = sample
        config_to_save["postproc_config"]["crop"] = crop_slice
        config_to_save["postproc_config"]["subcrop"] = sub_crop_slice
        config_to_save["postproc_config"]["local_attraction"] = local_attraction
        config_to_save["postproc_config"]["edge_prob"] = edge_prob
        config_to_save["postproc_config"]['noise_factor'] = noise_factor

        # Restrict to GT box:
        if grow_WS:
            pred_segm_WS += 1
            pred_segm_WS[GT == 0] = 0
            pred_segm_WS = vigra.analysis.labelVolumeWithBackground(pred_segm_WS.astype('uint32'))
        pred_segm += 1
        pred_segm[GT == 0] = 0
        pred_segm = vigra.analysis.labelVolumeWithBackground(pred_segm.astype('uint32'))

        # Compute scores:
        config_to_save.update({'multicut_energy': multicut_energy.item(), 'run_GASP_runtime': out_dict['runtime'],
                               'full_GASP_pipeline_runtime': comp_time})
        if post_proc_config.get("compute_scores", False):
            evals = segm_utils.cremi_score(GT, pred_segm, border_threshold=None, return_all_scores=True)
            print("Scores achieved ({}): \n {}".format(presets_collected, evals))
            if grow_WS:
                evals_WS = segm_utils.cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
                print("Scores achieved WS ({}): \n {}".format(presets_collected, evals_WS))
            else:
                evals_WS = None
            config_to_save.update(
                {'score': evals, 'score_WS': evals_WS})

        # Dump config:
        with open(config_file_path, 'w') as f:
            # json.dump(config_to_save, f, indent=4, sort_keys=True)
            yaml.dump(config_to_save, f)

        # Save segmentation:
        if post_proc_config.get("save_segm", True):
            print(segm_file_path)
            if grow_WS:
                pred_segm_WS = np.pad(pred_segm_WS, pad_width=[(pad, pad) for pad in global_pad[1:]], mode="constant")
                writeHDF5(pred_segm_WS.astype('uint32'), segm_file_path, 'segm_WS', compression='gzip')
            pred_segm = np.pad(pred_segm, pad_width=[(pad, pad) for pad in global_pad[1:]], mode="constant")
            writeHDF5(pred_segm.astype('uint32'), segm_file_path, 'segm', compression='gzip')

            if post_proc_config.get("save_submission_tiff", False):
                # Compute submission tiff file (boundary: 0, inner: 1)
                from skimage.segmentation import find_boundaries
                from skimage import io
                binary_boundaries = np.empty_like(pred_segm_WS, dtype="bool")
                for z in range(pred_segm_WS.shape[0]):
                    binary_boundaries[z] = find_boundaries(pred_segm_WS[z], connectivity=1, mode='thick', background=0)
                binary_boundaries = np.logical_not(binary_boundaries)
                io.imsave(segm_file_path.replace(".h5", ".tif"), binary_boundaries.astype('float32'))

            if post_proc_config.get("prepare_submission", False):
                raise DeprecationWarning("Get it back from old repo")
                from vaeAffs.postproc.utils import prepare_submission
                # FIXME: generalize the ds factor and path to bbox
                path_bbox_slice = self.get("volume_config/paths_padded_boxes", ensure_exists=True)
                prepare_submission(sample, segm_file_path,
                                   inner_path_segm="segm_WS" if grow_WS else "segm",
                                   path_bbox_slice=path_bbox_slice[sample],
                                   ds_factor=(1,2,2))

        if post_proc_config.get("save_agglomeration_data", False):
            #node_stats, edge_data, action_data = out_dict["agglomeration_data"]
            #_, constrain_stats, merge_stats = edge_data
            #is_local_edge = out_dict["is_local_edge"]
            #edge_sizes = out_dict["edge_sizes"]
            #merge_stats_mapped = map_features_to_label_array(edge_ids, merge_stats)
            #constraint_stats_mapped = map_features_to_label_array(edge_ids, constrain_stats)
            #node_stats_mapped = node_stats.reshape(pred_segm.shape + (node_stats.shape[1],))

            graph = out_dict["graph"]
            edge_ids = np.rollaxis(graph.projectEdgesIDToPixels(), axis=3, start=0)
            edge_weights = out_dict['edge_weights']
            edge_weights_mapped = np.moveaxis(
                map_features_to_label_array(edge_ids, np.expand_dims(edge_weights, -1)).squeeze(), 3, 0)

            edge_labels1D = graph.nodeLabelsToEdgeLabels(pred_segm.reshape(-1))
            edge_labels = np.moveaxis(
                map_features_to_label_array(edge_ids, np.expand_dims(edge_labels1D, -1)).squeeze(), 3, 0)

            edge_labels1D_WS = graph.nodeLabelsToEdgeLabels(pred_segm_WS.reshape(-1))
            edge_labels_WS = np.moveaxis(
                map_features_to_label_array(edge_ids, np.expand_dims(edge_labels1D_WS, -1)).squeeze(), 3, 0)


            # To be saved:
            # - edge_ids, merge and constr. data (mapped)
            # - affinities and GT
            # - (final segm)
            exported_data_path = segm_file_path.replace(".h5", "_exported_data.h5")
            writeHDF5(GT, exported_data_path, 'GT', compression='gzip')
            #writeHDF5(affinities, exported_data_path, 'affinities', compression='gzip')
            #writeHDF5(edge_ids, exported_data_path, 'edge_ids', compression='gzip')
            writeHDF5(edge_weights_mapped, exported_data_path, 'edge_weights', compression='gzip')
            writeHDF5(pred_segm.astype('uint32'), exported_data_path, 'segm', compression='gzip')
            writeHDF5(pred_segm_WS.astype('uint32'), exported_data_path, 'segm_WS', compression='gzip')
            writeHDF5(edge_labels, exported_data_path, 'edge_labels', compression='gzip')
            writeHDF5(edge_labels_WS, exported_data_path, 'edge_labels_WS', compression='gzip')
            #writeHDF5(crop_slice, exported_data_path, 'crop', compression='gzip')





            #writeHDF5(merge_stats_mapped, exported_data_path, 'merge_stats', compression='gzip')
            #writeHDF5(constraint_stats_mapped, exported_data_path, 'constraint_stats', compression='gzip')
            #writeHDF5(node_stats_mapped, exported_data_path, 'node_stats', compression='gzip')


        # from segmfriends.transform.combine_segms_CY import find_segmentation_mistakes
        # result = find_segmentation_mistakes(segm_to_analyze, gt_to_analyze, ARAND_thresh=0.4, ignore_label=0,
        #                                     mode="undersegmentation")


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

    def get_kwargs_for_each_run(self):
        # TODO: add option to load GT and affs directly in run_agglomeration? (More memory efficient)
        nb_iterations = self.get("postproc_config/nb_iterations", 1)

        kwargs_collected = []

        postproc_config = self.get("postproc_config", ensure_exists=True)
        iterated_options = self.get("postproc_config/iterated_options", {})

        # Assert to have a sample:
        if "sample" not in iterated_options:
            assert "sample" in postproc_config, "At least one sample-dataset should be given"
            iterated_options["sample"] = postproc_config["sample"]

        # Initialize default iterated options:
        iterated_options.setdefault("noise_factor", [postproc_config.get("noise_factor", 0.)])
        iterated_options.setdefault("edge_prob", [postproc_config.get("edge_prob", 0.)])
        iterated_options.setdefault("preset", [postproc_config.get("preset", None)])
        iterated_options.setdefault("local_attraction", [postproc_config.get("local_attraction", False)])
        iterated_options.setdefault("crop_slice", [postproc_config.get("crop_slice", ":,:,:,:")])
        iterated_options.setdefault("sub_crop_slice", postproc_config.get("sub_crop_slice", [":,:,:,:"]))

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


        for _ in range(nb_iterations):
            collected_data = {"affs": {},
                              "GT": {},
                              "affs_mask": {}}

            for sample in iterated_options['sample']:
                print("Loading...")
                # Create new dict entry if needed:
                for dt_type in collected_data:
                    collected_data[dt_type][sample] = {} if sample not in collected_data[dt_type] else \
                        collected_data[dt_type][sample]

                # Check if we have a dictionary with single values for each sample:
                all_crops = iterated_options['crop_slice'][sample] if isinstance(iterated_options['crop_slice'], dict) \
                    else iterated_options['crop_slice']
                all_subcrops = iterated_options['sub_crop_slice'][sample] if isinstance(iterated_options['sub_crop_slice'], dict) \
                    else iterated_options['sub_crop_slice']

                # ----------------------------------------------------------------------
                # Load data (and possibly add noise or select long range edges):
                # ----------------------------------------------------------------------
                for crop in all_crops:
                    # Create new dict entry if needed:
                    for dt_type in collected_data:
                        collected_data[dt_type][sample][crop] = {} if crop not in collected_data[dt_type][sample] else \
                            collected_data[dt_type][sample][crop]

                    for sub_crop in all_subcrops:
                        # Create new dict entry if needed:
                        for dt_type in collected_data:
                            collected_data[dt_type][sample][crop][sub_crop] = {} \
                                if sub_crop not in collected_data[dt_type][sample][crop] else \
                                collected_data[dt_type][sample][crop][sub_crop]

                        noise_seed = np.random.randint(-100000, 100000)

                        GT_vol_config = deepcopy(self.get('volume_config/GT'))
                        affs_vol_config = deepcopy(self.get('affinities'))

                        # FIXME: if I pass multiple crops, they get ignored and I get an error below when I create the runs...
                        if "crop_slice" in GT_vol_config:
                            gt_crop_slc = GT_vol_config.pop("crop_slice")
                        else:
                            gt_crop_slc = segm_utils.parse_data_slice(crop)[1:]

                        GT = segm_utils.readHDF5_from_volume_config(sample,
                                            crop_slice=gt_crop_slc,
                                            run_connected_components=False,
                                            **GT_vol_config)


                        # Optionally, affinity paths are deduced dynamically:
                        if self.get("affinities_dir_path") is not None:
                            affs_vol_config['path'] = \
                                os.path.join(self.get("affinities_dir_path"), "predictions_sample_{}.h5".format(sample))

                        if "crop_slice" in affs_vol_config:
                            affs_crop_slc = affs_vol_config.pop("crop_slice")
                        else:
                            affs_crop_slc = segm_utils.parse_data_slice(crop)

                        affinities = segm_utils.readHDF5_from_volume_config(sample,
                                            crop_slice=affs_crop_slc,
                                            run_connected_components=False,
                                            **affs_vol_config)

                        assert GT.shape == affinities.shape[1:], "Loaded GT and affinities do not match: {} - {}".format(GT.shape, affinities.shape[1:])
                        sub_crop_slc = segm_utils.parse_data_slice(sub_crop)
                        affinities = affinities[sub_crop_slc]
                        GT = GT[sub_crop_slc[1:]]
                        GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))

                        # Add some more white noise to the affinities, which is usually beneficial
                        # to break ties with sigmoid outputs
                        affinities = affinities.astype("float64")
                        while True:
                            print("Adding some white noise to affinities...")
                            affinities += np.random.normal(scale=1e-3, size=affinities.shape)
                            # Map back to 0 and 1 interval:
                            affinities -= np.minimum(affinities.min(), 0.)
                            affinities /= np.maximum(affinities.max(), 1.)
                            # Debug: make sure not to have double values
                            counts = np.unique(affinities, return_counts=True)[1]
                            print("Max count: ", counts.max())
                            # if counts.max() == 1:
                            break

                        # affinities = np.clip(affinities, 0., 1.)

                        collected_data["GT"][sample][crop][sub_crop] = GT
                        collected_data["affs"][sample][crop][sub_crop] = {}
                        collected_data["affs_mask"][sample][crop][sub_crop] = {}
                        for long_range_prob in iterated_options['edge_prob']:
                            for noise in iterated_options['noise_factor']:
                                if noise != 0.:
                                    noise_mod = postproc_config["noise_mod"]
                                    collected_data["affs"][sample][crop][sub_crop][noise] = \
                                        add_opensimplex_noise_to_affs(
                                        affinities, noise,
                                        mod=noise_mod,
                                        target_affs='all',
                                        seed=noise_seed
                                        )
                                else:
                                    collected_data["affs"][sample][crop][sub_crop][noise] = affinities

                                # Fix already long-range edges that will be in the graph:
                                if long_range_prob < 1.0 and long_range_prob > 0.0:
                                    collected_data["affs_mask"][sample][crop][sub_crop][long_range_prob] = np.random.random(
                                        affinities.shape) < long_range_prob
                                    # Direct neighbors should be always added:
                                    offsets = self.get("offsets")
                                    is_offset_direct_neigh, _ = find_indices_direct_neighbors_in_offsets(offsets)
                                    collected_data["affs_mask"][sample][crop][sub_crop][long_range_prob][is_offset_direct_neigh] = True

                # ----------------------------------------------------------------------
                # Create iterators:
                # ----------------------------------------------------------------------
                print("Creating pool instances...")
                for crop in all_crops:
                    for sub_crop in all_subcrops:
                        assert collected_data["affs"][sample][crop][sub_crop] is not None
                        for local_attr in iterated_options['local_attraction']:
                            for preset in iterated_options['preset']:
                                if local_attr and preset in ['greedyFixation', 'GAEC']:
                                    continue
                                for edge_prob in iterated_options['edge_prob']:
                                    for noise in iterated_options['noise_factor']:
                                        # Build a new run dictionary with the iterated options:
                                        kwargs_collected.append({
                                            'affinities': collected_data['affs'][sample][crop][sub_crop][noise],
                                            'GT': collected_data['GT'][sample][crop][sub_crop],
                                            'sample': sample,
                                            'noise_factor': noise,
                                            'edge_prob': edge_prob,
                                            'preset': preset,
                                            'local_attraction': local_attr,
                                            'crop_slice': crop,
                                            'sub_crop_slice': sub_crop,
                                            'mask_used_edges': collected_data['affs_mask'][sample][crop][sub_crop].get(
                                                edge_prob, None)
                                        })

        return kwargs_collected

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
    cls = PostProcessingExperiment
    cls().run()
