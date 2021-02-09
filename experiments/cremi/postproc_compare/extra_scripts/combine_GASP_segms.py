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

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils import various as segm_utils
import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import collect_score_configs

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities, SegmentationFeeder


def combine_segm(project_directory, exp_name):
    scores_path = os.path.join(project_directory, exp_name, "scores")
    results_collected = collect_score_configs(
        scores_path,
        score_files_have_IDs=True,
        organize_configs_by=(
            ('postproc_config', 'sample'),
            ('postproc_config', 'crop'),)
    )



    for sample in results_collected:
        for crop in results_collected[sample]:
            all_configs = results_collected[sample][crop]


            # ----------------------------
            # Load data and affinities:
            # ----------------------------
            first_config = all_configs[list(all_configs.keys())[0]]
            GT_vol_config = deepcopy(first_config['volume_config']['GT'])
            affs_vol_config = deepcopy(first_config['affinities'])

            # FIXME:
            for smpl in GT_vol_config["path"]:
                GT_vol_config["path"][smpl] = GT_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/local_copy_home/", get_home_dir())
            for smpl in affs_vol_config["path"]:
                affs_vol_config["path"][smpl] = affs_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/local_copy_home/", get_home_dir())

            parsed_crop = segm_utils.parse_data_slice(crop)
            gt_crop_slc = parsed_crop[1:]

            GT = segm_utils.readHDF5_from_volume_config(sample,
                                                        crop_slice=gt_crop_slc,
                                                        run_connected_components=False,
                                                        **GT_vol_config)

            affinities = segm_utils.readHDF5_from_volume_config(sample,
                                                                crop_slice=parsed_crop,
                                                                run_connected_components=False,
                                                                **affs_vol_config)

            assert GT.shape == affinities.shape[1:], "Loaded GT and affinities do not match: {} - {}".format(GT.shape,
                                                                                                             affinities.shape[
                                                                                                             1:])
            # Apply sub-crop:
            sub_crop = first_config['postproc_config']['subcrop']
            sub_crop_slc = segm_utils.parse_data_slice(sub_crop)
            affinities = affinities[sub_crop_slc]
            GT = GT[sub_crop_slc[1:]]
            GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))

            # Load extra data:
            offsets = first_config['offsets']
            # FIXME:
            offsets_weights = [0. for _ in offsets]
            offsets_weights[0] = 1.
            offsets_weights[1] = 1.
            offsets_weights[2] = 1.


            # ----------------------------
            # Combine segmentations:
            # ----------------------------
            current_segm = np.ones_like(GT, dtype='uint64')
            for ID in all_configs:
                # Deduce path of the segmentation file:
                config = all_configs[ID]
                filename = config["score_filename"]
                scores_path = config["dir_scores"]
                segm_path = os.path.join(scores_path, "../out_segms", filename.replace(".yml", ".h5"))
                new_segm = segm_utils.readHDF5(segm_path, "segm", dtype='uint64')
                current_segm = segm_utils.cantor_pairing_fct(current_segm, new_segm)
                # Do I risk uint64 overflow...? Hopefully not
                current_segm = vigra.analysis.relabelConsecutive(current_segm)[0]

            # ----------------------------
            # Run final GASP avg agglo on combined segm:
            # ----------------------------
            run_GASP_kwargs = {'linkage_criteria': 'average',
                               'add_cannot_link_constraints': False}

            # Generalize options:
            gasp_instance = GaspFromAffinities(offsets,
                                               superpixel_generator=SegmentationFeeder(),
                                               n_threads=8,
                                               verbose=True,
                                               offsets_probabilities=0.,
                                               offsets_weights=offsets_weights,
                                               return_extra_outputs=True,
                                               run_GASP_kwargs=run_GASP_kwargs)
            import time
            tick = time.time()
            # pred_segm, out_dict = np.ones_like(GT), {'multicut_energy':np.array(0.), 'runtime':0.}
            pred_segm, out_dict = gasp_instance(affinities, current_segm)
            comp_time = time.time() - tick

            # ----------------------------
            # Postprocess, compute scores, and save:
            # ----------------------------
            post_proc_config = first_config['postproc_config']
            multicut_energy = out_dict['multicut_energy']
            # Relabel consecutive:
            pred_segm = vigra.analysis.relabelConsecutive(pred_segm.astype('uint64'))[0]

            # If necessary, get rid of small segments with seeded watershed:
            grow_WS = post_proc_config.get('thresh_segm_size', 0) != 0 and post_proc_config.get("WS_growing", False)
            if grow_WS:
                grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                                               offsets,
                                               hmap_kwargs=post_proc_config['prob_map_kwargs'],
                                               apply_WS_growing=True,
                                               size_of_2d_slices=False,
                                               invert_affinities=False)
                pred_segm_WS = grow(affinities, pred_segm)

            config_to_save = deepcopy(first_config)


            # TODO: add back support for GT box

            # Compute scores:
            config_to_save.update({'multicut_energy': multicut_energy.item(), 'run_GASP_runtime': out_dict['runtime'],
                                   'full_GASP_pipeline_runtime': comp_time})
            if post_proc_config.get("compute_scores", False):
                evals = segm_utils.cremi_score(GT, pred_segm, border_threshold=None, return_all_scores=True)
                if grow_WS:
                    evals_WS = segm_utils.cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
                    print("Scores achieved: \n {}".format(evals_WS))
                else:
                    evals_WS = None
                    print("Scores achieved: \n {}".format(evals))
                config_to_save.update(
                    {'score': evals, 'score_WS': evals_WS})

            # Deduce output paths:
            scores_path = first_config["dir_scores"]
            ID = str(np.random.randint(1000000000))
            out_segm_path = os.path.join(scores_path, "../out_segms", "combined_sample_{}_{}.h5".format(sample, ID))
            config_file_path = os.path.join(scores_path, "combined_sample_{}_{}.yml".format(sample, ID))

            # Dump config:
            with open(config_file_path, 'w') as f:
                # json.dump(config_to_save, f, indent=4, sort_keys=True)
                yaml.dump(config_to_save, f)

            # Save segmentation:
            if post_proc_config.get("save_segm", True):
                print(out_segm_path)
                if grow_WS:
                    segm_utils.writeHDF5(pred_segm_WS.astype('uint32'), out_segm_path, 'segm_WS', compression='gzip')
                segm_utils.writeHDF5(pred_segm.astype('uint32'), out_segm_path, 'segm', compression='gzip')

#








combine_segm(os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare"), "subcrop_train_samples_LR0")
