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
from segmfriends.utils.cremi_utils import prepare_submission
from segmfriends.utils.various import check_dir_and_create, parse_data_slice
from segmfriends.utils import various as segm_utils
import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import collect_score_configs

from GASP.segmentation.GASP.run_from_affinities import GaspFromAffinities, SegmentationFeeder


def run_block_segm(project_directory, exp_name):
    scores_path = os.path.join(project_directory, exp_name, "scores")
    results_collected = collect_score_configs(
        scores_path,
        score_files_have_IDs=True,
        organize_configs_by=(
            ('postproc_config', 'presets_collected'),
            ('postproc_config', 'sample'),
            ('postproc_config', 'crop'),)
    )


    for agglo_type in results_collected:
        for sample in results_collected[agglo_type]:
            GT, affinities = None, None
            segmentations_collected = {}
            for crop in results_collected[agglo_type][sample]:
                all_configs = results_collected[agglo_type][sample][crop]


                # ----------------------------
                # Load data and affinities:
                # ----------------------------
                if GT is None:
                    first_config = all_configs[list(all_configs.keys())[0]]
                    GT_vol_config = deepcopy(first_config['volume_config']['GT'])
                    affs_vol_config = deepcopy(first_config['affinities'])

                    # FIXME:
                    for smpl in GT_vol_config["path"]:
                        GT_vol_config["path"][smpl] = GT_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/local_copy_home/", get_home_dir())
                        GT_vol_config["path"][smpl] = GT_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/trendyTukan_drive/", get_trendytukan_drive_dir())
                    for smpl in affs_vol_config["path"]:
                        affs_vol_config["path"][smpl] = affs_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/local_copy_home/", get_home_dir())
                        affs_vol_config["path"][smpl] = affs_vol_config["path"][smpl].replace("/home_sdb/abailoni_tmp/trendyTukan_drive/", get_trendytukan_drive_dir())


                    GT = segm_utils.readHDF5_from_volume_config(sample,
                                                                crop_slice=":,:,:",
                                                                run_connected_components=False,
                                                                **GT_vol_config)

                    affinities = segm_utils.readHDF5_from_volume_config(sample,
                                                                        crop_slice=":,:,:,:",
                                                                        run_connected_components=False,
                                                                        **affs_vol_config)

                    assert GT.shape == affinities.shape[1:], "Loaded GT and affinities do not match: {} - {}".format(GT.shape,
                                                                                                                     affinities.shape[
                                                                                                                     1:])

                    if affinities.dtype == 'uint8':
                        print("Converting to float32")
                        affinities = affinities.astype('float32') / 255.

                    # Load extra data:
                    offsets = first_config['offsets']
                    # FIXME:
                    # offsets_weights = [0. for _ in offsets]
                    # offsets_weights[0] = 1.
                    # offsets_weights[1] = 1.
                    # offsets_weights[2] = 1.


                # ----------------------------
                # Combine segmentations:
                # ----------------------------
                assert len(all_configs) == 1, "More than one ID found"
                for ID in all_configs:
                    # Deduce path of the segmentation file:
                    config = all_configs[ID]
                    filename = config["score_filename"]
                    scores_path = config["dir_scores"]
                    segm_path = os.path.join(scores_path, "../out_segms", filename.replace(".yml", ".h5"))
                    new_segm = segm_utils.readHDF5(segm_path, "segm", dtype='uint64')
                    # Do I risk uint64 overflow...? Hopefully not
                    segmentations_collected[crop] = vigra.analysis.relabelConsecutive(new_segm)[0]

            # ----------------------------
            # Run final GASP avg agglo on combined blockiwe segm:
            # ----------------------------
            # TODO: merge segmetations; prepare submission (but save both), use the correct criteria
            # Merge segmentations:
            combined_segmentation = np.zeros_like(GT)
            max = 0
            for i, crop in enumerate([":,:34",":,30:66", ":,62:98", ":,94:"]):
                actual_crop = parse_data_slice(crop)[1:]
                # Crop the redundant padding before the segm (but not for the first crop)
                sub_crop = slice(2, None, None) if i != 0 else slice(None)
                new_max = segmentations_collected[crop][sub_crop].max()
                combined_segmentation[actual_crop][sub_crop] = segmentations_collected[crop][sub_crop] + max
                max += new_max + 1

            print(first_config["volume_config"]["paths_padded_boxes"])
            run_GASP_kwargs = first_config['postproc_config']['GASP_kwargs']['run_GASP_kwargs']

            # Generalize options:
            gasp_instance = GaspFromAffinities(offsets,
                                               superpixel_generator=SegmentationFeeder(),
                                               n_threads=5,
                                               verbose=True,
                                               offsets_probabilities=0.,
                                               # offsets_weights=offsets_weights,
                                               return_extra_outputs=True,
                                               run_GASP_kwargs=run_GASP_kwargs)

            run_kwargs = {}
            restrict_to_GT_bbox = True
            run_kwargs["foreground_mask"] = GT != 0


            import time
            tick = time.time()
            # pred_segm, out_dict = np.ones_like(GT), {'multicut_energy':np.array(0.), 'runtime':0.}
            pred_segm, out_dict = gasp_instance(affinities, combined_segmentation, **run_kwargs)
            comp_time = time.time() - tick

            # ----------------------------
            # Postprocess, compute scores, and save:
            # ----------------------------
            post_proc_config = first_config['postproc_config']
            multicut_energy = out_dict['multicut_energy']
            # Relabel consecutive:
            pred_segm = vigra.analysis.relabelConsecutive(pred_segm.astype('uint64'))[0]

            # If necessary, get rid of small segments with seeded watershed:
            grow_WS = True
            if grow_WS:
                grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                                               offsets,
                                               hmap_kwargs=post_proc_config['prob_map_kwargs'],
                                               apply_WS_growing=True,
                                               with_background=restrict_to_GT_bbox,
                                               size_of_2d_slices=False,
                                               invert_affinities=False)
                pred_segm_WS = grow(affinities, pred_segm)

            config_to_save = deepcopy(first_config)



            # Compute scores:
            config_to_save.update({'multicut_energy': multicut_energy.item(), 'run_GASP_runtime': out_dict['runtime'],
                                   'full_GASP_pipeline_runtime': comp_time})

            # Deduce output paths:
            scores_path = first_config["dir_scores"]
            out_segm_dir = os.path.join(scores_path, "../out_segms/combined")
            check_dir_and_create(out_segm_dir)
            out_segm_path = os.path.join(out_segm_dir, "combined_sample_{}_{}.h5".format(sample, agglo_type))
            config_file_dir = os.path.join(scores_path, "combined")
            check_dir_and_create(config_file_dir)
            config_file_path = os.path.join(config_file_dir, "combined_sample_{}_{}.yml".format(sample, agglo_type))

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

                # if True:
                #     # TODO: generalize the ds factor
                #     path_bbox_slice = first_config["volume_config"]["paths_padded_boxes"][sample]
                #     path_bbox_slice = path_bbox_slice.replace(
                #         "/home_sdb/abailoni_tmp/local_copy_home/", get_home_dir())
                #     path_bbox_slice = path_bbox_slice.replace(
                #         "/home_sdb/abailoni_tmp/trendyTukan_drive/", get_trendytukan_drive_dir())
                #
                #     prepare_submission(sample, out_segm_path,
                #                        inner_path_segm="segm_WS" if grow_WS else "segm",
                #                        path_bbox_slice=path_bbox_slice)

#








run_block_segm(os.path.join(get_trendytukan_drive_dir(), "projects/new_agglo_compare"), "test_samples")
