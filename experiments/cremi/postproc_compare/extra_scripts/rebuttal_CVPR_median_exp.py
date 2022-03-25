import os
from copy import deepcopy

# -----------------------
# Script options:
# -----------------------



type = "postproc"
CUDA = "CUDA_VISIBLE_DEVICES=0"

# WSDT superpixels plus GASP:

# list_of_args = [
#     (["--"], ["deb_infer"]),
#     (["--inherit"], [
#         "debug.yml",
#       ]),
#     # (["--config.experiment_name", "--config.offsets_file_name"],
#     #  ["mainFullTrain_cls", "bigUNet_cls", "main_classic", "clsDefct_cls", "noSideLoss_cls", "noGlia_cls", "main_dice", "2patches_cls"],
#     #  ["default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "dice_affs.json", "two_patches_only.json"],
#     #  ),
#     ([
#          "--config.experiment_name",
#          "--config.offsets_file_name",
#          "--config.postproc_config.invert_affinities"
#      ],
#      [
#          "v2_ignoreGlia_trainedAffs_thinBound",
#          # "v2_ignoreGlia_trainedAffs",
#          # "v2_main_trainedAffs_thinBound",
#          # "v2_main_trainedAffs",
#          # "v2_diceAffs_trainedAffs_thinBound",
#          # "v2_diceAffs_trainedAffs",
#      ],
#      [
#          "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "dice_affs.json",
#          # "dice_affs.json",
#      ],
#      [
#          "True",
#          # "True",
#          # "True",
#          # "True",
#          # "True",
#          # "True",
#      ],
#      ),
#
#     (["--config.postproc_config.save_name_postfix",
#       "--config.volume_config.ignore_glia"],
#      [
#          "fullGT",
#          # "ignoreGlia"
#      ],
#      [
#          "False",
#          # "True"
#      ]),
#     # (["--config.postproc_config.iterated_options.preset"], ["MEAN"]),
#     # (["--config.postproc_config.iterated_options.sample"], [
#     #     ["B", "C", "A"],
#     #     ["0", "1", "2"],
#     #     # "C"
#     # ]),
# ]

list_of_args = [
    (["--"], ["deb_infer"]),
    (["--inherit"], [
        "main_config.yml",
      ]),
    (["--update0"], ["subcrop_train_samples_SP_LR0_full.yml",]),
    # ([
    #   "--config.experiment_name",
    #     "--config.offsets_file_name",
    #     "--config.postproc_config.invert_affinities",
    #     "--update2",
    #  ],
    #  [
    #      # "v4_addSparseAffs_fullGT_eff",
    #      "v4_addSparseAffs_fullGT_avgDirectVar"
    #  ], [
    #     # "dice_affs_v3.json",
    #     "aggr_affs_v4.json",
    #  ], [
    #     # "True",
    #     "False"
    #  ], [
    #     # "empty_config.yml",
    #     "crop_avg_affs.yml",
    #  ]
    #  ),

    # (["--config.postproc_config.save_name_postfix",
    #   "--config.volume_config.ignore_glia"],
    #  [
    #      "combinedAffs_fullGT",
    #      # "ignoreGlia"
    #  ],
    #  [
    #      # "False",
    #      "True"
    #  ]),
    # (["--update1"], [
    #     # "empty_config.yml",
    #     # "longRange_DWST.yml",
    #     # "multicut.yml",
    #     # "multicut_longR.yml",
    #     "GASP_from_pix.yml"
    # ]),
    # (["--update2"], ["run_on_test.yml"]),
    # (["--config.offsets_file_name"], ["dice_affs_v3.json"]),
    # (["--config.volume_config.affinities.inner_path", "--config.postproc_config.save_name_postfix"],
    #  [
    #      "data",
    #      # "affs_plus_glia_2",
    #  ],
    #  [
    #      "affs",
    #      # "plusGliaMask2",
    #  ]
    #  ),
    # (["--config.postproc_config.nb_thread_pools"], ["1"]),
    # (["--config.volume_config.ignore_glia"], ["False"]),
    (["--config.postproc_config.iterated_options.sample"], [
        "A",
        "B",
        "C",
    ]),
]

"""
- MWS and DTWS+GASP (local)
- direct and averaged affs (not for dice..)
- partial sample C, 0, 1, 2 (all)
- ignore glia (all)
- offsets

"""


# -----------------------
# Compose list of commands to execute:
# -----------------------

cmd_base_string = CUDA
if type == "infer":
    cmd_base_string += " ipython experiments/cremi/infer_IoU.py"
elif type == "postproc":
    cmd_base_string += " ipython segmfriends/speedrun_exps/compare_postprocessing_from_affs.py"
else:
    raise ValueError


def recursively_get_cmd(current_cmd, accumulated_cmds):
    current_arg_spec_indx = len(current_cmd)
    if current_arg_spec_indx == len(list_of_args):

        # We are done, compose the command:
        new_cmd = cmd_base_string
        for i, arg_spec in enumerate(list_of_args):
            for nb_arg, arg_name in enumerate(arg_spec[0]):
                new_arg_str = current_cmd[i][nb_arg]
                if "///" in new_arg_str:
                    new_arg_str = new_arg_str.split("///")[0]

                new_cmd += " {} {}".format(arg_name, new_arg_str)
        accumulated_cmds.append(new_cmd)

    elif current_arg_spec_indx < len(list_of_args):
        # Here we add all options at current level and then recursively go deeper:
        current_arg_spec = list_of_args[current_arg_spec_indx]
        total_current_options = len(current_arg_spec[1])

        for nb_option in range(total_current_options):
            new_cmd_entry = []
            for arg in current_arg_spec[1:]:
                assert len(arg) == total_current_options, "All args  passed in the same entry should have the same number of options! {}, {}".format(arg, total_current_options)
                if isinstance(arg[nb_option], str):
                    # Here we simply append the string:
                    new_cmd_entry.append(arg[nb_option])
                else:
                    # Format the string from previously chosen options:
                    assert isinstance(arg[nb_option], tuple)
                    assert len(arg[nb_option]) >= 2

                    collected_format_args = []
                    for format_args in arg[nb_option][1:]:
                        indx1, indx2 = format_args.split(":")
                        assert int(indx1) < current_arg_spec_indx
                        collected_str = current_cmd[int(indx1)][int(indx2)]
                        if "///" in collected_str:
                            collected_str = collected_str.split("///")[1]
                        collected_format_args.append(collected_str)

                    # Compose new command entry:
                    new_cmd_entry.append(arg[nb_option][0].format(*collected_format_args))
            # Recursively go deeper:
            accumulated_cmds = recursively_get_cmd(current_cmd+[new_cmd_entry], accumulated_cmds)
    else:
        raise ValueError("Something went wrong")

    return accumulated_cmds

cmds_to_run = recursively_get_cmd([], [])
print("Number of commands to run: {}".format(len(cmds_to_run)))

for cmd in cmds_to_run:
    print("\n\n\n\n{}\n\n".format(cmd))
    os.system(cmd)
