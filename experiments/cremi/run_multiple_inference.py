import os
from segmfriends.utils.config_utils import recursively_get_cmd

# -----------------------
# Define list of scripts to run:
# -----------------------

CUDA = "CUDA_VISIBLE_DEVICES=0"

list_of_args = [
    (["--"], ["deb_infer"]),
    (["--inherit"], [
        "example_train_affs_dowscaled_data.yml",
      ]),
    # Add three args at the same time:
    (["--update0", "--config.model.model_kwargs.loadfrom", "--config.export_path"
      ], [
        "infer_config.yml///test_infer",
    ],[
        "RUNS__HOME/deb_affs/checkpoint.pytorch", # Load your model from the RUNS folder in segmfriend
    ], [
        "RUNS__HOME/deb_affs",
    ]
     ),
    # Here loop over the following arg (two values):
    (["--update1"], [
        "empty_config.yml///_option1",
        "empty_config.yml///_option2",
    ]),
    # Here for example the name of the two experiments
    # will depend on the previously passed parameters and it will be 'test_infer_option1' and 'test_infer_option2':
    (["--config.name_experiment"], [("{}{}", "2:0", "3:0") ]),
    (["--config.loaders.infer.loader_config.batch_size"], ["1"]),
    (["--config.loaders.infer.loader_config.num_workers"], ["20"]),
    # (["--config.model.model_kwargs.path_backbone"], ["RUNS__HOME/deb/checkpoint.pytorch"]),
    (["--config.loaders.infer.name"], [
        "C",
        # "B",
        # "A",
    ]),
]

# -----------------------
# Compose list of commands to execute:
# -----------------------

cmd_base_string = CUDA + " ipython experiments/cremi/infer.py"

cmds_to_run = recursively_get_cmd(list_of_args, cmd_base_string)

print("Number to run: ", len(cmds_to_run))

for cmd in cmds_to_run:
    print("\n\n\n\n{}\n\n".format(cmd))
    os.system(cmd)
