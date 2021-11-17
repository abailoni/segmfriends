import os

from ..utils.paths import get_vars_from_argv_and_pop


def process_speedrun_sys_argv(argv, source_path, default_config_rel_path='../../../configs',
                              default_exp_path="/scratch/bailoni/projects/cellpose_projects"):
    print("Experiment name: ", argv[1])
    collected_paths, argv = get_vars_from_argv_and_pop(argv,
                                                       config_path=os.path.join(source_path,
                                                                                default_config_rel_path),
                                                       exp_path=default_exp_path)
    config_path = collected_paths["config_path"]
    experiments_path = collected_paths["exp_path"]

    argv[1] = os.path.join(experiments_path, argv[1])
    if '--inherit' in argv:
        i = argv.index('--inherit') + 1
        if argv[i].endswith(('.yml', '.yaml')):
            argv[i] = os.path.join(config_path, argv[i])
        else:
            argv[i] = os.path.join(experiments_path, argv[i])
    if '--update' in argv:
        i = argv.index('--update') + 1
        argv[i] = os.path.join(config_path, argv[i])
    i = 0
    while True:
        if f'--update{i}' in argv:
            ind = argv.index(f'--update{i}') + 1
            argv[ind] = os.path.join(config_path, argv[ind])
            i += 1
        else:
            break

    return argv
