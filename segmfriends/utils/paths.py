import os
from shutil import copyfile
from .various import check_dir_and_create


try:
    import pathutils
except ImportError:
    pathutils = None


def change_paths_config_file(template_path, path_keys, path_values):
    assert isinstance(path_values, (tuple, list))
    assert isinstance(path_keys, (tuple, list))
    template_dir, template_name = os.path.split(template_path)
    new_template_dir = os.path.join(template_dir, "temp")
    check_dir_and_create(new_template_dir)
    output_path = os.path.join(new_template_dir, template_name)
    # Make a temp copy of the template:
    copyfile(template_path, output_path)

    for key, path in zip(path_keys, path_values):
        path_placeholder = "${}\/".format(key)
        path = path.replace("/", "\/")
        cmd_string = "sed -i 's/{}/{}/g' {}".format(path_placeholder, path,
                                                      output_path)
        os.system(cmd_string)

    return output_path



def get_vars_from_argv(argv, vars=("DATA_HOME", "LOCAL_DRIVE")):
    def fix_path(path):
        return path if path.endswith('/') else path + '/'

    assert isinstance(vars, (list, tuple))

    collected_values = []
    for var in vars:
        var_argv = "--{}".format(var)
        if var_argv in argv:
            idx = argv.index(var_argv)
            value = argv.pop(idx + 1)
            argv.pop(idx)
        else:
            assert pathutils is not None, "{} was not correctly passed as a script parameter".format(var)
            if var == "DATA_HOME":
                value = pathutils.get_home_dir()
            elif var == "LOCAL_DRIVE":
                value = pathutils.get_trendytukan_drive_dir()
            else:
                raise ValueError("{} was not correctly passed as a script parameter".format(var))
        value = fix_path(value)
        collected_values.append(value)

    return collected_values, argv
