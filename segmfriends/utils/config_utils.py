from .various import yaml2dict

def recursive_dict_update(source, target):
    for key, value in source.items():
        if isinstance(value, dict):
            sub_target = target[key] if key in target else {}
            target[key] = recursive_dict_update(source[key], sub_target)
        else:
            target[key] = source[key]
    return target

def return_recursive_key_in_dict(dictionary, keys):
    assert isinstance(dictionary, dict)
    assert isinstance(keys, list)
    output = dictionary
    for key in keys:
        output = output[key]
    return output

def adapt_configs_to_model(model_IDs,
                           debug=False,
                            **configs):
    """
    :param model_ID: can be an int ID or the name of the model
    :param configs: list of strings with the paths to .yml files
    """
    for key in configs:
        assert key in ['models', 'train', 'valid', 'data', 'postproc', 'infer']


    def get_model_configs(model_IDs, model_configs=None):
        model_configs = {} if model_configs is None else model_configs
        model_IDs = [model_IDs] if not isinstance(model_IDs, list) else model_IDs

        for model_ID in model_IDs:# Look for the given model:
            # Look for the given model:
            model_name = None
            for name in configs['models']:
                if isinstance(model_ID, int):
                    if 'model_ID' in configs['models'][name]:
                        if configs['models'][name]['model_ID'] == model_ID:
                            model_name = name
                            break
                elif isinstance(model_ID, str):
                    if name == model_ID:
                        model_name = name
                        break
                else:
                    raise ValueError("Model ID should be a int. or a string")
            assert model_name is not None, "Model ID {} not found in the config file".format(model_ID)
            if debug:
                print("Using model ", model_name)


            new_model_configs = configs['models'][model_name]

            # Check parents models and update them recursively:
            if 'parent_model' in new_model_configs:
                model_configs = get_model_configs(new_model_configs['parent_model'], model_configs)

            # Update config with current options:
            model_configs = recursive_dict_update(new_model_configs, model_configs)

        return model_configs

    model_configs = get_model_configs(model_IDs)

    # Update paths init. segm and GT:
    if 'volume_config' in model_configs:
        samples = ['A', 'B', 'C']
        model_volume_config = model_configs['volume_config']

        def update_paths(target_vol_config, source_vol_config):
            # Loop over 'init_segmentation', 'GT', ...
            # If the path is not specified, then the one of 'init_segmentation' will be used
            for input_key in source_vol_config:
                target_vol_config[input_key] = {'dtype': 'int64', 'path': {},
                                                          'path_in_h5_dataset': {}} if input_key not in target_vol_config else target_vol_config[input_key]
                for smpl in samples:
                    path = source_vol_config[input_key]['path'] if 'path' in source_vol_config[input_key] else source_vol_config['init_segmentation']['path']
                    path = path.replace('$', smpl)
                    h5_path = source_vol_config[input_key]['path_in_h5_dataset'].replace('$', smpl)
                    target_vol_config[input_key]['path'][smpl] = path
                    target_vol_config[input_key]['path_in_h5_dataset'][smpl] = h5_path

            return target_vol_config

        for key in ['data', 'valid', 'infer', 'postproc']:
            if key in configs:
                configs[key]['volume_config'] = {} if 'volume_config' not in configs[key] else configs[key][
                    'volume_config']
                configs[key]['volume_config'] = update_paths(configs[key]['volume_config'], model_volume_config)

    # Update model-specific parameters:
    for key in configs:
        configs[key] = recursive_dict_update(model_configs.get(key, {}), configs[key])

    return configs
