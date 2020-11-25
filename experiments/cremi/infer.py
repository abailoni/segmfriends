from pathutils import get_home_dir, get_trendytukan_drive_dir, change_paths_config_file
import sys
import numpy as np


from speedrun import BaseExperiment, TensorboardMixin, AffinityInferenceMixin

from copy import deepcopy

import os
import torch

from segmfriends.utils.config_utils import recursive_dict_update
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils.various import writeHDF5

from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.io.transform import Compose


from speedrun.py_utils import create_instance
from segmfriends.datasets.cremi import get_cremi_loader

torch.backends.cudnn.benchmark = True

class BaseCremiExperiment(BaseExperiment, AffinityInferenceMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        self.set_devices()

        # self.build_infer_loader()
        if "model_class" in self.get('model'):
            self.model_class = self.get('model/model_class')
        else:
            self.model_class = list(self.get('model').keys())[0]

    def set_devices(self):
        # # --------- In case of multiple GPUs: ------------
        # n_gpus = torch.cuda.device_count()
        # gpu_list = range(n_gpus)
        # self.set("gpu_list", gpu_list)
        # self.trainer.cuda(gpu_list)

        # --------- Debug on trendytukan, force to use only GPU 0: ------------
        self.set("gpu_list", [0])
        self.trainer.cuda([0])

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config

        assert "model_class" in model_config
        assert "model_kwargs" in model_config
        model_class = model_config["model_class"]
        model_kwargs = model_config["model_kwargs"]
        model_path = model_kwargs.pop('loadfrom', None)
        model_config = {model_class: model_kwargs}
        model = create_instance(model_config, self.MODEL_LOCATIONS)

        if model_path is not None:
            print(f"loading model from {model_path}")
            loaded_model = torch.load(model_path)["_model"]
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict)

        return model

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_kwargs = self.get("trainer/criterion/kwargs", {})
        # from vaeAffs.models.losses import EncodingLoss, PatchLoss, PatchBasedLoss, StackedAffinityLoss
        loss_name = self.get("trainer/criterion/loss_name",
                             "inferno.extensions.criteria.set_similarity_measures.SorensenDiceLoss")
        loss_config = {loss_name: loss_kwargs}

        criterion = create_instance(loss_config, self.CRITERION_LOCATIONS)
        transforms = self.get("trainer/criterion/transforms")
        if transforms is not None:
            assert isinstance(transforms, list)
            transforms_instances = []
            # Build transforms:
            for transf in transforms:
                transforms_instances.append(create_instance(transf, []))
            # Wrap criterion:
            criterion = LossWrapper(criterion, transforms=Compose(*transforms_instances))

        self._trainer.build_criterion(criterion)
        self._trainer.build_validation_criterion(criterion)

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)


    def build_infer_loader(self):
        kwargs = deepcopy(self.get('loaders/infer'))
        loader = get_cremi_loader(kwargs)
        return loader

    def save_infer_output(self, output):
        print("Saving....")
        if self.get("export_path") is not None:
            dir_path = os.path.join(self.get("export_path"),
                                    self.get("name_experiment", default="generic_experiment"))
        else:
            try:
                # Only works for my experiments saving on trendyTukan, otherwise it will throw an error:
                trendyTukan_path = get_trendytukan_drive_dir()
            except ValueError:
                raise ValueError("TrendyTukan drive not found. Please specify an `export_path` in the config file.")
            dir_path = os.path.join(trendyTukan_path, "projects/pixel_embeddings", self.get("name_experiment", default="generic_experiment"))
        check_dir_and_create(dir_path)
        filename = os.path.join(dir_path, "predictions_sample_{}.h5".format(self.get("loaders/infer/name")))
        print("Writing to ", self.get("inner_path_output", 'data'))
        writeHDF5(output.astype(np.float16), filename, self.get("inner_path_output", 'data'))
        print("Saved to ", filename)

        # Dump configuration to export folder:
        self.dump_configuration(os.path.join(dir_path, "prediction_config.yml"))

if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    # Update HCI_HOME paths:
    for i, key in enumerate(sys.argv):
        if "HCI__HOME" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("HCI__HOME/", get_home_dir())

    # Update RUNS paths:
    for i, key in enumerate(sys.argv):
        if "RUNS__HOME" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("RUNS__HOME", experiments_path)

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = change_paths_config_file(os.path.join(config_path, sys.argv[ind]))
            i += 1
        else:
            break
    cls = BaseCremiExperiment
    cls().infer()
