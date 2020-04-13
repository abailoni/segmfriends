import sys
from copy import deepcopy
import os
import torch

# Imports for models/criteria
import neurofire
import confnets
import inferno
import segmfriends


from inferno.io.transform import Compose
from neurofire.criteria.loss_wrapper import LossWrapper
from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin
from speedrun.log_anywhere import register_logger
from speedrun.py_utils import create_instance

from segmfriends.utils.config_utils import recursive_dict_update
from segmfriends.datasets.cremi import get_cremi_loader


# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class BaseCremiExperiment(BaseExperiment, InfernoMixin, TensorboardMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)


        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        # register_logger(FirelightLogger, "image")
        register_logger(self, 'scalars')

        self.model_class = self.get('model/model_class')

        self.set_devices()

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

    def set_devices(self):
        # # --------- In case of multiple GPUs: ------------
        # n_gpus = torch.cuda.device_count()
        # gpu_list = range(n_gpus)
        # self.set("gpu_list", gpu_list)
        # self.trainer.cuda(gpu_list)

        # --------- Debug on trendytukan, force to use only GPU 0: ------------
        self.set("gpu_list", [0])
        self.trainer.cuda([0])

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[i])
            i += 1
        else:
            break
    cls = BaseCremiExperiment
    cls().run()

