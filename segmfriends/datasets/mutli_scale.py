from copy import deepcopy
import numpy as np

try:
    from inferno.io.core import Zip, Concatenate
    from inferno.io.transform import Compose, Transform
    from inferno.io.transform.generic import AsTorchBatch
    from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
    from inferno.io.transform.image import RandomRotate, ElasticTransform
except ImportError:
    raise ImportError("CremiDataset requires inferno")

try:
    from neurofire.datasets.loader import RawVolume, SegmentationVolume
    from neurofire.transform.artifact_source import RejectNonZeroThreshold
    from neurofire.transform.volume import RandomSlide
except ImportError:
    raise ImportError("CremiDataset requires neurofire")

from ..utils.various import yaml2dict
from ..transform.volume import DownSampleAndCropTensorsInBatch, ReplicateTensorsInBatch
from ..transform.affinities import affinity_config_to_transform, Segmentation2AffinitiesDynamicOffsets


class MultiScaleDataset(Zip):
    def __init__(self, name, volume_config, slicing_config,
                 transform_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'volume_keys_to_load' in volume_config

        volume_config = deepcopy(volume_config)

        volumes_to_load = []
        volume_keys_to_load = volume_config.pop("volume_keys_to_load")
        for volume_key in volume_keys_to_load:
            if volume_key not in volume_config:
                print("Warning: no kwargs passed for volume {}".format(volume_key))
            # Get default slicing configs:
            current_volume_kwargs = deepcopy(slicing_config)
            current_volume_kwargs.update(volume_config.get(volume_key, {}))
            dtype = current_volume_kwargs.get("dtype")
            dtype = dtype[name] if isinstance(dtype, dict) else dtype
            if dtype == "float32":
                volumes_to_load.append(RawVolume(name=name,
                                                 **current_volume_kwargs))
            elif dtype == "int32":
                volumes_to_load.append(SegmentationVolume(name=name,
                                                          **current_volume_kwargs))
            else:
                raise ValueError("Atm only float32 and int32 datasets are supported. {} was given".format(dtype ))

        super().__init__(*volumes_to_load,
                         sync=True)

        # Set master config (for transforms)
        self.transform_config = {} if transform_config is None else deepcopy(transform_config)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        if self.transform_config.get('random_flip', False):
            transforms.add(RandomFlip3D())
            transforms.add(RandomRotate())

        # Elastic transforms can be skipped by
        # setting elastic_transform to false in the
        # yaml config file.
        if self.transform_config.get('elastic_transform'):
            elastic_transform_config = self.transform_config.get('elastic_transform')
            if elastic_transform_config.get('apply', False):
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        # Replicate and downscale batch:
        nb_inputs = 1
        if self.transform_config.get("downscale_and_crop") is not None:
            ds_config = self.transform_config.get("downscale_and_crop")
            apply_to  = [conf.pop('apply_to') for conf in ds_config]
            nb_inputs = (np.array(apply_to) == 0).sum()
            transforms.add(ReplicateTensorsInBatch(apply_to))
            for indx, conf in enumerate(ds_config):
                transforms.add(DownSampleAndCropTensorsInBatch(apply_to=[indx], order=None, **conf))

        # Check if to compute binary-affinity-targets from GT labels:
        if self.transform_config.get("affinity_config") is not None:
            affs_config = deepcopy(self.transform_config.get("affinity_config"))
            global_kwargs = affs_config.pop("global", {})

            aff_transform = Segmentation2AffinitiesDynamicOffsets if affs_config.pop("use_dynamic_offsets", False) \
                else affinity_config_to_transform

            for input_index in affs_config:
                affs_kwargs = deepcopy(global_kwargs)
                affs_kwargs.update(affs_config[input_index])
                transforms.add(aff_transform(apply_to=[input_index+nb_inputs], **affs_kwargs))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.transform_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        transforms.add(AsTorchBatch(3))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        transform_config = config.get('transform_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   transform_config=transform_config)
